"""NER endpoint and reusable Arabic NER inference helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

import torch
from fastapi import APIRouter, Request

from api.schemas import Entity, NERRequest, NERResponse

ENTITY_TYPES = {"SCHOLAR", "BOOK", "CONCEPT", "PLACE", "HADITH_REF"}
MODEL_INPUT_KEYS = {"input_ids", "attention_mask", "token_type_ids"}

router = APIRouter(tags=["NER"])


@dataclass
class NERPipelineResult:
    normalized_text: str
    words: List[str]
    labels: List[str]
    confidences: List[float]
    offsets: List[Tuple[int, int]]
    entities: List[Entity]


def _normalize_label(label: str) -> str:
    if label == "O":
        return "O"
    if "-" not in label:
        return "O"

    prefix, entity_type = label.split("-", 1)
    prefix = prefix.upper()
    entity_type = entity_type.upper()
    if entity_type == "HADITH":
        entity_type = "HADITH_REF"

    if prefix not in {"B", "I"}:
        return "O"
    if entity_type not in ENTITY_TYPES:
        return "O"
    return f"{prefix}-{entity_type}"


def _repair_bio(labels: List[str]) -> List[str]:
    repaired: List[str] = []
    prev_entity_type = ""
    prev_is_entity = False

    for raw_label in labels:
        label = _normalize_label(raw_label)
        if label == "O":
            repaired.append("O")
            prev_entity_type = ""
            prev_is_entity = False
            continue

        prefix, entity_type = label.split("-", 1)
        if prefix == "I" and (not prev_is_entity or prev_entity_type != entity_type):
            repaired.append(f"B-{entity_type}")
        else:
            repaired.append(label)

        prev_entity_type = entity_type
        prev_is_entity = True

    return repaired


def _compute_word_offsets(text: str, words: List[str]) -> List[Tuple[int, int]]:
    offsets: List[Tuple[int, int]] = []
    cursor = 0

    for word in words:
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1

        start = text.find(word, cursor)
        if start == -1:
            start = text.find(word)
        if start == -1:
            start = cursor

        end = start + len(word)
        offsets.append((start, end))
        cursor = end

    return offsets


def _predict_word_labels_with_model(words: List[str], model: Any, tokenizer: Any) -> tuple[List[str], List[float]]:
    encoded = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    if not hasattr(encoded, "word_ids"):
        raise RuntimeError("Tokenizer must support `word_ids` for word-level decoding.")

    try:
        device = next(model.parameters()).device
    except Exception:
        device = torch.device("cpu")

    model_inputs = {
        key: value.to(device)
        for key, value in encoded.items()
        if key in MODEL_INPUT_KEYS
    }
    with torch.no_grad():
        logits = model(**model_inputs).logits[0]
        probabilities = torch.softmax(logits, dim=-1)
        prediction_ids = torch.argmax(logits, dim=-1)

    id2label = getattr(getattr(model, "config", object()), "id2label", {}) or {}
    word_ids = encoded.word_ids(batch_index=0)
    if word_ids is None:
        raise RuntimeError("Unable to decode word IDs from tokenizer output.")

    labels = ["O"] * len(words)
    confidences = [0.0] * len(words)
    seen_word_ids = set()

    for token_index, word_id in enumerate(word_ids):
        if word_id is None or word_id in seen_word_ids or word_id >= len(words):
            continue

        seen_word_ids.add(word_id)
        pred_id = int(prediction_ids[token_index].item())
        raw_label = str(id2label.get(pred_id, "O"))

        labels[word_id] = _normalize_label(raw_label)
        confidences[word_id] = float(probabilities[token_index, pred_id].item())

    return _repair_bio(labels), confidences


def _predict_word_labels_with_gazetteer(
    normalized_text: str,
    words: List[str],
    offsets: List[Tuple[int, int]],
    gazetteer_matcher: Any,
) -> tuple[List[str], List[float]]:
    labels = ["O"] * len(words)
    confidences = [0.0] * len(words)
    if gazetteer_matcher is None:
        return labels, confidences

    matches = gazetteer_matcher.match(normalized_text)
    if not matches:
        return labels, confidences

    for match in matches:
        entity_type = str(match.get("entity_type", "")).upper()
        if entity_type == "HADITH":
            entity_type = "HADITH_REF"
        if entity_type not in ENTITY_TYPES:
            continue

        start_char = int(match.get("start", -1))
        end_char = int(match.get("end", -1))
        if start_char < 0 or end_char <= start_char:
            continue

        overlap_indexes: List[int] = []
        for index, (word_start, word_end) in enumerate(offsets):
            has_overlap = word_start < end_char and word_end > start_char
            if has_overlap:
                overlap_indexes.append(index)

        if not overlap_indexes:
            continue

        for local_index, word_index in enumerate(overlap_indexes):
            prefix = "B" if local_index == 0 else "I"
            labels[word_index] = f"{prefix}-{entity_type}"
            confidences[word_index] = max(confidences[word_index], 0.75 if prefix == "B" else 0.7)

    return _repair_bio(labels), confidences


def _build_entity(
    words: List[str],
    offsets: List[Tuple[int, int]],
    entity_type: str,
    start_word_index: int,
    end_word_index: int,
    confidences: List[float],
) -> Entity:
    text = " ".join(words[start_word_index:end_word_index])
    start_char = offsets[start_word_index][0]
    end_char = offsets[end_word_index - 1][1]
    span_confidences = confidences[start_word_index:end_word_index] or [0.0]
    confidence = float(sum(span_confidences) / len(span_confidences))
    return Entity(
        text=text,
        type=entity_type,
        start=start_char,
        end=end_char,
        confidence=round(confidence, 4),
    )


def _labels_to_entities(
    words: List[str],
    labels: List[str],
    offsets: List[Tuple[int, int]],
    confidences: List[float],
) -> List[Entity]:
    entities: List[Entity] = []
    current_type = ""
    span_start = -1

    for index, label in enumerate(labels):
        if label == "O":
            if current_type and span_start >= 0:
                entities.append(
                    _build_entity(
                        words=words,
                        offsets=offsets,
                        entity_type=current_type,
                        start_word_index=span_start,
                        end_word_index=index,
                        confidences=confidences,
                    )
                )
            current_type = ""
            span_start = -1
            continue

        prefix, entity_type = label.split("-", 1)

        if prefix == "B":
            if current_type and span_start >= 0:
                entities.append(
                    _build_entity(
                        words=words,
                        offsets=offsets,
                        entity_type=current_type,
                        start_word_index=span_start,
                        end_word_index=index,
                        confidences=confidences,
                    )
                )
            current_type = entity_type
            span_start = index
            continue

        # I- tag
        if current_type != entity_type or span_start < 0:
            if current_type and span_start >= 0:
                entities.append(
                    _build_entity(
                        words=words,
                        offsets=offsets,
                        entity_type=current_type,
                        start_word_index=span_start,
                        end_word_index=index,
                        confidences=confidences,
                    )
                )
            current_type = entity_type
            span_start = index

    if current_type and span_start >= 0:
        entities.append(
            _build_entity(
                words=words,
                offsets=offsets,
                entity_type=current_type,
                start_word_index=span_start,
                end_word_index=len(words),
                confidences=confidences,
            )
        )

    return entities


def run_ner_pipeline(
    text: str,
    model: Any,
    tokenizer: Any,
    normalizer: Any,
    gazetteer_matcher: Any = None,
) -> NERPipelineResult:
    normalized_text = normalizer.normalize(text)
    words = normalized_text.split()
    if not words:
        return NERPipelineResult(
            normalized_text=normalized_text,
            words=[],
            labels=[],
            confidences=[],
            offsets=[],
            entities=[],
        )

    offsets = _compute_word_offsets(normalized_text, words)
    if model is not None and tokenizer is not None:
        try:
            labels, confidences = _predict_word_labels_with_model(words, model, tokenizer)
        except Exception:
            labels, confidences = _predict_word_labels_with_gazetteer(
                normalized_text=normalized_text,
                words=words,
                offsets=offsets,
                gazetteer_matcher=gazetteer_matcher,
            )
    else:
        labels, confidences = _predict_word_labels_with_gazetteer(
            normalized_text=normalized_text,
            words=words,
            offsets=offsets,
            gazetteer_matcher=gazetteer_matcher,
        )

    labels = _repair_bio(labels)
    entities = _labels_to_entities(words=words, labels=labels, offsets=offsets, confidences=confidences)
    return NERPipelineResult(
        normalized_text=normalized_text,
        words=words,
        labels=labels,
        confidences=confidences,
        offsets=offsets,
        entities=entities,
    )


def run_ner_inference(text: str, model: Any, tokenizer: Any, normalizer: Any) -> List[Entity]:
    """
    1. Normalize text
    2. Split into words (whitespace tokenization for Arabic)
    3. Tokenize with AraBERT (is_split_into_words=True)
    4. Run model forward pass (torch.no_grad())
    5. Get predicted label ids via argmax
    6. Map subword predictions back to word-level labels
       (take first subword prediction for each word, skip -100 positions)
    7. Convert BIO labels to entity spans
    8. Compute confidence from softmax probabilities
    9. Return list of Entity objects with text, type, start, end, confidence
    """

    result = run_ner_pipeline(
        text=text,
        model=model,
        tokenizer=tokenizer,
        normalizer=normalizer,
    )
    return result.entities


def _build_token_payload(pipeline: NERPipelineResult) -> List[dict[str, Any]]:
    rows: List[dict[str, Any]] = []
    for index, word in enumerate(pipeline.words):
        start, end = pipeline.offsets[index]
        rows.append(
            {
                "index": index,
                "token": word,
                "label": pipeline.labels[index],
                "confidence": round(float(pipeline.confidences[index]), 4),
                "start": start,
                "end": end,
            }
        )
    return rows


@router.post("/ner", response_model=NERResponse)
async def extract_entities(request: NERRequest, fastapi_request: Request):
    """
    Extract named entities from Arabic Islamic text.

    Pipeline:
    1. Normalize input text
    2. Tokenize with AraBERT tokenizer
    3. Run model inference
    4. Post-process: merge subword tokens, extract entity spans
    5. Return entities with types and confidence scores
    """

    state = fastapi_request.app.state
    pipeline = run_ner_pipeline(
        text=request.text,
        model=getattr(state, "model", None),
        tokenizer=getattr(state, "tokenizer", None),
        normalizer=state.normalizer,
        gazetteer_matcher=getattr(state, "gazetteer_matcher", None),
    )

    return NERResponse(
        text=request.text,
        normalized_text=pipeline.normalized_text,
        entities=pipeline.entities,
        tokens=_build_token_payload(pipeline) if request.return_tokens else None,
    )
