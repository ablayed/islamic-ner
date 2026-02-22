"""Run CAMeLBERT-CA ablation and build AraBERT/CAMeLBERT comparison table.

This script mirrors notebook 05 training settings for fair comparison.
"""

from __future__ import annotations

import inspect
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "data").exists() and (candidate / "models").exists():
            return candidate
    return start


ROOT = find_project_root(Path(__file__).resolve().parents[1])
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

SILVER_TRAIN_PATH = ROOT / "data" / "silver" / "train.json"
SILVER_DEV_PATH = ROOT / "data" / "silver" / "dev.json"

CAMEL_MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-ca"
MAX_SEQ_LENGTH = 128
CHECKPOINT_STEPS = 50

ARABERT_STANDARD_RUN_DIR = ROOT / "models" / "islamic_ner_standard"
ARABERT_WEIGHTED_RUN_DIR = ROOT / "models" / "islamic_ner_weighted"
ARABERT_STANDARD_FINAL = ARABERT_STANDARD_RUN_DIR / "final_model"
ARABERT_WEIGHTED_FINAL = ARABERT_WEIGHTED_RUN_DIR / "final_model"

CAMEL_RUN_DIR = ROOT / "models" / "islamic_ner_camelbert_ca"
CAMEL_RUN_DIR.mkdir(parents=True, exist_ok=True)

labels = [
    "O",
    "B-SCHOLAR",
    "I-SCHOLAR",
    "B-BOOK",
    "I-BOOK",
    "B-CONCEPT",
    "I-CONCEPT",
    "B-PLACE",
    "I-PLACE",
    "B-HADITH_REF",
    "I-HADITH_REF",
]

label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}
seqeval_metric = evaluate.load("seqeval")


def load_silver_split(path: Path) -> Dataset:
    records = json.loads(path.read_text(encoding="utf-8"))
    cleaned = []
    for i, record in enumerate(records):
        tokens = record.get("tokens") or []
        tags = record.get("ner_tags") or []
        if not isinstance(tokens, list) or not isinstance(tags, list):
            continue
        n = min(len(tokens), len(tags))
        if n == 0:
            continue
        cleaned.append(
            {
                "id": record.get("id", f"{path.stem}_{i}"),
                "tokens": tokens[:n],
                "ner_tags": tags[:n],
            }
        )
    return Dataset.from_list(cleaned)


def load_dataset_dict() -> DatasetDict:
    dataset_dict = DatasetDict(
        {
            "train": load_silver_split(SILVER_TRAIN_PATH),
            "dev": load_silver_split(SILVER_DEV_PATH),
        }
    )

    def count_labels(split_dataset: Dataset) -> Counter:
        counts = Counter()
        for row in split_dataset:
            counts.update(row["ner_tags"])
        return counts

    train_counts = count_labels(dataset_dict["train"])
    dev_counts = count_labels(dataset_dict["dev"])
    unknown_train = sorted(set(train_counts) - set(labels))
    unknown_dev = sorted(set(dev_counts) - set(labels))
    assert not unknown_train, f"Unknown labels in train: {unknown_train}"
    assert not unknown_dev, f"Unknown labels in dev: {unknown_dev}"
    return dataset_dict


def tokenize_and_align_labels_for_tokenizer(
    examples: Dict[str, List[List[str]]], tokenizer
) -> Dict[str, List[List[int]]]:
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=MAX_SEQ_LENGTH,
    )

    aligned_labels = []
    for i, word_level_tags in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[word_level_tags[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        aligned_labels.append(label_ids)

    tokenized["labels"] = aligned_labels
    return tokenized


def decode_predictions(
    pred_ids: np.ndarray, label_ids: np.ndarray
) -> Tuple[List[List[str]], List[List[str]]]:
    y_true: List[List[str]] = []
    y_pred: List[List[str]] = []

    for pred_row, label_row in zip(pred_ids, label_ids):
        row_true: List[str] = []
        row_pred: List[str] = []
        for pred_id, label_id in zip(pred_row, label_row):
            if int(label_id) == -100:
                continue
            row_true.append(id2label[int(label_id)])
            row_pred.append(id2label[int(pred_id)])
        y_true.append(row_true)
        y_pred.append(row_pred)

    return y_true, y_pred


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels_arr = eval_pred
    pred_ids = np.argmax(logits, axis=2)
    y_true, y_pred = decode_predictions(pred_ids, labels_arr)

    scores = seqeval_metric.compute(predictions=y_pred, references=y_true)
    return {
        "precision": float(scores["overall_precision"]),
        "recall": float(scores["overall_recall"]),
        "f1": float(scores["overall_f1"]),
        "accuracy": float(scores["overall_accuracy"]),
    }


def evaluate_and_report(trainer: Trainer, dataset, run_name: str, verbose: bool = True):
    pred_output = trainer.predict(dataset)
    pred_ids = np.argmax(pred_output.predictions, axis=2)
    y_true, y_pred = decode_predictions(pred_ids, pred_output.label_ids)

    overall = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    report_text = classification_report(y_true, y_pred, digits=4, zero_division=0)
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    if verbose:
        print(f"[{run_name}] Overall Precision: {overall['precision']:.4f}")
        print(f"[{run_name}] Overall Recall:    {overall['recall']:.4f}")
        print(f"[{run_name}] Overall F1:        {overall['f1']:.4f}\n")
        print(report_text)

    return overall, report_dict, pred_output


def make_training_args(output_dir: Path) -> TrainingArguments:
    kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": 5,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "learning_rate": 3e-5,
        "weight_decay": 0.01,
        # Step-based saves make resume robust when long CPU runs are interrupted.
        "save_strategy": "steps",
        "save_steps": CHECKPOINT_STEPS,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "logging_steps": 25,
        "seed": SEED,
        "report_to": "none",
    }

    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "steps"
        kwargs["eval_steps"] = CHECKPOINT_STEPS
    else:
        kwargs["evaluation_strategy"] = "steps"
        kwargs["eval_steps"] = CHECKPOINT_STEPS
    return TrainingArguments(**kwargs)


def init_model(model_path_or_name: str):
    kwargs = {}
    if model_path_or_name == CAMEL_MODEL_NAME:
        # Avoid network dependency after first download.
        kwargs["local_files_only"] = True
    return AutoModelForTokenClassification.from_pretrained(
        model_path_or_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        **kwargs,
    )


def find_resumable_checkpoint(output_dir: Path) -> str | None:
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        try:
            step = int(path.name.split("-")[-1])
        except ValueError:
            continue
        required = [
            path / "model.safetensors",
            path / "trainer_state.json",
            path / "optimizer.pt",
            path / "scheduler.pt",
            path / "training_args.bin",
        ]
        if all(p.exists() for p in required):
            checkpoints.append((step, path))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda item: item[0])
    return str(checkpoints[-1][1])


def load_metrics_from_run_dir(run_dir: Path):
    overall_path = run_dir / "dev_overall_metrics.json"
    report_path = run_dir / "dev_classification_report.json"
    if overall_path.exists() and report_path.exists():
        return (
            json.loads(overall_path.read_text(encoding="utf-8")),
            json.loads(report_path.read_text(encoding="utf-8")),
        )
    return None, None


def evaluate_saved_model(model_dir: Path, run_name: str, dataset_dict: DatasetDict):
    if not model_dir.exists() or not (model_dir / "config.json").exists():
        print(f"[{run_name}] model not found at: {model_dir}")
        return None, None

    model_tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    tokenized_dev = dataset_dict["dev"].map(
        lambda examples: tokenize_and_align_labels_for_tokenizer(examples, model_tokenizer),
        batched=True,
        remove_columns=dataset_dict["dev"].column_names,
        desc=f"Tokenize dev ({run_name})",
    )

    model = init_model(str(model_dir))
    eval_args = TrainingArguments(
        output_dir=str(ROOT / "models" / "_tmp_eval" / run_name.replace(" ", "_").replace("(", "").replace(")", "")),
        per_device_eval_batch_size=16,
        report_to="none",
        seed=SEED,
    )

    eval_trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=tokenized_dev,
        tokenizer=model_tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=model_tokenizer),
        compute_metrics=compute_metrics,
    )

    overall, report, _ = evaluate_and_report(eval_trainer, tokenized_dev, run_name, verbose=False)
    print(f"[{run_name}] overall F1: {overall['f1']:.4f}")
    return overall, report


def ensure_metrics(run_dir: Path, model_dir: Path, run_name: str, dataset_dict: DatasetDict):
    overall, report = load_metrics_from_run_dir(run_dir)
    if overall is not None and report is not None:
        print(f"[{run_name}] loaded saved metrics from {run_dir}")
        return overall, report

    print(f"[{run_name}] saved metrics missing; evaluating saved model artifact instead.")
    overall, report = evaluate_saved_model(model_dir, run_name, dataset_dict)
    if overall is not None and report is not None:
        (run_dir / "dev_overall_metrics.json").write_text(
            json.dumps(overall, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (run_dir / "dev_classification_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return overall, report


def extract_overall_f1(overall: Dict | None) -> float:
    if not overall:
        return float("nan")
    if "f1" in overall:
        return float(overall["f1"])
    if "eval_f1" in overall:
        return float(overall["eval_f1"])
    return float("nan")


def extract_entity_f1(report: Dict | None, entity_name: str) -> float:
    if not report:
        return float("nan")
    entity_metrics = report.get(entity_name, {})
    if isinstance(entity_metrics, dict) and "f1-score" in entity_metrics:
        return float(entity_metrics["f1-score"])
    return float("nan")


def extract_macro_f1(report: Dict | None) -> float:
    if not report:
        return float("nan")
    macro = report.get("macro avg", {})
    if isinstance(macro, dict) and "f1-score" in macro:
        return float(macro["f1-score"])
    return float("nan")


def winner_for(comparison_df: pd.DataFrame, metric_name: str, model_cols: List[str]) -> str:
    row = comparison_df[comparison_df["Metric"] == metric_name]
    if row.empty:
        return "N/A"
    series = pd.to_numeric(row.iloc[0][model_cols], errors="coerce")
    if series.dropna().empty:
        return "N/A"
    return str(series.idxmax())


def main() -> int:
    print(f"Project root: {ROOT}")
    print(f"Torch device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"CAMeLBERT model: {CAMEL_MODEL_NAME}")

    dataset_dict = load_dataset_dict()
    print(dataset_dict)

    resume_ckpt = find_resumable_checkpoint(CAMEL_RUN_DIR)
    tokenizer_source = CAMEL_MODEL_NAME
    if resume_ckpt:
        ckpt_path = Path(resume_ckpt)
        if (ckpt_path / "tokenizer.json").exists() and (ckpt_path / "tokenizer_config.json").exists():
            tokenizer_source = resume_ckpt
        print("Resuming CAMeLBERT training from:", resume_ckpt)
        print("Tokenizer source:", tokenizer_source)
    else:
        print("No resumable CAMeLBERT checkpoint found; starting from scratch.")

    camel_tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=True)
    camel_data_collator = DataCollatorForTokenClassification(tokenizer=camel_tokenizer)
    tokenized_camel = dataset_dict.map(
        lambda examples: tokenize_and_align_labels_for_tokenizer(examples, camel_tokenizer),
        batched=True,
        remove_columns=dataset_dict["train"].column_names,
        desc="Tokenize + align labels (CAMeLBERT-CA)",
    )

    # Initialize from checkpoint when resuming to avoid re-initializing the classifier head.
    camel_model_source = resume_ckpt or CAMEL_MODEL_NAME
    camel_model = init_model(camel_model_source)
    camel_args = make_training_args(CAMEL_RUN_DIR)
    trainer_camel = Trainer(
        model=camel_model,
        args=camel_args,
        train_dataset=tokenized_camel["train"],
        eval_dataset=tokenized_camel["dev"],
        tokenizer=camel_tokenizer,
        data_collator=camel_data_collator,
        compute_metrics=compute_metrics,
    )

    # Fix non-contiguous tensor saving issue
    for param in camel_model.parameters():
        param.data = param.data.contiguous()

    if resume_ckpt:
        train_result_camel = trainer_camel.train(resume_from_checkpoint=resume_ckpt)
    else:
        train_result_camel = trainer_camel.train()

    eval_result_camel = trainer_camel.evaluate(tokenized_camel["dev"])
    print("CAMeLBERT-CA train metrics:", train_result_camel.metrics)
    print("CAMeLBERT-CA eval metrics:", eval_result_camel)

    overall_camel, report_camel, _ = evaluate_and_report(
        trainer_camel, tokenized_camel["dev"], "CAMeLBERT-CA", verbose=True
    )

    camel_final_dir = CAMEL_RUN_DIR / "final_model"
    camel_final_dir.mkdir(parents=True, exist_ok=True)
    trainer_camel.save_model(str(camel_final_dir))
    camel_tokenizer.save_pretrained(str(camel_final_dir))

    (CAMEL_RUN_DIR / "dev_overall_metrics.json").write_text(
        json.dumps(overall_camel, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (CAMEL_RUN_DIR / "dev_classification_report.json").write_text(
        json.dumps(report_camel, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (CAMEL_RUN_DIR / "run_camel_train_metrics.json").write_text(
        json.dumps(train_result_camel.metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (CAMEL_RUN_DIR / "run_camel_eval_metrics.json").write_text(
        json.dumps(eval_result_camel, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved CAMeLBERT-CA model and metrics under: {CAMEL_RUN_DIR}")

    arabert_standard_overall, arabert_standard_report = ensure_metrics(
        ARABERT_STANDARD_RUN_DIR, ARABERT_STANDARD_FINAL, "AraBERT (standard)", dataset_dict
    )
    arabert_weighted_overall, arabert_weighted_report = ensure_metrics(
        ARABERT_WEIGHTED_RUN_DIR, ARABERT_WEIGHTED_FINAL, "AraBERT (weighted)", dataset_dict
    )

    comparison_rows = [
        {
            "Metric": "Overall F1",
            "AraBERT (standard)": extract_overall_f1(arabert_standard_overall),
            "AraBERT (weighted)": extract_overall_f1(arabert_weighted_overall),
            "CAMeLBERT-CA": extract_overall_f1(overall_camel),
        },
        {
            "Metric": "SCHOLAR F1",
            "AraBERT (standard)": extract_entity_f1(arabert_standard_report, "SCHOLAR"),
            "AraBERT (weighted)": extract_entity_f1(arabert_weighted_report, "SCHOLAR"),
            "CAMeLBERT-CA": extract_entity_f1(report_camel, "SCHOLAR"),
        },
        {
            "Metric": "BOOK F1",
            "AraBERT (standard)": extract_entity_f1(arabert_standard_report, "BOOK"),
            "AraBERT (weighted)": extract_entity_f1(arabert_weighted_report, "BOOK"),
            "CAMeLBERT-CA": extract_entity_f1(report_camel, "BOOK"),
        },
        {
            "Metric": "CONCEPT F1",
            "AraBERT (standard)": extract_entity_f1(arabert_standard_report, "CONCEPT"),
            "AraBERT (weighted)": extract_entity_f1(arabert_weighted_report, "CONCEPT"),
            "CAMeLBERT-CA": extract_entity_f1(report_camel, "CONCEPT"),
        },
        {
            "Metric": "PLACE F1",
            "AraBERT (standard)": extract_entity_f1(arabert_standard_report, "PLACE"),
            "AraBERT (weighted)": extract_entity_f1(arabert_weighted_report, "PLACE"),
            "CAMeLBERT-CA": extract_entity_f1(report_camel, "PLACE"),
        },
        {
            "Metric": "HADITH_REF F1",
            "AraBERT (standard)": extract_entity_f1(arabert_standard_report, "HADITH_REF"),
            "AraBERT (weighted)": extract_entity_f1(arabert_weighted_report, "HADITH_REF"),
            "CAMeLBERT-CA": extract_entity_f1(report_camel, "HADITH_REF"),
        },
        {
            "Metric": "Macro F1",
            "AraBERT (standard)": extract_macro_f1(arabert_standard_report),
            "AraBERT (weighted)": extract_macro_f1(arabert_weighted_report),
            "CAMeLBERT-CA": extract_macro_f1(report_camel),
        },
    ]

    comparison_df = pd.DataFrame(comparison_rows)
    model_cols = ["AraBERT (standard)", "AraBERT (weighted)", "CAMeLBERT-CA"]
    for col in model_cols:
        comparison_df[col] = pd.to_numeric(comparison_df[col], errors="coerce").round(4)

    comparison_csv_path = ROOT / "models" / "islamic_ner_ablation_comparison_with_camelbert.csv"
    comparison_json_path = ROOT / "models" / "islamic_ner_ablation_comparison_with_camelbert.json"
    comparison_payload = {
        "camelbert_model_name": CAMEL_MODEL_NAME,
        "max_seq_length": MAX_SEQ_LENGTH,
        "arabert_standard_run_dir": str(ARABERT_STANDARD_RUN_DIR),
        "arabert_weighted_run_dir": str(ARABERT_WEIGHTED_RUN_DIR),
        "camelbert_run_dir": str(CAMEL_RUN_DIR),
        "table": comparison_rows,
    }
    comparison_df.to_csv(comparison_csv_path, index=False)
    comparison_json_path.write_text(
        json.dumps(comparison_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nComparison table:")
    print(comparison_df.to_string(index=False))
    print("\nSaved comparison files:")
    print("-", comparison_csv_path)
    print("-", comparison_json_path)
    try:
        print("\nMarkdown table:")
        print(comparison_df.to_markdown(index=False))
    except Exception:
        pass

    overall_winner = winner_for(comparison_df, "Overall F1", model_cols)
    scholar_winner = winner_for(comparison_df, "SCHOLAR F1", model_cols)
    concept_winner = winner_for(comparison_df, "CONCEPT F1", model_cols)
    macro_winner = winner_for(comparison_df, "Macro F1", model_cols)

    rare_metrics = ["BOOK F1", "PLACE F1", "HADITH_REF F1"]
    rare_score = {
        model: np.nanmean(
            [float(comparison_df.loc[comparison_df["Metric"] == metric, model].iloc[0]) for metric in rare_metrics]
        )
        for model in model_cols
    }
    rare_winner = max(rare_score, key=lambda k: (-np.inf if pd.isna(rare_score[k]) else rare_score[k]))

    print("\nQuick winners:")
    print("- Overall winner:", overall_winner)
    print("- SCHOLAR winner:", scholar_winner)
    print("- CONCEPT winner:", concept_winner)
    print("- Macro-F1 winner:", macro_winner)
    print("- Rare-class mean winner (BOOK/PLACE/HADITH_REF):", rare_winner)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
