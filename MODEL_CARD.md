# Model Card: IslamicNER

## Model Details

- Model: AraBERT v02 fine-tuned for token classification
- Base: `aubmindlab/bert-base-arabertv02`
- Task: Named Entity Recognition on Arabic Islamic texts
- Entity types: `SCHOLAR`, `BOOK`, `CONCEPT`, `PLACE`, `HADITH_REF`
- Training data: 3,603 silver-annotated sentences from Sanadset 650K + hadith-json
- Gold evaluation: 91.32% F1 on 200 hand-annotated sentences

## Intended Use

- Research support for Islamic studies and Arabic NLP
- Automated metadata extraction for digital hadith collections
- Educational tooling for students of hadith sciences
- Experiments in low-resource/domain-specific Arabic NER

## Out of Scope Uses

- This model is not a religious authority.
- It must not be used to determine hadith authenticity/grade.
- It must not be used to issue, validate, or automate religious rulings (fatawa).
- Outputs should be reviewed by qualified scholars for any scholarly or religious use.

## Limitations

- Training scope is primarily Sunni hadith collections (Kutub as-Sitta).
- `BOOK` and `HADITH_REF` have limited support in gold evaluation.
- Hallucinated `SCHOLAR` entities in non-isnad text are a major failure mode (47.5% of errors).
- Entity resolution is mostly gazetteer-based and can miss rare/novel name variants.
- Silver labels are weakly supervised with estimated noise around 20-30%.

## Ethical Considerations

- Misattribution risk: incorrect extraction can assign statements/chains to the wrong narrator.
- Sectarian scope: current data is Sunni-focused; extension to Shia collections needs separate data and review.
- Authority risk: users may over-trust model output despite uncertainty and domain sensitivity.
- Cultural sensitivity: hadith texts are religiously sensitive and should be handled respectfully.

## Training Details

- Optimizer: AdamW
- Learning rate: `3e-5`
- Weight decay: `0.01`
- Epochs: 5 with early stopping (`patience=2`)
- Batch size: 16
- Max sequence length: 128
- Hardware: local training environment (update with exact CPU/GPU and wall-clock time if needed)

## Evaluation

- Silver dev F1: 93.70%
- Gold test F1: 91.32%

### Per-Entity Performance (Gold)

| Entity | Precision | Recall | F1 | Support |
|--------|-----------|--------|----|---------|
| SCHOLAR | 0.889 | 0.949 | 0.919 | 465 |
| CONCEPT | 0.962 | 0.971 | 0.966 | 309 |
| HADITH_REF | 0.600 | 0.600 | 0.600 | 5 |
| PLACE | 0.391 | 0.529 | 0.450 | 17 |
| BOOK | 0.000 | 0.000 | 0.000 | 0 |

### Error Analysis Summary

- Hallucinated entities: 47.5%
- Boundary errors: 31.2%
- Missed entities: 20.6%
- Type confusion: 0.7%
- Morphological errors: 0.0%

## Responsible Use Statement

This model should be presented as a research assistant for text analysis, not as a source of religious verdicts. Any scholarly, educational, or public-facing usage should include human review and clear uncertainty communication.
