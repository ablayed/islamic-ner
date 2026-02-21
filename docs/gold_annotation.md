# Gold Test Annotation Workflow

Use this workflow to create a human-verified test set from silver held-out data.

## 1) Prepare 200-sentence annotation pack

```bash
python scripts/gold_annotation.py prepare --sample-size 200 --seed 42
```

This creates:

- `data/gold/gold_test.json` (sample template, same structure as silver)
- `data/gold/gold_test_tokens.csv` (token-level sheet; edit `gold_tag`)
- `data/gold/gold_test_sentence_meta.csv` (confidence/note/review flags)

## 2) Manually correct labels

Edit `data/gold/gold_test_tokens.csv`:

- Keep `token` unchanged.
- Update `gold_tag` only.
- Valid tags: `O`, `B-SCHOLAR`, `I-SCHOLAR`, `B-BOOK`, `I-BOOK`, `B-CONCEPT`, `I-CONCEPT`, `B-PLACE`, `I-PLACE`, `B-HADITH_REF`, `I-HADITH_REF`.

Optional sentence-level notes in `data/gold/gold_test_sentence_meta.csv`:

- `confidence`: numeric score (e.g. `1.0`, `0.7`)
- `note`: free text
- `reviewed`: `1` when sentence is complete

## 3) Build final gold JSON

```bash
python scripts/gold_annotation.py build --strict-reviewed
```

This writes:

- `data/gold/gold_test.json` (final corrected labels)
- `data/gold/gold_test_annotation_meta.json` (confidence/notes/review flags)

## 4) Validate final file

```bash
python scripts/gold_annotation.py validate --input-json data/gold/gold_test.json
```

Validation checks:

- token/tag length match
- tag vocabulary validity
- BIO transition consistency
