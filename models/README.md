# Models Directory

Model artifacts are typically large and are excluded from Git in normal usage.

## Expected contents

- `models/islamic_ner_standard/final_model/`
- `models/islamic_ner_weighted/final_model/`
- `models/islamic_ner_camelbert_ca/final_model/`
- evaluation summaries (`*.json`, `*.csv`)

## Training / reproduction

Primary workflow notebooks:

- `notebooks/05_islamic_ner_training.ipynb`
- `notebooks/06_ablation_camelbert.ipynb`

Scripted ablation run:

```bash
python scripts/run_camelbert_ablation.py
```

After training, place final checkpoints under the expected paths above so API/demo inference can load them.

By default, API startup reads:

- `models/islamic_ner_standard/final_model`

You can override with:

- `MODEL_PATH=/path/to/final_model`
