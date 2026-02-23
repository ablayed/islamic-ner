# Data Directory

This repository does not include large raw datasets in version control.

## What is expected here

- `data/raw/`: downloaded source corpora (gitignored)
- `data/silver/`: generated weakly supervised training splits
- `data/gold/`: hand-corrected evaluation set
- `data/gazetteers/`: domain lexicons used in weak supervision and resolution

## Download raw data

From project root:

```bash
python scripts/download_data.py
python scripts/download_sanadset.py
```

## Generate silver data

```bash
python scripts/generate_silver_data.py
python scripts/boost_silver_data.py
```

## Gold annotation workflow

```bash
python scripts/gold_annotation.py prepare --sample-size 200 --seed 42
python scripts/gold_annotation.py build --strict-reviewed
python scripts/gold_annotation.py validate --input-json data/gold/gold_test.json
```

If files are missing, re-run the scripts above and verify permissions for this workspace.
