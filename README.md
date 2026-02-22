# islamic-ner

Arabic NER project for Islamic text, focused on five entity types:
`SCHOLAR`, `BOOK`, `CONCEPT`, `PLACE`, and `HADITH_REF`.

## Current Status

This repository is now in a documented Phase 3 state with:

- silver data generation and boosting completed
- AraBERT standard and weighted training completed
- CAMeLBERT-CA ablation completed
- hand-corrected 200-sentence gold set completed and validated
- gold evaluation completed for all three final models
- dedicated gold error-analysis notebook completed
- technical interpretation paragraphs finalized
- rule-based relation extraction module completed (Task 1 MVP)
- entity normalization + graph construction/query stack completed (Tasks 2/3 MVP)
- relation/graph unit tests added and passing

## Entity Schema

- `SCHOLAR`: person names in sanad/matn context
- `BOOK`: hadith collection/work titles
- `CONCEPT`: Islamic concepts/terms
- `PLACE`: geographic names
- `HADITH_REF`: structural/citation references

## Relation Extraction Strategy (MVP)

For this dataset, rule-based relation extraction is the right first implementation: isnad chains follow rigid, stable syntax (for example, `حدثنا X عن Y عن Z`) where connector patterns encode direction explicitly. With limited labeled relation data, a trained RE classifier would require annotation volume that is currently unavailable, while deterministic patterns already recover core isnad structure at high precision. This repo therefore uses a rule-based RE engine for MVP and keeps classifier-based RE as a future extension once a dedicated relation-annotation set is available.

## Completed Work (So Far)

1. **Data ingestion and preprocessing**
- Download scripts for hadith-json and Open-Hadith-Data (`scripts/download_data.py`)
- Sanadset download helper (`scripts/download_sanadset.py`)
- Arabic normalization/tokenization and gazetteer-driven annotation in `src/`

2. **Silver dataset pipeline**
- Silver generation pipeline (`scripts/generate_silver_data.py`)
- Silver boost pipeline for gazetteer refresh + targeted augmentation (`scripts/boost_silver_data.py`)
- Generated splits under `data/silver/` (`train.json`, `dev.json`, `test_held_out.json`)

3. **Training and ablation**
- Main NER training workflow in notebook `notebooks/05_islamic_ner_training.ipynb`
- CAMeLBERT ablation in notebook `notebooks/06_ablation_camelbert.ipynb`
- Scripted ablation runner: `scripts/run_camelbert_ablation.py`
- Added non-contiguous tensor fix before training in ablation script:
```python
for param in camel_model.parameters():
    param.data = param.data.contiguous()
```

4. **Gold annotation and validation**
- Gold annotation workflow script: `scripts/gold_annotation.py`
- Gold documentation: `docs/gold_annotation.md`
- Final validated gold file: `data/gold/gold_test.json` (200 sentences)

5. **Gold evaluation and error analysis**
- Gold evaluation results saved: `models/gold_evaluation_results.json`
- Error analysis notebook created: `notebooks/07_error_analysis.ipynb`
- Interpretation write-up saved: `interpretation.txt`

6. **Rule-based relation extraction (Task 1)**
- Implemented extractor: `src/relations/extract.py` (`RelationExtractor`)
- Supported relation types: `NARRATED_FROM`, `IN_BOOK`, `MENTIONS_CONCEPT`, `AUTHORED`
- Added tests: `tests/test_relations.py` (narration chain, book relation, concept mention, authorship, full pipeline)
- Added package export: `src/relations/__init__.py`
- Ran directional sanity check on 5 real gold hadiths: narration edges were ordered correctly (`earlier narrator -> later narrator`)

7. **Entity normalization + knowledge graph MVP (Tasks 2/3)**
- Implemented resolver: `src/graph/entity_resolver.py` (`EntityResolver`)
- Implemented graph builder: `src/graph/builder.py` (`KnowledgeGraphBuilder`)
- Implemented query helpers: `src/graph/query.py` (`GraphQuerier`)
- Added package export: `src/graph/__init__.py`
- Added graph/resolver tests: `tests/test_graph.py`
- Added deterministic resolver fixtures: `tests/fixtures/resolver_gazetteers/`

## Model Results

### Silver Dev (Overall F1)

| Model | F1 |
|---|---:|
| AraBERT (standard) | 0.9370 |
| CAMeLBERT-CA | 0.9287 |
| AraBERT (weighted) | 0.8720 |

### Gold Test (200 Sentences)

Source: `models/gold_evaluation_results.json`

| Model | Precision | Recall | F1 | Macro F1 |
|---|---:|---:|---:|---:|
| AraBERT (standard) | 0.8964 | 0.9306 | **0.9132** | **0.7396** |
| CAMeLBERT-CA | 0.8845 | 0.9327 | 0.9080 | 0.6181 |
| AraBERT (weighted) | 0.7876 | 0.8915 | 0.8363 | 0.6062 |

**Best model on gold:** AraBERT (standard)

### Per-Entity Gold Notes (Best Model)

- `CONCEPT`: near-perfect (~96% F1)
- `SCHOLAR`: strong (~92% F1)
- `HADITH_REF`: weak (~60% F1), very low support
- `PLACE`: weak (~45% F1), over-prediction behavior
- `BOOK`: not measurable in this gold sample (0 support)

## Error Analysis Summary (Gold)

From `notebooks/07_error_analysis.ipynb`:

| Error Type | Share |
|---|---:|
| Hallucinated entities | 47.5% |
| Boundary errors | 31.2% |
| Missed entities | 20.6% |
| Type confusion | 0.7% |
| Morphological errors | 0.0% |

Additional signal:

- 110 `O` tokens predicted as `SCHOLAR` in confusion analysis (main false-positive driver)
- `PLACE` precision is a known pain point (aggressive over-prediction)

## Interpretation (Technical Report Text)

Hallucinated entities are the dominant failure mode, accounting for 47.5% of all errors. These are false positives where non-entity text is labeled as an entity, most often SCHOLAR, and the confusion matrix shows this clearly with 110 O-tokens predicted as SCHOLAR. The root cause is dataset bias: silver training data is isnad-heavy, so the model overgeneralizes name-dense narration patterns into contexts where entities are actually sparse. A practical fix is to add more matn-focused negative examples, apply hard-negative mining for SCHOLAR/PLACE (training on cases the model is most likely to get wrong), and enforce a confidence threshold so low-confidence spans are suppressed at inference time.

Boundary errors are the second largest category at 31.2%, where the model detects an entity but chooses the wrong span limits. In practice this appears as over-extension (including honorifics or trailing words) and under-extension (cutting patronymic chains too early), such as predicting "Muhammad ibn Sahl ibn Abi Hathma" when gold ends at "Muhammad ibn Sahl ibn Abi". The root cause is Arabic name morphology and variable nasab length, especially when honorific phrases appear immediately after names, and this is a well-documented challenge in Arabic NER literature rather than a project-specific anomaly. This can be improved with targeted boundary supervision, post-processing rules that strip honorific tails, and additional gold examples of long multi-token scholar names.

Missed entities (false negatives) make up 20.6% of errors, where a gold entity is present but the model predicts O. These misses are concentrated in less frequent names (long-tail scholar mentions), indicating weak recall on sparse entities. The underlying issue is sparse exposure in silver data, consistent with the earlier finding that 70.3% of scholar names were singletons. To reduce this, expand rare-name coverage via lexicon-guided augmentation (for example, inserting known rare names into sentence templates to generate synthetic training examples), add semantically diverse contexts, and include more manually corrected gold examples for low-frequency entities; this is the same core idea as the Phase 2 book-boost strategy applied to scholar recall.

Type confusion is minimal at 0.7%, with only one observed confusion case between SCHOLAR and PLACE. In practical terms, this is 1 confusion out of 141 total errors, which strongly validates the entity schema design choices made in Phase 1. The likely cause in that single case is local context ambiguity rather than systematic label collapse. A lightweight fix is to add a few targeted disambiguation examples and context rules for ambiguous historical terms that can denote person, place, or event.

Morphological errors were not observed (0.0%), so Arabic-specific tokenization/normalization appears to be working as intended on this gold set. In other words, agglutination and diacritic-variant handling did not emerge as active failure drivers in the current evaluation sample. This validates the design decision to normalize text before model input rather than relying on the model alone to learn diacritic invariance. The right action is to preserve the current normalization pipeline and add regression tests so this strength is maintained as data and models evolve.

## Reproducible Commands

Install:

```bash
pip install -r requirements.txt
```

Download raw data:

```bash
python scripts/download_data.py
python scripts/download_sanadset.py
```

Generate and boost silver data:

```bash
python scripts/generate_silver_data.py
python scripts/boost_silver_data.py
```

Gold annotation workflow:

```bash
python scripts/gold_annotation.py prepare --sample-size 200 --seed 42
python scripts/gold_annotation.py build --strict-reviewed
python scripts/gold_annotation.py validate --input-json data/gold/gold_test.json
```

CAMeLBERT ablation script:

```bash
python scripts/run_camelbert_ablation.py
```

Relation extraction + graph tests:

```bash
pytest tests/test_relations.py tests/test_graph.py -q -p no:cacheprovider --basetemp=./tmp_pytest_rel_graph
```

Notebook map:

- `notebooks/01_eda.ipynb`: initial exploration
- `notebooks/02_baseline_anercorp.ipynb`: baseline model setup/eval
- `notebooks/03_sanadset_exploration.ipynb`: Sanadset-focused exploration
- `notebooks/04_silver_data_analysis.ipynb`: silver label distribution and quality checks
- `notebooks/05_islamic_ner_training.ipynb`: AraBERT standard/weighted training
- `notebooks/06_ablation_camelbert.ipynb`: CAMeLBERT-CA ablation and comparison
- `notebooks/07_error_analysis.ipynb`: gold inference, error taxonomy, confusion matrix, hardest sentences

## Key Artifacts

- `data/gold/gold_test.json`
- `models/islamic_ner_standard/final_model/`
- `models/islamic_ner_weighted/final_model/`
- `models/islamic_ner_camelbert_ca/final_model/`
- `models/islamic_ner_ablation_comparison.csv`
- `models/gold_evaluation_results.json`
- `interpretation.txt`

