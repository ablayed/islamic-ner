# Domain-Specific Named Entity Recognition for Arabic Islamic Texts: A Pipeline Approach

## Abstract
Arabic NLP has strong general-purpose models, but domain performance drops when applied to classical Islamic texts where narration chains, historical name morphology, and citation structure differ from modern corpora. This project builds an end-to-end pipeline for Arabic Islamic Named Entity Recognition (NER) and downstream knowledge graph construction. The system combines Arabic normalization, weakly supervised silver data generation, AraBERT fine-tuning, rule-based relation extraction, and Neo4j graph insertion with entity resolution. On a hand-annotated gold set of 200 sentences, the best model achieved 91.32% F1. The project demonstrates practical applied AI engineering: data bootstrapping under low supervision, domain-specific modeling decisions, deployment with API/demo interfaces, and transparent error analysis.

## 1. Introduction
Islamic textual corpora are large, structurally rich, and historically important, but difficult to process with generic Arabic NLP tools. Hadith texts in particular contain repeating isnad patterns, long narrator names, and domain terms that are underrepresented in mainstream benchmarks. As a result, off-the-shelf NER models trained on news-style datasets often fail to separate narrators, works, concepts, and references reliably in this domain.

The practical need is clear: researchers, digital library teams, and students benefit from machine-readable metadata that links scholars, books, and concepts across texts. But building high-quality supervised datasets for this domain is expensive because annotation requires both language competence and subject-matter awareness.

This work addresses that gap through a pipeline design optimized for limited annotation budgets. Instead of waiting for a fully labeled corpus, I used weak supervision to generate silver training data, then validated the resulting model on a manually corrected gold benchmark. I extended extraction beyond entity spans by adding relation rules and graph construction so outputs are directly queryable.

The main contribution is not only a model score, but an operational system: text input to entity extraction, relation inference, graph insertion, and interactive API/demo consumption.

## 2. Approach

### 2.1 Data Strategy: Weak Supervision + Gold Validation
The project uses a two-tier data strategy. First, I bootstrapped silver labels from large hadith sources (including Sanadset 650K and hadith-json aligned inputs) using gazetteers and heuristics. This enabled fast scaling to thousands of training sentences while preserving domain schema control. Second, I built a hand-corrected gold set (200 sentences) for trustworthy evaluation.

This split balances throughput and quality: silver data supplies training volume, gold data provides measurement integrity. The schema includes five entity types: `SCHOLAR`, `BOOK`, `CONCEPT`, `PLACE`, and `HADITH_REF`.

### 2.2 Preprocessing and Model Training
Arabic normalization is applied before tokenization to reduce orthographic variance (diacritics, Alif forms, taa marbuta, alif maqsura, tatweel, spacing). This lowers lexical fragmentation and stabilizes token-classification behavior.

For modeling, I fine-tuned AraBERT v02 (`aubmindlab/bert-base-arabertv02`) as a token classifier. Training settings used AdamW with `3e-5` learning rate, `0.01` weight decay, batch size 16, and early stopping. The objective was strong precision/recall balance under noisy silver supervision.

### 2.3 Relation Extraction via Isnad-Aware Rules
Entity extraction alone is insufficient for retrieval and analysis; links are required. I implemented a rule-based relation extractor with four relation types:

- `NARRATED_FROM`
- `IN_BOOK`
- `MENTIONS_CONCEPT`
- `AUTHORED`

I chose deterministic rules because isnad syntax contains reliable directional patterns (for example, connectors like `عن`). Given limited relation annotations, rule-based extraction delivered a better precision-first MVP than training a low-resource relation classifier.

### 2.4 Knowledge Graph Construction
Extracted entities are normalized into canonical forms before insertion into Neo4j. The graph builder merges entity variants, inserts relation edges, and tracks source metadata. Query utilities support scholar connection lookup and narration-chain traversal.

This graph layer converts flat NER output into a navigable structure suitable for analysis and product demos.

### 2.5 Deployment Layer
I exposed the pipeline through a FastAPI backend with endpoints for NER extraction, graph build, and scholar query. A Streamlit demo was built for portfolio and stakeholder interaction, including:

- Arabic RTL text input
- color-coded entity highlights
- extracted relation tables
- graph visualization tab

The demo degrades gracefully when Neo4j is unavailable, so NER remains usable independently.

## 3. Results
The best model (AraBERT standard) achieved **91.32% F1** on the gold set.

### 3.1 Comparative Model Results (Gold)

| Model | Precision | Recall | F1 | Macro F1 |
|-------|-----------|--------|----|----------|
| AraBERT (standard) | 0.8964 | 0.9306 | **0.9132** | **0.7396** |
| CAMeLBERT-CA | 0.8845 | 0.9327 | 0.9080 | 0.6181 |
| AraBERT (weighted) | 0.7876 | 0.8915 | 0.8363 | 0.6062 |

AraBERT standard produced the best overall and macro performance, while weighted training underperformed due to reduced precision stability.

### 3.2 Per-Entity Highlights
- `CONCEPT`: 96.6% F1 (strong class separability)
- `SCHOLAR`: 91.9% F1 (high recall on common narrators)
- `HADITH_REF`: 60.0% F1 (low support)
- `PLACE`: 45.0% F1 (false-positive sensitivity)
- `BOOK`: no meaningful gold support in the sample

### 3.3 Error Analysis Insights
Error distribution identified five categories:
- Hallucinated entities: 47.5%
- Boundary errors: 31.2%
- Missed entities: 20.6%
- Type confusion: 0.7%
- Morphological errors: 0.0%

The dominant issue is false-positive hallucination in contexts with sparse entity density, especially `SCHOLAR` overprediction.

## 4. Discussion and Limitations
Several design choices worked well. Weak supervision plus targeted gold evaluation provided a practical path to strong domain performance without full manual labeling. The schema aligned well with the domain, and type confusion remained low. Rule-based relation extraction was effective for a first release because narration syntax is structurally constrained.

At the same time, limitations are explicit. `BOOK` and `HADITH_REF` classes are data-sparse. `PLACE` precision remains weak. Entity resolution is still mostly gazetteer-driven, which creates isolated nodes for rare variants. Coverage is sectarian/domain-bound (primarily Sunni canonical collections), so broader usage requires new annotation and validation passes.

Future improvements include classifier-based relation extraction, embedding-based entity linking, broader corpus coverage (fiqh/tafsir), and cross-collection graph alignment.

## 5. Conclusion
This project demonstrates an end-to-end applied AI workflow for a difficult low-resource domain: problem framing, weakly supervised data design, model training/evaluation, error analysis, and deployable interfaces. The final system is not only a trained NER model but a complete pipeline with API and demo-ready graph outputs. For MSc and industry contexts, the key evidence is engineering depth under realistic constraints: measurable performance, transparent limitations, and a clear roadmap for scaling quality and scope.
