# IPS-2014: Domain-Adaptive RAG for Requirements Engineering

Anonymized dataset and reproducible pipeline for the paper submitted to Computers in Industry.

## What this is
- IPS-2014: anonymized RE corpus from a confidential German manufacturing-equipment specification family (535 requirement statements, 49 pseudonymized suppliers, 37 component subcategories, synthetic cost layer).
- A reproducible pipeline that fine-tunes a multilingual sentence encoder on in-domain contrastive pairs and evaluates retrieval (MRR@5, P@5, NDCG@5) vs an off-the-shelf baseline.

## Layout
- data/ : anonymized IPS-2014 corpus
- src/ : anonymize_pipeline.py, rag_finetune_pipeline.py
- results/ : results_measured.json
- paper/ : main.tex, references.bib

## Reproduce
Install: sentence-transformers, faiss-cpu, scikit-learn, numpy, datasets, accelerate
Run: python src/rag_finetune_pipeline.py --requirements data/IPS-2014_requirements_anon.json --stage all
Fixed seed (42) makes the run deterministic.

## Notes
- Source OEM specifications are copyright-restricted and NOT redistributed; only anonymized derivative data is released.
- Supplier identities are pseudonymized; the re-identification key is withheld.
- The cost layer is synthetic and excluded from all metrics.
- Single-annotator labels; no inter-rater statistics claimed.

## Acknowledgement
Supported by the Deutsche Forschungsgemeinschaft (DFG), grant UY-56/5-1.
