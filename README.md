# IPS-2014: Obligation-Aware Retrieval for Requirements Engineering

Anonymized dataset and reproducible pipeline for our work on failure-mode-targeted
retrieval over industrial manufacturing specifications.

## What this is
- **IPS-2014**: an anonymized RE corpus from a confidential German manufacturing-equipment
  specification family — 535 requirement statements, 49 pseudonymized suppliers,
  37 component subcategories, and a synthetic cost layer.
- A reproducible pipeline that (a) fine-tunes a multilingual sentence encoder on
  in-domain contrastive pairs and evaluates retrieval (MRR@5, P@5, NDCG@5) against an
  off-the-shelf baseline, and (b) performs **obligation-aware hard-negative mining** —
  constructing adversarial training pairs along four failure-mode axes (obligation-flip,
  modal-flip, numeric-perturbation, cross-section near-duplicate) that target the
  confusions procurement practice cannot tolerate.

## Layout
- `data/`    : anonymized IPS-2014 corpus (requirements, suppliers, components, synthetic prices)
- `src/`     : pipeline and mining code (see below)
- `results/` : measured metrics (`results_measured.json`)
- `paper/`   : manuscript source

## Source files
| File | Purpose |
|------|---------|
| `anonymize_pipeline.py`     | deterministic salted pseudonymization of the source spec |
| `rag_finetune_pipeline.py`  | contrastive fine-tuning + retrieval evaluation (baseline) |
| `obligation_mining.py`      | **Algorithm 1** — obligation-aware hard-negative mining (4 classes) |
| `train_and_stress.py`       | trains all configs + runs adversarial stress tests (Obl/Mod/Num-Acc) |
| `eval_extras.py`            | paired-bootstrap significance test + MRR@k sweep |
| `ablation.py`               | base-encoder × epochs ablation |
| `make_figures.py`           | figure generation |

## Reproduce
Install:
```
pip install sentence-transformers faiss-cpu scikit-learn numpy datasets "accelerate>=1.1.0"
```

1. **Mine hard negatives** (Algorithm 1):
   ```
   python src/obligation_mining.py \
     --requirements data/IPS-2014_requirements_anon.json \
     --out mined_tuples.json --stats mining_stats.json
   ```
2. **Train all configs + stress tests** (fills the paper's Tables 3–4):
   ```
   python src/train_and_stress.py \
     --requirements data/IPS-2014_requirements_anon.json \
     --tuples mined_tuples.json --out results_mining.json
   ```
3. **Significance + k-sweep**:
   ```
   python src/eval_extras.py --requirements data/IPS-2014_requirements_anon.json
   ```

Fixed seed (42) makes runs deterministic. A GPU is recommended for the fine-tuning
steps; mining and stress-set construction run on CPU.

## Measured (verified, no GPU needed)
- Corpus: 535 requirements, 1,478 same-section pairs, 496 queries
- Baseline MiniLM zero-shot: MRR@5 = 0.250, P@5 = 0.122, NDCG@5 = 0.140
- Mined tuples: 6,372 (491 real obligation-flips + flagged synthetic modal/numeric)
- Held-out stress-set sizes: |S_obl| = 477, |S_mod| = 224, |S_num| = 83

## Honesty notes
- The scripts bake in **no target values** — every reported metric must come from an
  actual run. A small honest gain is publishable; a large unverified one is a liability.
- Synthetic negatives (modal-flip, numeric-perturbation) are **flagged as synthetic** in
  the released tuples; obligation-flip and cross-section negatives are mined from real data.
- Source OEM specifications are copyright-restricted and **not redistributed**; only
  anonymized derivative data is released. The re-identification key is withheld.
- The cost layer is synthetic and excluded from all metrics.
- Single-annotator labels; no inter-rater statistics are claimed for the corpus itself.

## Acknowledgement
Supported by the Deutsche Forschungsgemeinschaft (DFG), grant UY-56/5-1.
