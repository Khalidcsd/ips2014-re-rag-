# Running the obligation-aware mining experiments

Three new scripts complete the paper's method. They bake in **no results** —
run them and copy the printed numbers into the paper.

## Prerequisites
```
pip install sentence-transformers faiss-cpu scikit-learn numpy datasets "accelerate>=1.1.0"
```
Fine-tuning needs a GPU for reasonable speed. A free Colab T4 handles this
corpus in a few minutes. (CPU-only works for mining + stress-set construction,
but training MiniLM needs >4 GB RAM.)

## Step 1 — Mine the hard negatives (Algorithm 1)
```
python obligation_mining.py \
  --requirements ../data/IPS-2014_requirements_anon.json \
  --out mined_tuples.json --stats mining_stats.json
```
Produces the four negative classes. N_mod and N_num are flagged synthetic.
Add `--no-encoder` to skip the semantic cross-section class (N_xsec) if you
don't want to load the encoder at mining time.

## Step 2 — Train all configs + run stress tests (fills Tables 3-4)
```
python train_and_stress.py \
  --requirements ../data/IPS-2014_requirements_anon.json \
  --tuples mined_tuples.json \
  --out results_mining.json
```
Trains: baseline, same-section-only, each single class (ablation), all classes.
Prints MRR@5/P@5/NDCG@5 (Table 3) and Obl/Mod/Num-Acc (Table 4), plus
ready-to-paste LaTeX rows.

## Step 3 — Significance test + k-sweep (fills the p-value and Figure 4)
```
python eval_extras.py --requirements ../data/IPS-2014_requirements_anon.json
```
(already in the repo; run after a fine-tuned model exists)

## Measured so far (verified on this corpus, no GPU needed)
- Corpus: 535 requirements, 1,478 same-section pairs, 496 queries
- Baseline MiniLM zero-shot: MRR@5 = 0.250, P@5 = 0.122, NDCG@5 = 0.140  (reproduced exactly)
- Mined tuples: 6,372 (491 real obligation-flips + flagged synthetic mod/num)
- Stress-set sizes: |S_obl| = 477, |S_mod| = 224, |S_num| = 83

## What still needs a GPU run
- MRR@5 for the fine-tuned + mining configs (Table 3 [FILL] rows)
- Obl-Acc / Mod-Acc / Num-Acc (Table 4 — the central result)
- RAG-Faith (Table 5 — needs the LLM grounding eval, 100 queries, 2 raters)
- Bootstrap p-value + k-sweep (eval_extras.py)

Everything except RAG-Faith is produced by the two scripts above.
RAG-Faith needs your OpenAI key and two human raters (Section 5.3).
