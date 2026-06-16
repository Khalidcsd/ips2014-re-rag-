#!/usr/bin/env python3
"""eval_extras.py - significance test + MRR@k sweep for IPS-2014.
Run after rag_finetune_pipeline.py (so the fine-tuned model exists):
  python eval_extras.py --requirements IPS-2014_requirements_anon.json
Writes results_extras.json. Every printed number is measured.
"""
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
import numpy as np
from rag_finetune_pipeline import (
    BASE_MODEL, FINETUNED_DIR, load_requirements, make_eval_queries, embed, build_index,
)
SEED = 42
random.seed(SEED); np.random.seed(SEED)

def per_query_rr(model, reqs, relevant, k):
    corpus = [r["text"] for r in reqs]
    emb = embed(model, corpus); index = build_index(emb)
    q_idx = list(relevant.keys())
    D, I = index.search(emb[q_idx], k + 1)
    rr = []
    for row, qi in enumerate(q_idx):
        ranked = [d for d in I[row].tolist() if d != qi][:k]
        val = 0.0
        for rank, d in enumerate(ranked, 1):
            if d in relevant[qi]:
                val = 1.0 / rank; break
        rr.append(val)
    return np.asarray(rr)

def paired_bootstrap(base_rr, ft_rr, iters=10000):
    diff = ft_rr - base_rr; obs = diff.mean(); n = len(diff)
    centered = diff - obs; count = 0
    for _ in range(iters):
        s = centered[np.random.randint(0, n, n)]
        if abs(s.mean()) >= abs(obs): count += 1
    return obs, (count + 1) / (iters + 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requirements", type=Path, default=Path("IPS-2014_requirements_anon.json"))
    ap.add_argument("--out", type=Path, default=Path("results_extras.json"))
    args = ap.parse_args()
    from sentence_transformers import SentenceTransformer
    reqs = load_requirements(args.requirements)
    _, relevant = make_eval_queries(reqs)
    base = SentenceTransformer(BASE_MODEL); ft = SentenceTransformer(FINETUNED_DIR)
    b5, f5 = per_query_rr(base, reqs, relevant, 5), per_query_rr(ft, reqs, relevant, 5)
    obs, p = paired_bootstrap(b5, f5)
    sweep = {"baseline": {}, "finetuned": {}}
    for k in (1, 3, 5, 10):
        sweep["baseline"][k] = float(per_query_rr(base, reqs, relevant, k).mean())
        sweep["finetuned"][k] = float(per_query_rr(ft, reqs, relevant, k).mean())
    out = {"n_queries": len(b5), "mean_rr_gain": obs,
           "p_value_paired_bootstrap": p, "mrr_at_k": sweep}
    args.out.write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=2))
    print("\nPaper fills:"); print(f"  significance: paired bootstrap, p = {p:.4f}")
    for k in (1, 3, 5, 10):
        print(f"  FILLn b{k} -> {sweep['baseline'][k]:.3f}   FILLn f{k} -> {sweep['finetuned'][k]:.3f}")

if __name__ == "__main__":
    main()
