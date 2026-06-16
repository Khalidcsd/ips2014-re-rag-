#!/usr/bin/env python3
"""ablation.py - base-encoder x epochs ablation for IPS-2014.
Run after rag_finetune_pipeline.py works:
  python ablation.py --requirements IPS-2014_requirements_anon.json
Trains several models (slow; use a GPU). Prints LaTeX-ready table rows.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from rag_finetune_pipeline import (
    load_requirements, build_contrastive_corpus, make_eval_queries, ir_metrics,
)
ENCODERS = [
    ("multilingual MiniLM", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
    ("German sentence enc.", "T-Systems-onsite/cross-en-de-roberta-sentence-transformer"),
]
EPOCHS = [1, 2, 3]

def finetune_model(base_id, pairs, out_dir, epochs):
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader
    model = SentenceTransformer(base_id)
    train = [InputExample(texts=[a, b]) for a, b in pairs]
    loader = DataLoader(train, shuffle=True, batch_size=32)
    loss = losses.MultipleNegativesRankingLoss(model)
    warmup = math.ceil(len(loader) * epochs * 0.1)
    model.fit(train_objectives=[(loader, loss)], epochs=epochs,
              warmup_steps=warmup, show_progress_bar=True)
    model.save(out_dir); return out_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requirements", type=Path, default=Path("IPS-2014_requirements_anon.json"))
    ap.add_argument("--out", type=Path, default=Path("results_ablation.json"))
    args = ap.parse_args()
    from sentence_transformers import SentenceTransformer
    reqs = load_requirements(args.requirements)
    pairs = build_contrastive_corpus(reqs)
    _, relevant = make_eval_queries(reqs)
    rows = []
    for label, model_id in ENCODERS:
        for ep in EPOCHS:
            tag = f"abl_{label.split()[0]}_{ep}".replace(" ", "")
            print(f"\n=== {label} | {ep} epoch(s) ===")
            try:
                finetune_model(model_id, pairs, tag, ep)
                m = ir_metrics(SentenceTransformer(tag), reqs, relevant)
                rows.append({"encoder": label, "epochs": ep,
                             "MRR@5": m["MRR@5"], "P@5": m["P@5"], "NDCG@5": m["NDCG@5"]})
            except Exception as e:
                print(f"  skipped ({e})")
    args.out.write_text(json.dumps(rows, indent=2))
    print("\n=== LaTeX rows ===")
    for r in rows:
        print(f"{r['encoder']} & {r['epochs']} & {r['MRR@5']:.3f} & {r['P@5']:.3f} & {r['NDCG@5']:.3f} \\\\")

if __name__ == "__main__":
    main()
