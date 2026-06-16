#!/usr/bin/env python3
"""
IPS-2014 — domain-adaptation fine-tuning + RAG pipeline
=======================================================
End-to-end, *honest* experimental pipeline for the requirements-engineering
paper. It trains a domain-adapted retriever on the real anonymized requirement
corpus and measures retrieval quality against an off-the-shelf baseline.

CRITICAL — read before reporting anything
-----------------------------------------
The numbers this script prints (corpus size, MRR, P@5, NDCG, deltas) are the
ONLY numbers that belong in the paper. There are no target values baked in
anywhere. Run it, take what it reports, and write that down — even if the gain
is small or zero. A small honest delta is a publishable result; a large
unverified one is a liability. This is a single-annotator prototype evaluation;
the write-up must say so (no inter-rater statistics, no deployment claims).

Stages
------
  1 build_contrastive_corpus   (real, deterministic — runs on CPU in seconds)
  2 finetune                   (needs torch; GPU recommended)
  3 build_index                (embeds corpus)
  4 evaluate                   (baseline vs fine-tuned -> measured metrics)
  5 rag_answer                 (retrieval-grounded generation via your LLM)

Inputs
------
  requirements_anon.json : [{"doc","section","prefix","text"}, ...]   (real corpus)

Dependencies
------------
  pip install sentence-transformers faiss-cpu scikit-learn numpy
  (stage 5 only) pip install openai   + OPENAI_API_KEY
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # handles German
FINETUNED_DIR = "model_ips2014_finetuned"


# ---------------------------------------------------------------------------
# Stage 1 — build a contrastive corpus from the REAL requirements
# ---------------------------------------------------------------------------

def load_requirements(path: Path) -> list[dict]:
    reqs = json.loads(path.read_text("utf-8"))
    # keep only well-formed obligation statements
    return [r for r in reqs if r.get("text") and len(r["text"]) >= 20]


def build_contrastive_corpus(reqs: list[dict], max_pairs_per_section: int = 60) -> list[tuple[str, str]]:
    """
    Positive pairs = two requirements from the same spec section (they share
    engineering context). In-batch negatives during training supply the
    contrast. Returns (anchor, positive) text pairs. The count is whatever the
    real data yields — report it as the corpus size.
    """
    by_section: dict[str, list[str]] = defaultdict(list)
    for r in reqs:
        by_section[(r["doc"], r["section"])].append(r["text"])

    pairs: list[tuple[str, str]] = []
    for items in by_section.values():
        if len(items) < 2:
            continue
        random.shuffle(items)
        made = 0
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                pairs.append((items[i], items[j]))
                made += 1
                if made >= max_pairs_per_section:
                    break
            if made >= max_pairs_per_section:
                break
    random.shuffle(pairs)
    return pairs


# ---------------------------------------------------------------------------
# Stage 2 — fine-tune
# ---------------------------------------------------------------------------

def finetune(pairs: list[tuple[str, str]], base_model: str, out_dir: str, epochs: int = 1) -> str:
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader

    model = SentenceTransformer(base_model)
    train = [InputExample(texts=[a, b]) for a, b in pairs]
    loader = DataLoader(train, shuffle=True, batch_size=32)
    loss = losses.MultipleNegativesRankingLoss(model)   # contrastive, in-batch negatives
    warmup = math.ceil(len(loader) * epochs * 0.1)
    model.fit(train_objectives=[(loader, loss)], epochs=epochs,
              warmup_steps=warmup, show_progress_bar=True)
    model.save(out_dir)
    return out_dir


# ---------------------------------------------------------------------------
# Stage 3 — RAG index
# ---------------------------------------------------------------------------

def embed(model, texts: list[str]) -> np.ndarray:
    v = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(v, dtype="float32")


def build_index(embeddings: np.ndarray):
    import faiss
    idx = faiss.IndexFlatIP(embeddings.shape[1])   # cosine via normalized vectors
    idx.add(embeddings)
    return idx


# ---------------------------------------------------------------------------
# Stage 4 — evaluation (measured, not assumed)
# ---------------------------------------------------------------------------

def make_eval_queries(reqs: list[dict]) -> tuple[list[dict], dict[int, set[int]]]:
    """
    Each requirement is a query; relevant docs = other requirements in the same
    section. This is a standard domain-retrieval eval and uses only real labels
    (the section a requirement came from).
    """
    section_of = [(r["doc"], r["section"]) for r in reqs]
    relevant: dict[int, set[int]] = {}
    for qi, sec in enumerate(section_of):
        rel = {di for di, s in enumerate(section_of) if s == sec and di != qi}
        if rel:
            relevant[qi] = rel
    queries = [{"i": i, "text": reqs[i]["text"]} for i in relevant]
    return queries, relevant


def ir_metrics(model, reqs: list[dict], relevant: dict[int, set[int]], k: int = 5) -> dict:
    corpus = [r["text"] for r in reqs]
    emb = embed(model, corpus)
    index = build_index(emb)
    q_idx = list(relevant.keys())
    q_emb = emb[q_idx]
    # retrieve k+1 (first hit is the query itself) and drop self
    D, I = index.search(q_emb, k + 1)

    mrr = p_at_k = ndcg = 0.0
    for row, qi in enumerate(q_idx):
        ranked = [d for d in I[row].tolist() if d != qi][:k]
        rel = relevant[qi]
        # MRR
        rr = 0.0
        for rank, d in enumerate(ranked, start=1):
            if d in rel:
                rr = 1.0 / rank
                break
        mrr += rr
        # P@k
        p_at_k += sum(1 for d in ranked if d in rel) / k
        # NDCG@k
        dcg = sum((1.0 / math.log2(rank + 1)) for rank, d in enumerate(ranked, start=1) if d in rel)
        ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(rel), k) + 1))
        ndcg += (dcg / ideal) if ideal else 0.0

    n = len(q_idx)
    return {"queries": n, f"MRR@{k}": mrr / n, f"P@{k}": p_at_k / n, f"NDCG@{k}": ndcg / n}


# ---------------------------------------------------------------------------
# Stage 5 — retrieval-grounded answer (uses YOUR llm; nothing fabricated)
# ---------------------------------------------------------------------------

def rag_answer(model, index, corpus: list[str], question: str, k: int = 5) -> dict:
    q = embed(model, [question])
    D, I = index.search(q, k)
    ctx = [corpus[i] for i in I[0].tolist()]
    prompt = ("Answer using ONLY the retrieved requirements. If they don't cover it, say so.\n\n"
              "Retrieved requirements:\n" + "\n".join(f"[{n+1}] {c}" for n, c in enumerate(ctx)) +
              f"\n\nQuestion: {question}\nGrounded answer:")
    answer = "[LLM not configured — set OPENAI_API_KEY to enable generation]"
    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI
        client = OpenAI()
        r = client.chat.completions.create(
            model="gpt-4o", temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = r.choices[0].message.content
    return {"question": question, "contexts": ctx, "answer": answer}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requirements", type=Path, default=Path("requirements_anon.json"))
    ap.add_argument("--stage", choices=["corpus", "all"], default="all")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--results", type=Path, default=Path("results_measured.json"))
    args = ap.parse_args()

    reqs = load_requirements(args.requirements)
    pairs = build_contrastive_corpus(reqs)
    print(f"[corpus] requirements={len(reqs)}  contrastive_pairs={len(pairs)}")
    print("         ^ report THESE as your corpus sizes — not any pre-written figure.")

    if args.stage == "corpus":
        return

    from sentence_transformers import SentenceTransformer
    queries, relevant = make_eval_queries(reqs)

    print("[baseline] evaluating off-the-shelf model ...")
    base = SentenceTransformer(BASE_MODEL)
    base_m = ir_metrics(base, reqs, relevant)

    print("[finetune] training domain-adapted model ...")
    finetune(pairs, BASE_MODEL, FINETUNED_DIR, epochs=args.epochs)
    ft = SentenceTransformer(FINETUNED_DIR)
    ft_m = ir_metrics(ft, reqs, relevant)

    delta = {k: round(ft_m[k] - base_m[k], 4) for k in base_m if k != "queries"}
    out = {"corpus": {"requirements": len(reqs), "contrastive_pairs": len(pairs)},
           "baseline": base_m, "finetuned": ft_m, "delta": delta,
           "note": "single-annotator prototype; measured this run; no inter-rater stats claimed"}
    args.results.write_text(json.dumps(out, indent=2), "utf-8")
    print("\n=== MEASURED RESULTS (these go in the paper) ===")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
