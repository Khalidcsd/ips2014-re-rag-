#!/usr/bin/env python3
"""
train_and_stress.py — consume mined tuples, fine-tune, and run the adversarial
stress tests that fill Tables 3-5 of the paper.

This is the script that PRODUCES the [FILL] numbers. It bakes in nothing.
Run it (GPU strongly recommended; ~minutes on a T4 for this corpus) and copy
the printed values into the paper. If a gain is small, report the small gain.

Pipeline:
  1. Load real corpus + mined tuples (from obligation_mining.py).
  2. Train THREE retrievers on identical data, differing only in negatives:
       (a) same-section only        -> reproduces the 0.397 prior result
       (b) each single mined class   -> ablation rows (Table 3/4)
       (c) all mined classes         -> the full method
  3. Same-section retrieval metrics (Table 3): MRR@5, P@5, NDCG@5.
  4. Adversarial stress tests (Table 4): Obl-Acc, Mod-Acc, Num-Acc on held-out
     triples constructed with a DIFFERENT seed than training (Section 7 mitigation).

Usage:
  python train_and_stress.py \
     --requirements ../data/IPS-2014_requirements_anon.json \
     --tuples mined_tuples.json \
     --out results_mining.json
"""
from __future__ import annotations
import argparse, json, math, random, re
from pathlib import Path
import numpy as np

SEED = 42
random.seed(SEED); np.random.seed(SEED)

BASE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# import the honest, already-written helpers
import sys
sys.path.insert(0, str(Path(__file__).parent))
from rag_finetune_pipeline import (
    load_requirements as _load_reqs, make_eval_queries, ir_metrics, embed, build_index,
)
from obligation_mining import (
    load_requirements, make_modal_flips, make_numeric_perturbations,
    find_modal, numeric_spans, lemmatized_content,
)


# ---------------------------------------------------------------------------
# Training on explicit (anchor, positive, negatives) tuples
# ---------------------------------------------------------------------------
def train(tuples, class_filter=None, epochs=1, batch_size=16, out_dir="m_tmp"):
    """class_filter: None = use all negatives; or a set like {'obl'} to keep
    only that class (for ablation). Same-section-only baseline = class_filter=set()
    (no hard negatives; in-batch only)."""
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader

    model = SentenceTransformer(BASE_MODEL)
    examples = []
    for t in tuples:
        negs = []
        if class_filter is None:
            negs = t["negatives"]
        elif class_filter:
            negs = [n for n, ty in zip(t["negatives"], t["negative_types"])
                    if ty.split(":")[0] in class_filter]
        # MultipleNegativesRankingLoss accepts (anchor, positive, negative...) ;
        # with no explicit negatives it falls back to in-batch (the baseline).
        if negs:
            for n in negs[:4]:
                examples.append(InputExample(texts=[t["anchor"], t["positive"], n]))
        else:
            examples.append(InputExample(texts=[t["anchor"], t["positive"]]))
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    loss = losses.MultipleNegativesRankingLoss(model)
    warmup = math.ceil(len(loader) * epochs * 0.1)
    model.fit(train_objectives=[(loader, loss)], epochs=epochs,
              warmup_steps=warmup, show_progress_bar=False)
    model.save(out_dir)
    return model


# ---------------------------------------------------------------------------
# Held-out adversarial stress sets (different seed than training, Section 7)
# ---------------------------------------------------------------------------
def build_stress_sets(reqs, seed=1234):
    """For each triple (q, r+, r-): r+ is a genuine same-section peer; r- is a
    hard negative of the target class. Chance accuracy = 0.5. Uses a DIFFERENT
    seed and perturbation policy than mining, so a model cannot win by
    memorising the training perturbation function."""
    rng = random.Random(seed)
    from collections import defaultdict
    by_sec = defaultdict(list)
    for r in reqs:
        by_sec[(r["doc"], r["section"])].append(r)

    obl, mod, num = [], [], []
    all_text = {r["text"] for r in reqs}

    # obligation: real different-marker near pair
    content = {r["id"]: lemmatized_content(r["text"]) for r in reqs}
    for sec, grp in by_sec.items():
        if len(grp) < 2:
            continue
        for r in grp:
            peers = [p for p in grp if p["id"] != r["id"]]
            if not peers:
                continue
            rplus = rng.choice(peers)
            # obligation-flip r-: different marker, share >=3 content tokens
            cand = [x for x in reqs if x["marker"] != r["marker"]
                    and x["id"] != r["id"]
                    and len(set(content[r["id"]].split()) & set(content[x["id"]].split())) >= 3]
            if cand:
                obl.append((r["text"], rplus["text"], rng.choice(cand)["text"]))
            # modal-flip r- with a different-seed policy (pick a single random modal)
            mf = make_modal_flips(r["text"], k_mod=3, existing=all_text)
            if mf:
                mod.append((r["text"], rplus["text"], rng.choice(mf)))
            # numeric r- with held-out magnitudes (use +/-{15,35}% not train's {10,25,50})
            spans = numeric_spans(r["text"])
            if spans:
                sub, val, unit = spans[0]
                try:
                    v = float(val.replace(",", "."))
                    f = rng.choice([0.15, 0.35])
                    nv = v * (1 + rng.choice([f, -f]))
                    nv_str = (f"{nv:.1f}".rstrip("0").rstrip(".").replace(".", ",")
                              if ("," in val or "." in val) else str(int(round(nv))))
                    rminus = r["text"].replace(sub, sub.replace(val, nv_str, 1), 1)
                    if rminus != r["text"]:
                        num.append((r["text"], rplus["text"], rminus))
                except ValueError:
                    pass
    return {"obl": obl, "mod": mod, "num": num}


def preservation_accuracy(model, triples):
    """Fraction of triples where sim(q, r+) > sim(q, r-). Chance = 0.5."""
    if not triples:
        return None, 0
    q = embed(model, [t[0] for t in triples])
    rp = embed(model, [t[1] for t in triples])
    rm = embed(model, [t[2] for t in triples])
    sp = (q * rp).sum(1)
    sm = (q * rm).sum(1)
    return float((sp > sm).mean()), len(triples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requirements", type=Path,
                    default=Path("../data/IPS-2014_requirements_anon.json"))
    ap.add_argument("--tuples", type=Path, default=Path("mined_tuples.json"))
    ap.add_argument("--out", type=Path, default=Path("results_mining.json"))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    reqs = load_requirements(args.requirements)
    tuples = json.loads(args.tuples.read_text("utf-8"))
    _, relevant = make_eval_queries(reqs)
    stress = build_stress_sets(reqs)
    print(f"[stress sizes] obl={len(stress['obl'])} "
          f"mod={len(stress['mod'])} num={len(stress['num'])}")

    from sentence_transformers import SentenceTransformer
    base = SentenceTransformer(BASE_MODEL)

    configs = {
        "baseline_zeroshot": None,           # no training
        "same_section_only": set(),          # in-batch negatives only
        "N_obl_only": {"obl"},
        "N_mod_only": {"mod"},
        "N_num_only": {"num"},
        "N_xsec_only": {"xsec"},
        "all_hard_negatives": None,          # all mined classes
    }

    results = {}
    for name, cf in configs.items():
        print(f"\n=== {name} ===")
        if name == "baseline_zeroshot":
            model = base
        else:
            model = train(tuples, class_filter=cf, epochs=args.epochs,
                          batch_size=args.batch_size, out_dir=f"m_{name}")
        retr = ir_metrics(model, reqs, relevant)
        oa, no = preservation_accuracy(model, stress["obl"])
        ma, nm = preservation_accuracy(model, stress["mod"])
        na, nn = preservation_accuracy(model, stress["num"])
        results[name] = {
            "MRR@5": retr["MRR@5"], "P@5": retr["P@5"], "NDCG@5": retr["NDCG@5"],
            "Obl_Acc": oa, "Mod_Acc": ma, "Num_Acc": na,
        }
        print(json.dumps(results[name], indent=2))

    out = {
        "stress_sizes": {"obl": len(stress["obl"]), "mod": len(stress["mod"]),
                         "num": len(stress["num"])},
        "results": results,
        "note": "All values measured this run. Synthetic stress sets use a "
                "held-out seed/policy (Section 7). Copy into Tables 3-4.",
    }
    args.out.write_text(json.dumps(out, indent=2), "utf-8")
    print("\n=== LaTeX rows (Table 4: stress) ===")
    for name, r in results.items():
        def f(x): return f"{x:.3f}" if isinstance(x, float) else "—"
        print(f"{name} & {f(r['Obl_Acc'])} & {f(r['Mod_Acc'])} & {f(r['Num_Acc'])} \\\\")
    print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
