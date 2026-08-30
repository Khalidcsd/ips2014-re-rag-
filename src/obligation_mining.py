#!/usr/bin/env python3
"""
obligation_mining.py — Obligation-aware hard-negative mining for IPS-2014.

Faithful implementation of Algorithm 1 from the paper
"Obligation-Aware Hard-Negative Mining for Requirements Retrieval".

This constructs, for each anchor requirement, hard negatives along four
failure-mode axes:
    N_obl  (obligation-flip)      — mined from REAL corpus (different marker,
                                     lexically near-identical)
    N_mod  (modal-flip)           — SYNTHETIC (muss<->soll<->kann<->darf swap)
    N_num  (numeric-perturbation) — SYNTHETIC (+/-{10,25,50}% within unit class)
    N_xsec (cross-section dup)     — mined from REAL corpus (top-k cosine,
                                     different section)

HONESTY NOTE (mirrors rag_finetune_pipeline.py):
    This script MINES training tuples. It bakes in no results. Any metric that
    ends up in the paper must come from actually training on these tuples and
    evaluating (train_with_mining.py + eval_stress.py). Synthetic negatives
    (N_mod, N_num) are flagged in the output so the dataset release can mark
    them, exactly as Section 4.1 of the paper promises.

Usage:
    python obligation_mining.py \
        --requirements ../data/IPS-2014_requirements_anon.json \
        --out mined_tuples.json

Dependencies: numpy, scikit-learn (TF-IDF), sentence-transformers + faiss
(only for the N_xsec class; if unavailable, that class is skipped and flagged).
"""
from __future__ import annotations
import argparse, json, random, re, copy
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# German modal alphabet (Section 4.2). Closed set, as the paper states.
MODALS = ["muss", "soll", "kann", "darf"]
# Plural / conjugated surface forms mapped to their lemma, so we can detect and
# swap them while keeping the sentence well-formed at the modal position.
MODAL_FORMS = {
    "muss": ["muss", "müssen", "musste", "mussten"],
    "soll": ["soll", "sollen", "sollte", "sollten"],
    "kann": ["kann", "können", "konnte", "konnten"],
    "darf": ["darf", "dürfen", "durfte", "durften"],
}
FORM_TO_LEMMA = {f: lemma for lemma, forms in MODAL_FORMS.items() for f in forms}

# Numeric-with-unit pattern. Captures value + unit so we can enforce unit-class
# plausibility (perturb 10 bar -> 7.5 bar, never 10 mm), per Section 4.2.
NUM_UNIT_RE = re.compile(
    r"(?P<val>\d+(?:[.,]\d+)?)\s?(?P<unit>°C|°|bar|mm|cm|m|kg|g|Nm|N|kW|W|V|A|Hz|%|s|min|h|l|ml)",
    re.IGNORECASE,
)
PERTURB_FACTORS = [0.10, 0.25, 0.50]


# ---------------------------------------------------------------------------
# Loading & light structure extraction
# ---------------------------------------------------------------------------
def load_requirements(path: Path) -> list[dict]:
    reqs = json.loads(path.read_text("utf-8"))
    kept = []
    for i, r in enumerate(reqs):
        if r.get("text") and len(r["text"]) >= 20:
            rr = dict(r)
            rr["id"] = i
            rr["marker"] = r.get("prefix", "?")
            kept.append(rr)
    return kept


def lemmatized_content(text: str) -> str:
    """Cheap content signature: lowercase, strip modal verbs and numbers so that
    obligation-flip / modal-flip near-pairs collapse to the same content key.
    (The paper uses a lemmatized-content TF-IDF; we approximate lemmatization
    with lowercasing + modal/number removal, which is sufficient to surface
    near-identical clauses differing mainly in marker or modal.)"""
    t = text.lower()
    for form in FORM_TO_LEMMA:
        t = re.sub(rf"\b{form}\b", " ", t)
    t = NUM_UNIT_RE.sub(" ", t)
    t = re.sub(r"[^\wäöüß ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def find_modal(text: str) -> str | None:
    for form, lemma in FORM_TO_LEMMA.items():
        if re.search(rf"\b{form}\b", text.lower()):
            return lemma
    return None


def numeric_spans(text: str) -> list[tuple[str, str, str]]:
    """Return (matched_substring, value, unit) tuples."""
    out = []
    for m in NUM_UNIT_RE.finditer(text):
        out.append((m.group(0), m.group("val"), m.group("unit")))
    return out


# ---------------------------------------------------------------------------
# Synthetic negative generators (flagged synthetic in output)
# ---------------------------------------------------------------------------
def make_modal_flips(text: str, k_mod: int, existing: set[str]) -> list[str]:
    lemma = find_modal(text)
    if lemma is None:
        return []
    # find the actual surface form present, swap it for another modal's base form
    present_form = None
    for form in FORM_TO_LEMMA:
        if re.search(rf"\b{form}\b", text.lower()):
            present_form = form
            break
    out = []
    for other in MODALS:
        if other == lemma:
            continue
        # case-preserving swap at the modal position
        def repl(mo):
            w = mo.group(0)
            return other.capitalize() if w[0].isupper() else other
        swapped = re.sub(rf"\b{re.escape(present_form)}\b", repl, text, count=1,
                         flags=re.IGNORECASE)
        if swapped != text and swapped not in existing:
            out.append(swapped)
    random.shuffle(out)
    return out[:k_mod]


def make_numeric_perturbations(text: str, k_num: int, existing: set[str]) -> list[str]:
    spans = numeric_spans(text)
    if not spans:
        return []
    out = []
    for sub, val, unit in spans:
        try:
            v = float(val.replace(",", "."))
        except ValueError:
            continue
        for f in PERTURB_FACTORS:
            for signed in (1 + f, 1 - f):
                nv = v * signed
                # format like the original (int stays int-ish)
                nv_str = f"{nv:.1f}".rstrip("0").rstrip(".").replace(".", ",") \
                    if "," in val or "." in val else str(int(round(nv)))
                new_sub = sub.replace(val, nv_str, 1)
                cand = text.replace(sub, new_sub, 1)
                if cand != text and cand not in existing:
                    out.append(cand)
    # dedup preserve order
    seen, uniq = set(), []
    for c in out:
        if c not in seen:
            seen.add(c); uniq.append(c)
    random.shuffle(uniq)
    return uniq[:k_num]


# ---------------------------------------------------------------------------
# Algorithm 1
# ---------------------------------------------------------------------------
def mine(reqs: list[dict], k_obl=4, k_mod=2, k_num=2, k_xsec=4,
         tau_lex=0.5, use_encoder=True) -> tuple[list[dict], dict]:
    texts = [r["text"] for r in reqs]
    all_texts = set(texts)
    content = [lemmatized_content(t) for t in texts]

    # TF-IDF over lemmatized content for obligation-flip candidate surfacing
    vec = TfidfVectorizer(min_df=1)
    C = vec.fit_transform(content)
    sim_lex = cosine_similarity(C)  # 535x535, fine at this scale

    # Optional cross-section semantic negatives (needs the off-the-shelf encoder)
    xsec_ok = False
    if use_encoder:
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            enc = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            emb = enc.encode(texts, normalize_embeddings=True,
                             show_progress_bar=False).astype("float32")
            index = faiss.IndexFlatIP(emb.shape[1]); index.add(emb)
            xsec_ok = True
        except Exception as e:
            print(f"[warn] encoder/faiss unavailable, N_xsec skipped: {e}")

    by_section = defaultdict(list)
    for r in reqs:
        by_section[(r["doc"], r["section"])].append(r)

    tuples = []
    counts = defaultdict(int)

    for section, group in by_section.items():
        if len(group) < 2:
            continue
        for r in group:
            ri = r["id"]
            for rp in group:
                if rp["id"] == ri:
                    continue
                neg_texts, neg_meta = [], []

                # --- N_obl: real, different marker, lexically near-identical ---
                obl_cands = []
                for x in reqs:
                    if x["marker"] != r["marker"] and sim_lex[ri, x["id"]] >= tau_lex:
                        obl_cands.append((sim_lex[ri, x["id"]], x["text"]))
                obl_cands.sort(reverse=True)
                for _, t in obl_cands[:k_obl]:
                    neg_texts.append(t); neg_meta.append("obl:real")
                counts["obl"] += min(len(obl_cands), k_obl)

                # --- N_mod: synthetic modal flips ---
                for t in make_modal_flips(r["text"], k_mod, all_texts):
                    neg_texts.append(t); neg_meta.append("mod:synthetic")
                    counts["mod"] += 1

                # --- N_num: synthetic numeric perturbations ---
                for t in make_numeric_perturbations(r["text"], k_num, all_texts):
                    neg_texts.append(t); neg_meta.append("num:synthetic")
                    counts["num"] += 1

                # --- N_xsec: real, other section, top cosine ---
                if xsec_ok:
                    D, I = index.search(emb[ri:ri+1], 4 * k_xsec)
                    picked = 0
                    for j in I[0].tolist():
                        if j == ri:
                            continue
                        if reqs[j]["section"] != r["section"]:
                            neg_texts.append(reqs[j]["text"])
                            neg_meta.append("xsec:real")
                            picked += 1
                            counts["xsec"] += 1
                            if picked >= k_xsec:
                                break

                # dedup negatives, keep first-seen meta
                seen, negs, metas = set(), [], []
                for t, mmeta in zip(neg_texts, neg_meta):
                    if t not in seen and t != r["text"] and t != rp["text"]:
                        seen.add(t); negs.append(t); metas.append(mmeta)

                tuples.append({
                    "anchor": r["text"],
                    "anchor_marker": r["marker"],
                    "positive": rp["text"],
                    "negatives": negs,
                    "negative_types": metas,
                })

    stats = {
        "n_tuples": len(tuples),
        "negatives_by_class_generated": dict(counts),
        "params": {"k_obl": k_obl, "k_mod": k_mod, "k_num": k_num,
                   "k_xsec": k_xsec, "tau_lex": tau_lex},
        "xsec_enabled": xsec_ok,
        "note": "N_mod and N_num are SYNTHETIC (flagged); N_obl and N_xsec are "
                "mined from real corpus. No results are baked in — train and "
                "evaluate to obtain reportable metrics.",
    }
    return tuples, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requirements", type=Path,
                    default=Path("../data/IPS-2014_requirements_anon.json"))
    ap.add_argument("--out", type=Path, default=Path("mined_tuples.json"))
    ap.add_argument("--stats", type=Path, default=Path("mining_stats.json"))
    ap.add_argument("--no-encoder", action="store_true",
                    help="skip N_xsec (semantic cross-section) mining")
    ap.add_argument("--tau_lex", type=float, default=0.5)
    args = ap.parse_args()

    reqs = load_requirements(args.requirements)
    print(f"[mining] {len(reqs)} requirements loaded")
    tuples, stats = mine(reqs, tau_lex=args.tau_lex,
                         use_encoder=not args.no_encoder)
    args.out.write_text(json.dumps(tuples, ensure_ascii=False, indent=2), "utf-8")
    args.stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2), "utf-8")
    print(f"[mining] wrote {len(tuples)} tuples -> {args.out}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
