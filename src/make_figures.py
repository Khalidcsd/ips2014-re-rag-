#!/usr/bin/env python3
"""
make_figures.py — publication-quality figures for the IPS-2014 paper.

Generates vector PDFs with a hand-tuned, colorblind-aware palette:
  fig_reqtypes.pdf   requirement-type counts          (real data, baked in)
  fig_reqlen.pdf     requirement word-length histogram (real data, baked in)
  fig_suppliers.pdf  approved suppliers per category   (real data, baked in)
  fig_ksens.pdf      MRR@k curve  (only if results_extras.json is present)

The first three use measured corpus statistics and render immediately.
Place the PDFs next to main.tex (Overleaf: upload them) so \includegraphics finds them.
"""
from __future__ import annotations
import json, os
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- palette
INK    = "#1F2933"   # near-black text
BLUE_D = "#28527A"   # deep blue (primary)
BLUE_M = "#4A7AB0"   # mid blue
BLUE_L = "#9DBAD6"   # light blue
CLAY   = "#B5483D"   # muted terracotta (accent / highlight)
CLAY_L = "#D8A39B"
SLATE  = "#7C8A9A"   # neutral

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.edgecolor": "#C2CAD2", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})


def _label_bars(ax, bars, vals, horizontal=False, fmt="{:.0f}"):
    for b, v in zip(bars, vals):
        if horizontal:
            ax.text(b.get_width() + max(vals) * 0.01, b.get_y() + b.get_height() / 2,
                    fmt.format(v), va="center", ha="left", fontsize=8, color=INK)
        else:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(vals) * 0.015,
                    fmt.format(v), ha="center", va="bottom", fontsize=8, color=INK)


def fig_reqtypes():
    labels = ["AM", "A", "M", "other"]
    vals = [260, 232, 10, 33]
    colors = [BLUE_D, BLUE_M, BLUE_L, SLATE]   # monochrome ramp + neutral
    fig, ax = plt.subplots(figsize=(3.2, 2.5))
    bars = ax.bar(labels, vals, color=colors, width=0.66, edgecolor="white", linewidth=0.6)
    ax.set_ylabel("# requirements")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.grid(axis="y", color="#E5E9EE", linewidth=0.8)
    ax.set_axisbelow(True)
    _label_bars(ax, bars, vals)
    fig.savefig("fig_reqtypes.pdf"); plt.close(fig)


def fig_reqlen():
    labels = ["0–10", "10–20", "20–30", "30–40", "40–50", "50+"]
    vals = [105, 278, 90, 43, 15, 4]
    peak = vals.index(max(vals))
    colors = [CLAY if i == peak else CLAY_L for i in range(len(vals))]  # highlight modal bin
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    bars = ax.bar(labels, vals, color=colors, width=0.82, edgecolor="white", linewidth=0.6)
    ax.set_ylabel("# requirements"); ax.set_xlabel("words per requirement")
    ax.set_ylim(0, max(vals) * 1.16)
    ax.grid(axis="y", color="#E5E9EE", linewidth=0.8); ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelrotation=25)
    _label_bars(ax, bars, vals)
    fig.savefig("fig_reqlen.pdf"); plt.close(fig)


def fig_suppliers():
    data = [("Monitoring", 13), ("Valves", 10), ("Cylinders", 8), ("Fittings", 7),
            ("Motors", 6), ("Couplings", 6), ("Filters", 5), ("Accumulators", 5),
            ("Clamping", 5), ("Pumps", 4), ("Power Units", 2), ("Rotary Unions", 2)]
    data.sort(key=lambda x: x[1])                      # ascending for horizontal bars
    cats = [c for c, _ in data]; vals = [v for _, v in data]
    vmax = max(vals)
    colors = [mpl.colors.to_hex(c) for c in
              [plt.cm.Blues(0.35 + 0.6 * (v / vmax)) for v in vals]]  # sequential by value
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    bars = ax.barh(cats, vals, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_xlabel("# approved suppliers")
    ax.set_xlim(0, vmax * 1.12)
    ax.grid(axis="x", color="#E5E9EE", linewidth=0.8); ax.set_axisbelow(True)
    _label_bars(ax, bars, vals, horizontal=True)
    fig.savefig("fig_suppliers.pdf"); plt.close(fig)


def fig_ksens(path="results_extras.json"):
    if not os.path.exists(path):
        print("  (skipping fig_ksens.pdf — run eval_extras.py first to produce results_extras.json)")
        return
    d = json.load(open(path))["mrr_at_k"]
    ks = sorted(int(k) for k in d["baseline"])
    base = [d["baseline"][str(k)] for k in ks]
    ft = [d["finetuned"][str(k)] for k in ks]
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ax.plot(ks, base, "-o", color=SLATE, label="Baseline", linewidth=1.8, markersize=5)
    ax.plot(ks, ft, "-s", color=CLAY, label="Fine-tuned", linewidth=1.8, markersize=5)
    ax.set_xlabel("k"); ax.set_ylabel("MRR@k"); ax.set_xticks(ks)
    ax.grid(color="#E5E9EE", linewidth=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig("fig_ksens.pdf"); plt.close(fig)


if __name__ == "__main__":
    fig_reqtypes(); fig_reqlen(); fig_suppliers(); fig_ksens()
    print("wrote: fig_reqtypes.pdf, fig_reqlen.pdf, fig_suppliers.pdf"
          " (+ fig_ksens.pdf if results_extras.json present)")
