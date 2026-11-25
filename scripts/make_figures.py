"""Generate 3 publication-quality 300-DPI figures for Paper 24.
Author: Lluis Eriksson
"""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9.5,
    "axes.titlesize": 10.0,
    "axes.labelsize": 9.2,
    "legend.fontsize": 7.3,
    "xtick.labelsize": 8.4,
    "ytick.labelsize": 8.4,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linestyle": "--",
})


def make_fig1() -> None:
    d2 = json.loads((RESULTS / "exp2_chord_entropy_gap.json").read_text(encoding="utf-8"))
    rows = d2["rows"]
    ts = [r["t0"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.65), dpi=300)

    # Panel (a): Off-center qubit chord rates: Schumacher S, Endpoint C, Holevo chi
    ax1.plot(ts, [r["endpoint_C"] for r in rows], "s--", color="#c1121f", lw=1.9, ms=5.4,
             label=r"Endpoint simulation $C = I(X:U)$")
    ax1.plot(ts, [r["schumacher_S"] for r in rows], "^-.", color="#e07a5f", lw=1.9, ms=5.4,
             label=r"Schumacher bound $S(\bar{\omega})$")
    ax1.plot(ts, [r["holevo_chi"] for r in rows], "o-", color="#1b6ca8", lw=2.2, ms=5.8,
             label=r"Holevo information $\chi = I(X:W)$")
    ax1.set_xlabel(r"Chord mixing parameter $t_0 \in [0, 1/2]$")
    ax1.set_ylabel(r"Rate (qubits / bits per signal)")
    ax1.set_title(r"(a) Off-Center Qubit Chord Rates vs. Holevo $\chi$")
    ax1.legend(loc="upper right", frameon=True, framealpha=0.94)

    # Panel (b): Exact rational mixing defect r_lin & upper rate gap min(C, S) - chi
    ax2.plot(ts, [r["r_lin"] for r in rows], "o-", color="#2a9d8f", lw=2.1, ms=5.8,
             label=r"Exact linear-entropy defect $r_{\mathrm{lin}} \in \mathbb{Q}$")
    ax2.plot(ts, [r["best_upper_gap"] for r in rows], "D--", color="#c1121f", lw=1.9, ms=5.4,
             label=r"Protocol gap $\min(C, S(\bar{\omega})) - \chi$")
    ax2.axvline(1.0 / 6.0, color="#222222", lw=1.3, ls=":", label=r"Certified point $t_0 = 1/6$ ($r_{\mathrm{lin}}=1/10$)")
    ax2.set_ylim(-0.01, 0.255)
    ax2.set_xlabel(r"Chord mixing parameter $t_0 \in [0, 1/2]$")
    ax2.set_ylabel(r"Mixing defect / rate gap")
    ax2.set_title(r"(b) Exact Purity Defect vs. Visible Rate Gap")
    ax2.legend(loc="upper left", frameon=True, framealpha=0.94)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_chord_rates_and_defect.png", dpi=300)
    plt.close(fig)


def make_fig2() -> None:
    d3 = json.loads((RESULTS / "exp3_erasure_blind_vs_visible.json").read_text(encoding="utf-8"))
    d4 = json.loads((RESULTS / "exp4_caratheodory_scaling.json").read_text(encoding="utf-8"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.65), dpi=300)

    # Panel (a): Qutrit erasure separation: Blind rate vs Visible rate (= Holevo chi)
    r3 = d3["rows"]
    es = [r["erasure_e"] for r in r3]
    ax1.plot(es, [r["blind_Rblind"] for r in r3], "s--", color="#c1121f", lw=2.0, ms=5.6,
             label=r"Blind rate $R_{\mathrm{blind}} = \chi + h_2(e)$")
    ax1.plot(es, [r["visible_Rcr"] for r in r3], "o-", color="#1b6ca8", lw=2.2, ms=5.8,
             label=r"Visible rate $R_{\mathrm{cr}} = \chi$ ($r_{\mathrm{mix}} = 0$)")
    ax1.fill_between(es, [r["visible_Rcr"] for r in r3], [r["blind_Rblind"] for r in r3],
                     color="#c1121f", alpha=0.12, label=r"Blind-only gap $H(J|X) = h_2(e)$")
    ax1.set_ylim(-0.06, 1.72)
    ax1.set_xlabel(r"Erasure probability $e \in [0, 1]$")
    ax1.set_ylabel(r"Compression rate (qubits per signal)")
    ax1.set_title(r"(a) Qutrit Erasure: Visible $R_{\mathrm{cr}}=\chi < R_{\mathrm{blind}}$")
    ax1.legend(loc="upper right", frameon=True, framealpha=0.94)

    # Panel (b): Lemma 3 Caratheodory pure-atom bound |X|d_W^2 + 1 vs redundant dim d_N
    r4 = d4["rows"]
    dns = [r["dim_N"] for r in r4]
    carats = [r["caratheodory_bound"] for r in r4]
    bars = ax2.bar(dns, carats, width=0.55, color="#2a9d8f", edgecolor="#222", lw=0.75,
                   label="Lemma 3 Carathéodory bound $|X| d_W^2 + 1$")
    for x_v, y_v in zip(dns, carats):
        ax2.text(x_v, y_v + 3.0, str(y_v), ha="center", va="bottom", fontweight="bold", fontsize=8.2)
    ax2.set_xticks(dns)
    ax2.set_xticklabels([f"dN={d}\n(dW={2*d})" for d in dns], fontsize=7.8)
    ax2.set_ylim(0, 152)
    ax2.set_xlabel(r"Redundant factor dimension $d_N$ (core $d_Q = 2$)")
    ax2.set_ylabel(r"Max pure-atom flags $|X|d_W^2 + 1$")
    ax2.set_title(" (b) Lemma 3 Carathéodory Scaling ($|X|=2$, Pure Core)".strip())
    ax2.legend(loc="upper left", frameon=True, framealpha=0.94)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_erasure_and_caratheodory.png", dpi=300)
    plt.close(fig)


def make_fig3() -> None:
    d1 = json.loads((RESULTS / "exp1_certified_atlas.json").read_text(encoding="utf-8"))
    rows = d1["rows"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.95), dpi=300, gridspec_kw={"width_ratios": [1.62, 1.0]})

    mode_colors = {
        "qubit_bloch_dichotomy_exact": "#1b6ca8",
        "ki_block_tensor_criterion_exact": "#2a9d8f",
        "pure_atom_caratheodory_decomp_exact": "#e07a5f",
        "sector_splitting_and_partial_additivity_exact": "#c1121f",
    }
    labels = [r["file"].replace(".cert.json", "") for r in rows]
    ms = [r["verify_ms"] for r in rows]
    cols = [mode_colors[r["mode"]] for r in rows]
    y = np.arange(len(rows))
    ax1.barh(y, ms, color=cols, edgecolor="#222", lw=0.6, height=0.68)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=6.1)
    ax1.invert_yaxis()
    ax1.set_xlabel(r"Exact-$\mathbb{Q}$ verification latency (ms)")
    ax1.set_title(r"(a) Zero-Float Replay Latency Across 16 Certificates")

    counts = [
        sum(1 for r in rows if r["mode"] == "qubit_bloch_dichotomy_exact"),
        sum(1 for r in rows if r["mode"] == "ki_block_tensor_criterion_exact"),
        sum(1 for r in rows if r["mode"] == "pure_atom_caratheodory_decomp_exact"),
        sum(1 for r in rows if r["mode"] == "sector_splitting_and_partial_additivity_exact"),
    ]
    m_labels = ["Mode 1\nQubit\nBloch", "Mode 2\nKI Block\nTensor", "Mode 3\nPure-Atom\nCarath.", "Mode 4\nSector &\nAdditiv."]
    bars = ax2.bar(range(4), counts, color=["#1b6ca8", "#2a9d8f", "#e07a5f", "#c1121f"], edgecolor="#222", lw=0.7)
    for b, c in zip(bars, counts):
        ax2.text(b.get_x() + b.get_width() / 2.0, c + 0.15, str(c), ha="center", va="bottom", fontweight="bold", fontsize=8.5)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(m_labels, fontsize=7.6)
    ax2.set_ylabel("Certified instances")
    ax2.set_ylim(0, 5.4)
    ax2.set_title("(b) Coverage Across All 4 Modes")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_atlas_summary.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    make_fig1()
    make_fig2()
    make_fig3()
    print("[PASS] Generated all 3 publication figures in paper/figures/.")
