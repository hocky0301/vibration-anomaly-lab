#!/usr/bin/env python3
"""
Render the README figures from the synthetic data and the results/ JSONs.

    python3 scripts/make_figures.py [--data PATH] [--results DIR] [--out DIR]

Every figure is drawn from files this repository regenerates itself:

    figures/01-sweep-structure.png       from data/synthetic/train.csv
    figures/02-amplitude-vs-setpoint.png from data/synthetic/train.csv
    figures/03-unit-fingerprints.png     from data/synthetic/train.csv
    figures/04-protocol-unit-scores.png  from results/default_draw.json
    figures/05-seed-sweep.png            from results/seed_sweep.json
    figures/06-robust-scale.png          from data/synthetic/train.csv
    figures/07-leakage-candidates.png    from vibration_anomaly_lab.metric_design

A figure whose input is missing is skipped with a notice (exit status stays
0), so `make figures` works right after `make verify`, before `make sweep`.
Nothing here hard-codes a unit id or a number: faulty units are read from the
label column / the JSON, and every plotted value is computed on the spot.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:                 # `python3 scripts/make_figures.py` puts scripts/ on the path, not the repo
    sys.path.insert(0, REPO)

# --------------------------------------------------------------------------- #
# style
# --------------------------------------------------------------------------- #
DPI = 150
WIDTH = 8.0                       # inches; 1200 px at 150 dpi
HEALTHY = "#9a9a9a"
FAULTY_COLOURS = ["#d95f02", "#7570b3"]   # orange, purple (ColorBrewer Dark2)
TITLE_PREFIX = "Synthetic data: "
STATS = ("rms", "std", "mad", "iqr", "p90", "absmean", "ptp")
PROTOCOL_ORDER = ("a", "b", "b_linear", "b_noctx", "c", "c_meanstd", "d", "e", "e_linear")

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": DPI,
})


def faulty_colour(i: int) -> str:
    """Colour for the i-th faulty unit (two distinct colours, then tab10)."""
    if i < len(FAULTY_COLOURS):
        return FAULTY_COLOURS[i]
    return plt.get_cmap("tab10")(i % 10)


def unit_colours(units: list[str], faulty: list[str]) -> dict[str, str]:
    cols = {u: HEALTHY for u in units}
    for i, u in enumerate(sorted(faulty)):
        cols[u] = faulty_colour(i)
    return cols


def save(fig: plt.Figure, out_dir: str, name: str, rect=None) -> None:
    fig.tight_layout(rect=rect)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"wrote {path}  ({os.path.getsize(path) / 1024:.0f} KB)")


def notice_skip(name: str, why: str) -> None:
    print(f"skip  {name}: {why}")


# --------------------------------------------------------------------------- #
# data helpers
# --------------------------------------------------------------------------- #
def load_train(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def trial_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per trial with unit_id, label, setpoint, speed_rpm and the 7 stats of vib_y.

    Stat definitions mirror the package (rms, std, mad about the median,
    IQR, 90th percentile of |v|, mean |v|, peak-to-peak).
    """
    rows = []
    for tid, g in df.groupby("trial_id", sort=True):
        v = g["vib_y"].to_numpy(dtype=np.float64)
        m = np.median(v)
        rows.append({
            "trial_id": tid,
            "unit_id": g["unit_id"].iloc[0],
            "label": int(g["label"].iloc[0]) if "label" in g else 0,
            "setpoint": float(g["setpoint"].iloc[0]),
            "speed_rpm": float(g["speed_rpm"].mean()),
            "rms": float(np.sqrt(np.mean(v * v))),
            "std": float(v.std()),
            "mad": float(np.median(np.abs(v - m))),
            "iqr": float(np.subtract(*np.percentile(v, [75, 25]))),
            "p90": float(np.percentile(np.abs(v), 90)),
            "absmean": float(np.abs(v).mean()),
            "ptp": float(v.max() - v.min()),
        })
    return pd.DataFrame(rows)


def faulty_units_of(T: pd.DataFrame) -> list[str]:
    return sorted(T.loc[T["label"] == 1, "unit_id"].unique().tolist())


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def fig01_sweep_structure(df: pd.DataFrame, out_dir: str) -> None:
    units = list(pd.unique(df["unit_id"]))          # order of appearance
    faulty = sorted(df.loc[df["label"] == 1, "unit_id"].unique().tolist())
    cols = unit_colours(units, faulty)
    sp = df["setpoint"].to_numpy()
    idx = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(WIDTH, 3.6))
    ax.plot(idx, sp, color="black", lw=0.6)
    # unit boundaries + shading of faulty units
    starts = df.groupby("unit_id", sort=False).head(1).index.to_numpy()
    ends = np.append(starts[1:], len(df))
    top = sp.max() + 0.12 * (sp.max() - sp.min())
    for u, s, e in zip(units, starts, ends):
        if u in faulty:
            ax.axvspan(s, e, color=cols[u], alpha=0.22, lw=0)
        ax.axvline(s, color="#cccccc", lw=0.6)
        ax.text((s + e) / 2, top, u, ha="center", va="bottom", fontsize=9,
                color=cols[u] if u in faulty else "#555555",
                fontweight="bold" if u in faulty else "normal")
    ax.set_ylim(sp.min() - 0.3, top + 0.35 * (sp.max() - sp.min()))
    ax.set_xlim(0, len(df))
    ax.set_xlabel("row index in train.csv")
    ax.set_ylabel("setpoint [a.u.]")
    ax.set_title(TITLE_PREFIX + "each unit is a block of ascending setpoint sweeps "
                 "(faulty units shaded)")
    save(fig, out_dir, "01-sweep-structure.png")


def fig02_amplitude_vs_setpoint(T: pd.DataFrame, out_dir: str) -> None:
    units = sorted(T["unit_id"].unique().tolist())
    faulty = faulty_units_of(T)
    cols = unit_colours(units, faulty)
    med = T.groupby(["unit_id", "setpoint"])["rms"].median().unstack("unit_id")
    grid = med.index.to_numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(WIDTH, 6.6), sharex=True)
    for u in units:
        if u in faulty:
            continue
        ax1.plot(grid, np.log(med[u]), color=HEALTHY, lw=1.2, alpha=0.9)
    for u in faulty:
        ax1.plot(grid, np.log(med[u]), color=cols[u], lw=2.2, marker="o", ms=4, label=f"{u} (faulty)")
    ax1.plot([], [], color=HEALTHY, lw=1.2, label="healthy units")
    ax1.set_ylabel("median log(vib_y RMS)  [log a.u.]")
    ax1.legend(loc="lower right", frameon=False)
    ax1.set_title(TITLE_PREFIX + "amplitude grows with the setpoint on every unit")

    healthy_ref = med[[u for u in units if u not in faulty]].median(axis=1)
    for u in faulty:
        ratio = med[u] / healthy_ref
        ax2.plot(grid, ratio, color=cols[u], lw=2.2, marker="o", ms=4, label=u)
    ax2.axhline(1.0, color="black", lw=0.8, ls="--")
    ax2.set_ylabel("faulty / healthy-median RMS ratio")
    ax2.set_xlabel("setpoint [a.u.]")
    ax2.set_title("the fault is a deficit at low setpoints that closes towards the top of the sweep")
    ax2.legend(loc="lower right", frameon=False)
    save(fig, out_dir, "02-amplitude-vs-setpoint.png")


def fig03_unit_fingerprints(T: pd.DataFrame, out_dir: str) -> None:
    units = sorted(T["unit_id"].unique().tolist())
    faulty = faulty_units_of(T)
    cols = unit_colours(units, faulty)
    grid = np.sort(T["setpoint"].unique())
    mid = float(grid[len(grid) // 2])
    S = T[np.isclose(T["setpoint"], mid)].copy()
    S["rpm_per_setpoint"] = S["speed_rpm"] / S["setpoint"]
    rng = np.random.default_rng(0)
    x_of = {u: i for i, u in enumerate(units)}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(WIDTH, 6.6), sharex=True)
    for ax, col, ylabel in ((ax1, "rms", "log(vib_y RMS) per trial  [log a.u.]"),
                            (ax2, "rpm_per_setpoint", "speed_rpm / setpoint  [rpm per a.u.]")):
        for u in units:
            g = S[S["unit_id"] == u]
            y = np.log(g[col].to_numpy()) if col == "rms" else g[col].to_numpy()
            x = x_of[u] + rng.uniform(-0.18, 0.18, size=len(y))
            ax.scatter(x, y, s=22, color=cols[u], alpha=0.85, edgecolor="none")
            ax.hlines(np.median(y), x_of[u] - 0.3, x_of[u] + 0.3, color="black", lw=1.2)
        ax.set_ylabel(ylabel)
    ax1.set_title(TITLE_PREFIX + f"per-trial values at one setpoint ({mid:g}); "
                  "units have fingerprints")
    ax2.set_title("the speed response also differs per unit (operating-point fingerprint)")
    ax2.set_xticks(range(len(units)))
    ax2.set_xticklabels(units, rotation=45, ha="right")
    for lab in ax2.get_xticklabels():
        if lab.get_text() in faulty:
            lab.set_color(cols[lab.get_text()])
            lab.set_fontweight("bold")
    ax2.set_xlabel("unit (faulty units in colour)")
    save(fig, out_dir, "03-unit-fingerprints.png")


def fig04_protocol_unit_scores(doc: dict, out_dir: str) -> None:
    ids = [p for p in ("a", "b", "c", "e") if p in doc.get("protocols", {})]
    if not ids:
        notice_skip("04-protocol-unit-scores.png", "no protocols a/b/c/e in default_draw.json")
        return
    faulty = sorted(doc.get("faulty_units", []))
    units = sorted(doc["protocols"][ids[0]]["unit_scores"].keys())
    cols = unit_colours(units, faulty)
    proto_cols = {"a": "#c7c7c7", "b": "#8c8c8c", "c": "#4d4d4d", "e": "#111111"}
    labels = {"a": "a: random K-fold (leaky)", "b": "b: leave-one-unit-out, raw",
              "c": "c: LOUO, conditional robust-z", "e": "e: LOUO, unit-relative"}
    n = len(ids)
    w = 0.8 / n
    x = np.arange(len(units))

    fig, ax = plt.subplots(figsize=(WIDTH, 4.2))
    for j, pid in enumerate(ids):
        sc = doc["protocols"][pid]["unit_scores"]
        y = [float(sc.get(u, np.nan)) for u in units]
        ax.bar(x + (j - (n - 1) / 2) * w, y, width=w, color=proto_cols.get(pid, "#333333"),
               label=labels.get(pid, pid), edgecolor="none")
    for u in faulty:
        if u in units:
            ax.axvspan(x[units.index(u)] - 0.5, x[units.index(u)] + 0.5,
                       color=cols[u], alpha=0.18, lw=0)
    ax.axhline(0.5, color="black", lw=0.9, ls="--")
    ax.text(len(units) - 0.5, 0.52, "threshold 0.5", ha="right", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(units, rotation=45, ha="right")
    for lab in ax.get_xticklabels():
        if lab.get_text() in faulty:
            lab.set_color(cols[lab.get_text()])
            lab.set_fontweight("bold")
    ax.set_ylabel("unit score (median out-of-fold p)")
    ax.set_xlabel("unit (faulty units shaded)")
    ax.set_title(TITLE_PREFIX + "per-unit scores by protocol, default draw")
    fig.legend(loc="lower center", ncol=2, frameon=False)
    save(fig, out_dir, "04-protocol-unit-scores.png", rect=(0, 0.14, 1, 1))


def fig05_seed_sweep(docs: list[dict], out_dir: str) -> None:
    ids = [p for p in PROTOCOL_ORDER if all(p in d.get("protocols", {}) for d in docs)]
    if not ids:
        notice_skip("05-seed-sweep.png", "no common protocols in seed_sweep.json")
        return
    rng = np.random.default_rng(0)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(WIDTH, 6.8), sharex=True)
    for ax, key, ylabel in ((ax1, "unit_f1_at_0_5", "unit F1 @ 0.5"),
                            (ax2, "trial_f1_at_0_5", "trial F1 @ 0.5")):
        data = [np.array([float(d["protocols"][p][key]) for d in docs]) for p in ids]
        ax.boxplot(data, positions=range(len(ids)), widths=0.55, showfliers=False,
                   medianprops={"color": "black", "lw": 1.4},
                   boxprops={"color": "#555555"}, whiskerprops={"color": "#555555"},
                   capprops={"color": "#555555"})
        for i, y in enumerate(data):
            ax.scatter(i + rng.uniform(-0.16, 0.16, size=len(y)), y, s=14,
                       color=FAULTY_COLOURS[0], alpha=0.55, edgecolor="none")
        ax.axhline(0.5, color="#bbbbbb", lw=0.8, ls=":")
        ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel(ylabel)
    ax1.set_title(TITLE_PREFIX + f"F1 across {len(docs)} generator seeds, one dot per seed")
    ax2.set_xticks(range(len(ids)))
    ax2.set_xticklabels(ids, rotation=35, ha="right")
    ax2.set_xlabel("protocol")
    save(fig, out_dir, "05-seed-sweep.png")


def fig06_robust_scale(T: pd.DataFrame, out_dir: str) -> None:
    H = T[T["label"] == 0]
    if H.empty:
        notice_skip("06-robust-scale.png", "no healthy trials in train.csv")
        return
    cv = {}
    for s in STATS:
        g = H.groupby(["unit_id", "setpoint"])[s]
        cells = (g.std(ddof=1) / g.mean()).dropna()
        cv[s] = float(cells.median())
    order = sorted(STATS, key=lambda s: cv[s])
    robust = {"mad", "iqr", "p90"}
    fig, ax = plt.subplots(figsize=(WIDTH, 3.8))
    ax.bar(range(len(order)), [cv[s] for s in order],
           color=[FAULTY_COLOURS[1] if s in robust else HEALTHY for s in order], edgecolor="none")
    for i, s in enumerate(order):
        ax.text(i, cv[s], f"{cv[s]:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel("median CV within (unit, setpoint) cell")
    ax.set_xlabel("vib_y statistic (robust statistics in colour)")
    ax.set_ylim(0, max(cv.values()) * 1.2)
    ax.set_title(TITLE_PREFIX + "trial-to-trial spread of each statistic, healthy trials")
    save(fig, out_dir, "06-robust-scale.png")


def fig07_leakage_candidates(train_csv: str, out_dir: str, n_truths: int = 20,
                             k_max: int = 12) -> None:
    try:
        import vibration_anomaly_lab.metric_design as md
    except ImportError as exc:
        notice_skip("07-leakage-candidates.png", f"cannot import vibration_anomaly_lab: {exc}")
        return
    ids, rows, _labels = md.load_units(train_csv)
    G = len(ids)
    designs = [
        ("no mitigation (raw float)", {"displayed_digits": None}, None),
        ("1-digit rounding", {"displayed_digits": 1}, None),
        ("public/private split by ROW (50%)", {"displayed_digits": None}, "row"),
        ("public/private split by GROUP (50%)", {"displayed_digits": None}, "group"),
    ]
    styles = [{"color": "black", "ls": "-"}, {"color": FAULTY_COLOURS[0], "ls": "-"},
              {"color": HEALTHY, "ls": "--"}, {"color": FAULTY_COLOURS[1], "ls": "-"}]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(WIDTH, 4.4))
    for (name, kw, split), st in zip(designs, styles):
        traces = np.zeros((n_truths, k_max + 1))
        for t in range(n_truths):
            public = None
            if split is not None:
                public = md.mitigation_public_private_split(rows, 0.5, rng, by=split)
            res = md.leakage_budget(G, rows, n_submissions=k_max, strategy="greedy",
                                    rng=rng, public_rows_per_group=public, **kw)
            left = np.full(k_max + 1, float(res.prior_candidates))
            for i, _score, remaining in res.trace:
                left[i:] = remaining
            traces[t] = left
        med = np.median(traces, axis=0)
        ax.plot(range(k_max + 1), med, marker="o", ms=4, lw=2, label=name, **st)
    ax.set_yscale("log")
    ax.set_xlabel("scored submissions")
    ax.set_ylabel("label vectors still possible (median)")
    ax.set_xticks(range(0, k_max + 1, 2))
    ax.set_title(TITLE_PREFIX + f"how fast {G} hidden unit labels leak through the score")
    ax.legend(frameon=False, loc="upper right")
    save(fig, out_dir, "07-leakage-candidates.png")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data", default=os.path.join(REPO, "data", "synthetic", "train.csv"))
    ap.add_argument("--results", default=os.path.join(REPO, "results"))
    ap.add_argument("--out", default=os.path.join(REPO, "figures"))
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    if os.path.exists(args.data):
        df = load_train(args.data)
        T = trial_table(df)
        fig01_sweep_structure(df, args.out)
        fig02_amplitude_vs_setpoint(T, args.out)
        fig03_unit_fingerprints(T, args.out)
        fig06_robust_scale(T, args.out)
        fig07_leakage_candidates(args.data, args.out)
    else:
        for name in ("01-sweep-structure.png", "02-amplitude-vs-setpoint.png",
                     "03-unit-fingerprints.png", "06-robust-scale.png",
                     "07-leakage-candidates.png"):
            notice_skip(name, f"{args.data} not found (run `make data`)")

    default_draw = os.path.join(args.results, "default_draw.json")
    if os.path.exists(default_draw):
        with open(default_draw) as fh:
            fig04_protocol_unit_scores(json.load(fh), args.out)
    else:
        notice_skip("04-protocol-unit-scores.png", f"{default_draw} not found (run `make verify`)")

    sweep = os.path.join(args.results, "seed_sweep.json")
    if os.path.exists(sweep):
        with open(sweep) as fh:
            docs = json.load(fh)
        if isinstance(docs, list) and docs:
            fig05_seed_sweep(docs, args.out)
        else:
            notice_skip("05-seed-sweep.png", f"{sweep} holds no per-seed documents")
    else:
        notice_skip("05-seed-sweep.png", f"{sweep} not found (run `make sweep`)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
