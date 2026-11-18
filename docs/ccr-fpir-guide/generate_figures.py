"""Reproduce the threshold guide figures and verify its numeric examples."""

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parent.parent
RESULTS = ROOT / "results/part6-openset/scheme-B-only"
BLUE = "#2678b2"
ORANGE = "#bc5525"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def counts(present, correct, absent, tau):
    good = sum(ok and score >= tau for score, ok in zip(present, correct))
    wrong = sum(not ok and score >= tau for score, ok in zip(present, correct))
    false = sum(score >= tau for score in absent)
    return good, wrong, false


def point(present, correct, absent, tau):
    good, _, false = counts(present, correct, absent, tau)
    return false / len(absent), good / len(present)


def axes_style(ax):
    ax.set_xlabel("False positive identification rate (FPIR)")
    ax.set_ylabel("Correct accepted identification rate (CCR)")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)


def mark(ax, xy, label, offset, color=ORANGE):
    ax.scatter(*xy, s=38, color=color, zorder=4, edgecolor="white", linewidth=0.7)
    ax.annotate(
        label, xy, xytext=offset, textcoords="offset points", fontsize=9,
        color=color, ha="left" if offset[0] >= 0 else "right",
        va="bottom" if offset[1] >= 0 else "top",
        arrowprops={"arrowstyle": "-", "color": color, "lw": 0.65},
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none", "pad": 1.5},
    )


def save(fig, name):
    for extension in ("png", "pdf"):
        fig.savefig(OUTPUT / f"{name}.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def toy_figure():
    present = [0.98, 0.90, 0.95, 0.75, 0.60]
    correct = [True, True, False, True, True]
    absent = [0.82, 0.65, 0.94, 0.55, 0.40]
    thresholds = [math.inf] + sorted(set(present + absent), reverse=True) + [0.0]
    expected = [
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1),
        (2, 1, 1), (2, 1, 2), (3, 1, 2), (3, 1, 3),
        (4, 1, 3), (4, 1, 4), (4, 1, 5), (4, 1, 5),
    ]
    assert [counts(present, correct, absent, t) for t in thresholds] == expected
    assert point(present, correct, absent, 0.90) == (0.2, 0.4)
    assert point(present, correct, absent, 0.85) == (0.2, 0.4)

    xy = [point(present, correct, absent, t) for t in thresholds]
    fig, ax = plt.subplots(figsize=(8.8, 5.8), layout="constrained")
    ax.step(*zip(*xy), where="post", color=BLUE, lw=2)
    labels = [
        (math.inf, "All rejected", (15, 12)),
        (0.98, "tau = 0.98 / 0.95", (18, 14)),
        (0.94, "tau = 0.94", (15, -15)),
        (0.90, "tau = 0.90\n(FPIR 20%, CCR 40%)", (-15, 17)),
        (0.82, "tau = 0.82", (16, -15)),
        (0.75, "tau = 0.75", (-15, 15)),
        (0.65, "tau = 0.65", (16, -15)),
        (0.60, "tau = 0.60", (-14, 14)),
        (0.55, "tau = 0.55", (8, 14)),
        (0.40, "tau = 0.40 / 0\nAll accepted", (-10, -20)),
    ]
    for tau, label, offset in labels:
        mark(ax, point(present, correct, absent, tau), label, offset)
    axes_style(ax)
    ax.set(xlim=(-0.035, 1.04), ylim=(-0.035, 0.98),
           title="Five invented objects: each threshold gives one aggregate point")
    ax.text(0.55, 0.08, "Lower tau: accept more searches\nPredictions stay fixed",
            transform=ax.transAxes, fontsize=10, color="#444444")
    save(fig, "toy-thresholds")


def actual_figure():
    rows = read_csv(RESULTS / "scheme-B-top1-results.csv")
    present = [float(r["present_top_probability"]) for r in rows]
    correct = [int(r["present_top_is_true"]) == 1 for r in rows]
    absent = [float(r["absent_top_probability"]) for r in rows]
    assert len(rows) == 15080
    assert len({r["target"] for r in rows}) == 52
    assert len({r["sample"] for r in rows}) == 754

    expected = {
        0.9: (0, 0, 0), 0.7: (186, 1, 5), 0.534: (1051, 31, 146),
        0.5: (1420, 67, 263), 0.436: (2328, 270, 726),
        0.386: (3281, 637, 1500), 0.3: (5588, 2157, 4437),
        0.2: (8230, 5674, 12461), 0.1: (8522, 6558, 15080),
        0.0: (8522, 6558, 15080),
    }
    for tau, value in expected.items():
        assert counts(present, correct, absent, tau) == value, tau

    curve = read_csv(RESULTS / "scheme-B/ccr-fpir-curve.csv")
    for row in curve:
        x, y = point(present, correct, absent, float(row["tau"]))
        assert math.isclose(x, float(row["FPIR"]), abs_tol=1e-12)
        assert math.isclose(y, float(row["CCR"]), abs_tol=1e-12)
    for row in read_csv(RESULTS / "scheme-B-entropy-vs-probability-operating-points.csv"):
        if row["rule"] == "maximum_probability":
            x, y = point(present, correct, absent, float(row["threshold"]))
            assert math.isclose(x, float(row["FPIR"]), abs_tol=1e-12)
            assert math.isclose(y, float(row["CCR"]), abs_tol=1e-12)
            cap = float(row["fpir_cap"])
            assert x <= cap
            assert math.isclose(y, max(float(r["CCR"]) for r in curve
                                       if float(r["FPIR"]) <= cap), abs_tol=1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), layout="constrained")
    for ax in axes:
        ax.step([float(r["FPIR"]) for r in curve],
                [float(r["CCR"]) for r in curve], where="post", color=BLUE, lw=2)
        axes_style(ax)
    axes[0].set(xlim=(-0.015, 1.035), ylim=(-0.01, 0.63), title="Scheme B: full curve")
    axes[1].set(xlim=(-0.002, 0.105), ylim=(-0.005, 0.255),
                title="FPIR up to 10% (both axes enlarged)")
    for tau, offset in [(0.7, (18, 3)), (0.5, (28, -7)), (0.3, (23, -22)),
                        (0.2, (-12, 15)), (0.1, (-16, -22))]:
        label = "tau = 0.1 / 0" if tau == 0.1 else f"tau = {tau:g}"
        mark(axes[0], point(present, correct, absent, tau), label, offset)
    for tau, offset in [(0.534, (25, -22)), (0.5, (28, 12)),
                        (0.436, (20, -30)), (0.386, (-8, 15))]:
        x, y = point(present, correct, absent, tau)
        mark(axes[1], (x, y), f"tau = {tau:g}\n({x:.2%}, {y:.2%})", offset)
    fig.suptitle("Fixed saved predictions; sweep the acceptance threshold", fontsize=14)
    save(fig, "scheme-b-thresholds")
    print(f"Verified {len(curve)} saved curve rows, all guide counts and probability operating points.")


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "pdf.fonttype": 42})
    toy_figure()
    actual_figure()
    print(f"Saved PNG and PDF figures to {OUTPUT}")
