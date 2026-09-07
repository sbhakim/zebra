"""Generate evidence figures from machine-readable artifacts, never literals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

# Semantic families, shared by both panels and matched to the TikZ figures:
# rule-guided is the teal of the rule map in Fig. 1, rehearsal is the salmon of
# the replay buffer. A strategy therefore keeps one colour across the paper.
FAMILY: dict[str, tuple[str, str]] = {
    "no-cl": ("none", "muted"),
    "zebra": ("rule-guided", "symbolc"),
    "lwf": ("distillation", "neuralc"),
    "ewc": ("regularisation", "driftc"),
    "si": ("regularisation", "driftc"),
    "gen-replay": ("rehearsal", "bufferc"),
    "replay": ("rehearsal", "bufferc"),
}

# Display names, shared by both panels so a strategy reads identically in each.
LABEL = {
    "no-cl": "no-CL",
    "ewc": "EWC",
    "si": "SI",
    "lwf": "LwF",
    "gen-replay": "gen. replay",
    "replay": "replay",
    "zebra": "ZeBRa",
}


def memory_figure(source: Path, destination: Path) -> None:
    """Adaptation-tensor payload per strategy against RFC 7228 RAM envelopes."""
    import matplotlib.pyplot as plt

    from zebra.eval.figstyle import COL_W, PANEL_H, apply, save

    apply()
    from zebra.eval.figstyle import MUTED, PALETTE

    def hue(name: str) -> str:
        fam = FAMILY[name][1]
        return MUTED if fam == "muted" else PALETTE[fam]["ink"]

    payload = cast(dict[str, dict[str, Any]], json.loads(source.read_text()))
    order = ("no-cl", "zebra", "lwf", "ewc", "si", "gen-replay", "replay")
    missing = [name for name in order if name not in payload]
    if missing:
        raise ValueError(f"memory artifact missing strategies: {missing}")

    fields = (
        "model_bytes",
        "retained_bytes",
        "optimiser_bytes",
        "gradient_bytes",
        "preprocessor_bytes",
        "static_rule_bytes",
    )
    totals = [sum(int(payload[n].get(f, 0)) for f in fields) for n in order]
    labels = [LABEL.get(n, n) for n in order]

    fig, ax = plt.subplots(figsize=(COL_W, PANEL_H))
    # Dots, not bars: on a log axis a bar's length encodes the ratio to the
    # axis minimum rather than the value, so no-CL at 17 kB would read as
    # comparable to replay at 1.15 MB. The stem just leads the eye to the label.
    colours = [hue(n) for n in order]
    left = min(totals) / 2.0
    for i, (value, colour) in enumerate(zip(totals, colours, strict=True)):
        ax.plot([left, value], [i, i], color="0.80", lw=0.8, zorder=1)
        ax.plot([value], [i], "o", color=colour, markersize=5, zorder=3)

    for kib, name in ((10, "Class 1"), (50, "Class 2")):
        ax.axvline(kib * 1024, color="0.45", lw=0.9, ls="--", zorder=0)
        ax.text(
            kib * 1024,
            len(order) - 0.35,
            f" {name}",
            fontsize=5.5,
            color="0.35",
            ha="left",
            va="center",
        )

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xscale("log")
    ax.set_xlabel("Resident adaptation state (bytes, log scale)", fontsize=7)
    ax.set_xlim(left=left)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.grid(axis="y", visible=False)
    save(fig, destination)


def retention_figure(memory_source: Path, results_source: Path, destination: Path) -> None:
    """Backward transfer against retained CL state, as a cost--quality frontier.

    The figure carries three facts the previous scatter left to the caption:
    which points are Pareto-optimal, how flat the frontier is between them, and
    where the RFC 7228 Class 2 budget cuts the axis. The budget line is drawn on
    the *retained*-byte axis, so it is the headroom left after the fixed
    adaptation cost, not the class capacity itself.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from zebra.eval.figstyle import COL_W, apply, save

    tab10 = apply()

    # This panel keeps matplotlib's default tab10 cycle rather than the
    # manuscript palette: seven arms sit inside two decades here, and tab10
    # separates them further than five tuned hues do. The family grouping is
    # the same as elsewhere, only the hues differ.
    CYCLE = {
        "no-cl": "0.45",
        "zebra": tab10[2],
        "lwf": tab10[0],
        "ewc": tab10[4],
        "si": tab10[4],
        "gen-replay": tab10[1],
        "replay": tab10[1],
    }

    def hue(name: str) -> str:
        return CYCLE[name]

    memory = cast(dict[str, dict[str, Any]], json.loads(memory_source.read_text()))
    results = cast(dict[str, Any], json.loads(results_source.read_text()))
    strategies = results["strategies"]
    names = [n for n in strategies if n in memory]
    if not names:
        raise ValueError("results and memory artifacts have no strategies in common")

    # Bytes cannot be negative, so zero-retention arms sit on a dedicated tick
    # left of the log region rather than at a fabricated small positive value.
    zero_position, linthresh = 8.0, 16.0

    points = []
    for name in names:
        est = strategies[name]["final_bwt_auc"]
        raw = int(memory[name]["retained_bytes"]) + int(
            memory[name].get("static_rule_bytes", 0)
        )
        points.append((name, float(raw), max(float(raw), zero_position), est))

    # Pareto frontier: fewer retained bytes and less forgetting both count as
    # better, so a point is dominated when another is cheaper and forgets less.
    def dominated(item: Any) -> bool:
        _, raw, _, est = item
        y = float(est["mean"])
        return any(
            other_raw <= raw and float(other["mean"]) > y
            for _, other_raw, _, other in points
            if other is not est
        )

    frontier = sorted(
        (p for p in points if not dominated(p)), key=lambda item: item[1]
    )

    # Class 2 headroom on the retained axis: capacity minus everything the
    # device must hold regardless of strategy. Reproduces the fits_adapting
    # flags in the artifact exactly.
    fixed = min(
        int(m["model_bytes"])
        + int(m["optimiser_bytes"])
        + int(m["gradient_bytes"])
        + int(m["preprocessor_bytes"])
        for m in memory.values()
    )
    headroom = 50 * 1024 - fixed

    fig, ax = plt.subplots(figsize=(COL_W, 2.95))

    # --- infeasible region, drawn first so everything else sits on top ---
    ax.axvspan(headroom, 1e7, color="0.88", zorder=0, lw=0)
    ax.axvline(headroom, color="0.45", lw=0.9, ls="--", zorder=1)

    # --- Pareto frontier ---
    ax.step(
        [p[2] for p in frontier],
        [float(p[3]["mean"]) for p in frontier],
        where="post",
        color="0.35",
        lw=0.9,
        zorder=2,
    )

    frontier_names = {p[0] for p in frontier}
    for name, _, x, est in points:
        y = float(est["mean"])
        colour = hue(name)
        on_front = name in frontier_names
        ax.errorbar(
            x,
            y,
            yerr=[[y - float(est["low"])], [float(est["high"]) - y]],
            fmt="o" if on_front else "o",
            color=colour,
            markerfacecolor=colour if on_front else "white",
            markeredgecolor=colour,
            markeredgewidth=0.9,
            capsize=2,
            elinewidth=0.9,
            zorder=3,
        )

    # Offsets are hand-placed, with alignment carried alongside: four strategies
    # sit inside one decade, so a label that clears its own whisker can still
    # land on a neighbour's. Alternating parity was not enough.
    OFFSET = {
        "no-cl": (6, 5, "left"),
        "zebra": (5, -12, "left"),
        "lwf": (-19, 9, "left"),
        "ewc": (-17, -12, "left"),
        "si": (-13, 11, "left"),
        "gen-replay": (-6, -48, "left"),
        "replay": (-25, 8, "left"),
    }
    for name, _, x, est in points:
        dx, dy, align = OFFSET[name]
        ax.annotate(
            LABEL.get(name, name),
            (x, float(est["mean"])),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=6.5,
            ha=align,
            color=hue(name),
            zorder=4,
        )

    lookup = {p[0]: (p[2], float(p[3]["mean"])) for p in points}

    # --- the headline trade, drawn rather than left to the caption ---
    lwf_x, lwf_y = lookup["lwf"]
    rep_x, rep_y = lookup["replay"]
    bar_y = 0.0042
    ax.annotate(
        "",
        xy=(rep_x, bar_y),
        xytext=(lwf_x, bar_y),
        arrowprops={"arrowstyle": "<->", "color": "0.30", "lw": 0.8},
    )
    ax.text(
        (lwf_x * rep_x) ** 0.5,
        bar_y + 0.0012,
        f"{rep_x / lwf_x:.0f}$\\times$ state  $\\rightarrow$  "
        f"{abs(rep_y - lwf_y):.3f} BWT",
        ha="center",
        va="bottom",
        fontsize=6.5,
        color="0.25",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2},
    )
    for lx, ly in ((lwf_x, lwf_y), (rep_x, rep_y)):
        ax.plot([lx, lx], [ly, bar_y], color="0.60", lw=0.5, ls=":", zorder=1)

    ax.text(
        1.9e6,
        -0.0295,
        "exceeds Class 2",
        fontsize=6.5,
        color="0.35",
        rotation=90,
        ha="left",
        va="center",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
        zorder=4,
    )
    # Set horizontally in the strip above the zero rule: rotated in the plot
    # body it is tall enough to cross that rule, which is a solid reference
    # line rather than a faint gridline.
    ax.text(
        0.02,
        0.955,
        "less forgetting $\\uparrow$",
        transform=ax.transAxes,
        fontsize=6.5,
        color="0.45",
        ha="left",
        va="center",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
        zorder=4,
    )

    ax.grid(axis="x", visible=False)
    ax.axhline(0.0, color="0.45", lw=0.9, zorder=1)
    ax.set_xscale("symlog", linthresh=linthresh)
    ax.set_xlim(zero_position * 0.6, 4e6)
    ax.set_ylim(-0.0505, 0.0088)
    ax.set_xlabel("Retained CL state (bytes, zero at left)")
    ax.set_ylabel("Final backward transfer (AUC)")
    save(fig, destination)


def main(source: Path, output: Path, results: Path | None = None) -> int:
    memory_figure(source, output / "memory_classes.pdf")
    print(f"wrote {output / 'memory_classes.pdf'}")
    if results is not None:
        retention_figure(source, results, output / "retention_vs_bytes.pdf")
        print(f"wrote {output / 'retention_vs_bytes.pdf'}")
    return 0
