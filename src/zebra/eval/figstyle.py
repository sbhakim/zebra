"""Shared figure style, ported from the CTIForge set.

Kept rather than re-derived: one figure size everywhere (IEEEtran's
``\\columnwidth`` is ~3.5 in), 8 pt type just under the 9 pt body, vector PDF
for the paper and PNG only for previewing. ``pdf.fonttype`` 42 is what stops
IEEE PDF eXpress flagging a Type-3 font. No axes titles -- the caption has it.
"""

from __future__ import annotations

from typing import Any

# Manuscript palette. Hues are anchored on the project's reference diagram and
# the lightness of each family is solved to a distinct contrast on white, so the
# five inks stay legible at 8 pt and remain separable in a greyscale print.
# These values are the single source shared with the \definecolor block in
# main.tex; changing one without the other splits the paper's visual identity.
PALETTE: dict[str, dict[str, str]] = {
    "neuralc": {"ink": "#2F5473", "mid": "#5080A9", "tint": "#D5DFE6", "wash": "#F2F4F7"},
    "symbolc": {"ink": "#2F7E75", "mid": "#51B3A7", "tint": "#D4E7E5", "wash": "#F2F7F6"},
    "bufferc": {"ink": "#904A3B", "mid": "#B8776A", "tint": "#E6D8D5", "wash": "#F7F3F2"},
    "freec": {"ink": "#30764D", "mid": "#51AB77", "tint": "#D5E6DC", "wash": "#F2F7F4"},
    "driftc": {"ink": "#332B75", "mid": "#554BAC", "tint": "#D6D4E7", "wash": "#F2F2F7"},
}
INK, MUTED, GRID = "#1F2933", "#667085", "#D0D5DD"
PANEL, PALE, SHELL = "#E4E4EC", "#E8EEF5", "#F8F8F8"


def ink(family: str) -> str:
    """Line and marker colour for a semantic family."""
    return PALETTE[family]["ink"]


COL_W, COL_H = 3.45, 2.45  # inches, single column
PANEL_H = 1.90  # stacked subpanel height


def apply() -> list[str]:
    """Install the shared rcParams and return the tab10 cycle."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.figsize": (COL_W, COL_H),
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.3,
            "lines.markersize": 3.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    cycle: list[str] = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return cycle


def save(fig: Any, destination: Any) -> None:
    """Vector PDF for the paper, PNG beside it for previewing."""
    import matplotlib.pyplot as plt

    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination)
    fig.savefig(destination.with_suffix(".png"))
    plt.close(fig)
