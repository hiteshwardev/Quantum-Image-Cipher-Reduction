"""Figure style, layout verification and export.

Every figure is drawn at the final printed width (16 cm, the text width of the
manuscript and below the 17.4 cm limit of the journal) so that lettering
appears at its true size. Axes are placed explicitly in centimetres, with a
reserved column on the right for the legend, so that only plotted data sits
inside a plot box. :func:`check_layout` enforces three rules before a figure
is saved:

1. no text and no legend intrudes into a plot box;
2. no two text elements overlap;
3. no text is clipped at the canvas edge.

Colours follow the Okabe-Ito palette, and every categorical contrast also
carries a marker shape or a hatch pattern, so the figures remain readable in
greyscale and under colour-vision deficiency.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

OKABE_ITO = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9",
             "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
             "vermillion": "#D55E00", "purple": "#CC79A7"}
GREY = "#6E6E6E"
BAND = "#E6E6E6"

BROKEN = {"color": OKABE_ITO["vermillion"], "marker": "o"}
RESISTANT = {"color": OKABE_ITO["blue"], "marker": "s"}

WIDTH_CM = 16.0
CM = 1 / 2.54
DPI = 600

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "stixsans",
    "font.size": 9.0,
    "axes.labelsize": 9.0,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "axes.linewidth": 0.8,
    "axes.labelpad": 5.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.handlelength": 2.2,
    "hatch.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "lines.markersize": 5.0,
    "figure.dpi": 150,
    "savefig.dpi": DPI,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "image.interpolation": "nearest",
}


def apply_style() -> None:
    plt.rcParams.update(STYLE)


def figure(height_cm: float, legend_cm: float = 0.0, left_cm: float = 1.9,
           bottom_cm: float = 1.25, top_cm: float = 0.3, right_cm: float = 0.3):
    """Figure of the final width with one plot box placed in centimetres.

    ``legend_cm`` reserves a column of that width on the right for a legend
    drawn with :func:`legend`. Returns ``(fig, ax)``.
    """
    fig = plt.figure(figsize=(WIDTH_CM * CM, height_cm * CM))
    width = WIDTH_CM - left_cm - right_cm - legend_cm
    height = height_cm - bottom_cm - top_cm
    ax = fig.add_axes((left_cm / WIDTH_CM, bottom_cm / height_cm,
                       width / WIDTH_CM, height / height_cm))
    fig.qic_legend_x = (left_cm + width + 0.45) / WIDTH_CM
    fig.qic_legend_top = (height_cm - top_cm) / height_cm
    return fig, ax


def legend(fig, handles, labels=None, title: str | None = None, **kwargs):
    """Legend in the reserved right-hand column, aligned with the top of the plot box."""
    if labels is None:
        labels = [h.get_label() for h in handles]
    kw = dict(loc="upper left", bbox_to_anchor=(fig.qic_legend_x, fig.qic_legend_top),
              bbox_transform=fig.transFigure, borderaxespad=0.0, alignment="left",
              labelspacing=0.75)
    kw.update(kwargs)
    leg = fig.legend(handles, labels, title=title, **kw)
    if title:
        leg.get_title().set_fontsize(STYLE["legend.fontsize"])
    return leg


def marker_handle(style: dict, label: str, filled: bool = True,
                  edgecolor: str | None = None) -> Line2D:
    return Line2D([], [], linestyle="none", marker=style["marker"], color=style["color"],
                  markerfacecolor=style["color"] if filled else "white",
                  markeredgecolor=edgecolor or style["color"], label=label)


def dot_interval_plot(ax, rows: list[dict], generator: np.random.Generator,
                      jitter: float = 0.18) -> None:
    """Per-image values as jittered points, with the mean and its interval on top.

    Each row is a dict with ``values``, ``mean``, ``low``, ``high`` and
    ``style``; rows are drawn from top to bottom in the given order.
    """
    n = len(rows)
    for i, row in enumerate(rows):
        y = n - 1 - i
        v = np.asarray(row["values"], dtype=np.float64)
        offsets = generator.uniform(-jitter, jitter, v.size)
        st = row["style"]
        ax.plot(v, y + offsets, linestyle="none", marker=st["marker"], markersize=3.2,
                markerfacecolor="white", markeredgecolor=st["color"], markeredgewidth=0.6,
                alpha=0.75, zorder=2)
        ax.errorbar(row["mean"], y, xerr=[[row["mean"] - row["low"]], [row["high"] - row["mean"]]],
                    fmt=st["marker"], color="black", markerfacecolor=st["color"],
                    markeredgecolor="black", markersize=6.0, elinewidth=1.4, capsize=4.0,
                    capthick=1.4, zorder=4)
    ax.set_yticks(range(n))
    ax.set_yticklabels([row["label"] for row in reversed(rows)])
    ax.set_ylim(-0.6, n - 0.4)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, color="#C8C8C8", zorder=0)
    ax.set_axisbelow(True)


# ------------------------------------------------------------- layout check

_TOL = 0.5


def _extent(artist, renderer):
    if not artist.get_visible():
        return None
    try:
        bb = artist.get_window_extent(renderer=renderer)
    except (RuntimeError, ValueError):
        return None
    if bb.width <= 0 or bb.height <= 0:
        return None
    return bb


def _text_items(fig, renderer):
    items = []
    for i, ax in enumerate(fig.axes):
        for name, art in (("x label", ax.xaxis.label), ("y label", ax.yaxis.label),
                          ("title", ax.title)):
            bb = _extent(art, renderer)
            if bb is not None and art.get_text():
                items.append((f"axes {i} {name}", bb))
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = min(lim), max(lim)
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not lo - 1e-9 <= tick.get_loc() <= hi + 1e-9:
                    continue
                for lab in (tick.label1, tick.label2):
                    bb = _extent(lab, renderer)
                    if bb is not None and lab.get_text():
                        items.append((f"axes {i} tick '{lab.get_text()[:20]}'", bb))
        for t in ax.texts:
            bb = _extent(t, renderer)
            if bb is not None and t.get_text():
                items.append((f"axes {i} text '{t.get_text()[:20]}'", bb))
        if ax.get_legend() is not None:
            items.append((f"axes {i} legend", ax.get_legend().get_window_extent(renderer)))
    for leg in fig.legends:
        items.append(("figure legend", leg.get_window_extent(renderer)))
    for t in fig.texts:
        bb = _extent(t, renderer)
        if bb is not None and t.get_text():
            items.append((f"figure text '{t.get_text()[:20]}'", bb))
    return items


def _overlap(a, b) -> bool:
    return (min(a.x1, b.x1) - max(a.x0, b.x0) > _TOL
            and min(a.y1, b.y1) - max(a.y0, b.y0) > _TOL)


def check_layout(fig, schematic=()) -> list[str]:
    """Return a list of layout problems; an empty list means the figure passes.

    ``schematic`` lists axes whose interior text is the content of the
    figure (a diagram); they are exempt from rule 1 only.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    items = _text_items(fig, renderer)
    boxes = [ax.get_window_extent(renderer) for ax in fig.axes
             if ax not in schematic and ax.axison]
    problems = []
    for name, bb in items:
        if any(_overlap(bb, box) for box in boxes):
            problems.append(f"{name} intrudes into a plot box")
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            (n1, b1), (n2, b2) = items[i], items[j]
            if _overlap(b1, b2):
                problems.append(f"{n1} overlaps {n2}")
    canvas = fig.bbox
    for name, bb in items:
        if (bb.x0 < canvas.x0 - _TOL or bb.x1 > canvas.x1 + _TOL
                or bb.y0 < canvas.y0 - _TOL or bb.y1 > canvas.y1 + _TOL):
            problems.append(f"{name} is clipped at the canvas edge")
    return problems


def save(fig, stem: str, directory: str | Path, schematic=()) -> Path:
    """Verify the layout, then write ``stem.png`` (600 dpi) and ``stem.pdf``."""
    problems = check_layout(fig, schematic)
    if problems:
        raise AssertionError(f"{stem}: " + "; ".join(problems))
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    png = directory / f"{stem}.png"
    fig.savefig(png, dpi=DPI, metadata={"Software": None})
    fig.savefig(directory / f"{stem}.pdf", metadata={"Creator": None, "Producer": None,
                                                     "CreationDate": None})
    return png
