"""
Shared chart styling for the summary notebooks.

One place for the palette and the matplotlib defaults, so every figure in
`notebooks/` reads as one system instead of nine different matplotlib
accidents.

Design rules this encodes
-------------------------
- Categorical hues are assigned in a fixed slot order and never cycled. Three
  slots are used at most; past that we facet instead of inventing hues.
- Sequential encoding (confusion matrices, anything that is a magnitude) uses
  one hue, light to dark. Never a rainbow.
- Marks stay thin and the chrome stays recessive: hairline solid gridlines, no
  top/right spines, muted axis ink.
- Text never wears a series colour. Identity comes from the coloured mark, not
  from coloured type.

The palette is the validated reference instance: the three categorical slots
clear the colour-vision-deficiency and normal-vision separation gates on all
pairs. Aqua sits slightly under 3:1 contrast on this surface, which is why
every chart in the notebooks ships with its underlying table visible.
"""

from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
import numpy as np


# ---------------------------------------------------------
# Palette
# ---------------------------------------------------------

# Categorical slots, in fixed assignment order.
SERIES = [
    "#2a78d6",   # 1 blue
    "#eb6834",   # 2 orange
    "#1baf7a",   # 3 aqua
]

BLUE, ORANGE, AQUA = SERIES

# Chart chrome and ink.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# De-emphasis grey, for context series in an emphasis chart.
CONTEXT = "#c3c2b7"

# Sequential blue ramp, light to dark, for magnitude encoding.
SEQUENTIAL_STEPS = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "sage_blue",
    SEQUENTIAL_STEPS,
)


# ---------------------------------------------------------
# Semantic roles used across the notebooks
# ---------------------------------------------------------

# SAGE is the thing everything is compared against, so it keeps one identity
# in every figure it appears in. Colour follows the entity, never its rank.
SAGE_COLOR = INK_SECONDARY

# The two cross-validation regimes. Grouped is the honest one, so it takes the
# leading slot; random is the cautionary comparison.
SPLIT_COLORS = {
    "grouped": BLUE,
    "random": ORANGE,
}

# A fourth series is a neutral, never a fourth generated hue. Where four
# series genuinely have to share an axis, the neutral marks the one that
# belongs to a different family (PointNet works on raw points; the other
# three work on the 13D features), so the extra channel carries meaning
# rather than just being "the next colour".
NEUTRAL_SERIES = INK_SECONDARY


# ---------------------------------------------------------
# Matplotlib defaults
# ---------------------------------------------------------

def apply_style():
    """
    Install the chart defaults for a notebook session.

    Call once near the top of a notebook, after importing matplotlib.
    """

    mpl.rcParams.update(
        {
            # Surface
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",

            # Type. One sans throughout, no display faces.
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Segoe UI",
                "DejaVu Sans",
                "sans-serif",
            ],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlepad": 12,
            "axes.labelsize": 10,
            "axes.labelpad": 8,

            # Ink
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,

            # Recessive chrome: hairline, solid, never dashed.
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "axes.axisbelow": True,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.size": 0,
            "ytick.major.size": 0,

            # Marks
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "lines.markersize": 7,

            # Legend: present, quiet, out of the way.
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.labelcolor": INK_SECONDARY,

            "axes.prop_cycle": mpl.cycler(color=SERIES),
        }
    )


# ---------------------------------------------------------
# Small helpers
# ---------------------------------------------------------

def percent_axis(
    ax,
    axis="y",
    lo=0,
    top=100,
    step=None,
):
    """
    Format an axis as whole percentages on clean, round ticks.

    Matplotlib's automatic locator happily produces 80/82/85/88/90 when a
    range is narrow. Ticks carry the values that are not directly labelled,
    so they get an explicit round step instead.

    Parameters
    ----------
    axis : {"y", "x"}
    lo, top : float
        Axis limits, in percent.
    step : float, optional
        Tick spacing. Chosen from the range when omitted.
    """

    if step is None:
        span = top - lo
        step = next(
            (
                candidate
                for candidate in (5, 10, 20, 25)
                if span / candidate <= 6
            ),
            20,
        )

    ticks = np.arange(lo, top + step / 2, step)

    target = ax.yaxis if axis == "y" else ax.xaxis

    target.set_major_formatter(
        mpl.ticker.FuncFormatter(
            lambda value, _: f"{value:.0f}%"
        )
    )

    if axis == "y":
        ax.set_ylim(lo, top)
        ax.set_yticks(ticks)
    else:
        ax.set_xlim(lo, top)
        ax.set_xticks(ticks)


def label_bars(
    ax,
    bars,
    values,
    horizontal=True,
    fmt="{:.1f}",
    color=INK_SECONDARY,
):
    """
    Put the value at the tip of each bar, outside the mark.

    Labels sit outside rather than inside so they can never be clipped by a
    short bar, and they stay in text ink rather than the series colour.
    """

    for bar, value in zip(bars, values):

        if horizontal:
            ax.annotate(
                fmt.format(value),
                xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                ha="left",
                fontsize=9,
                color=color,
            )
        else:
            ax.annotate(
                fmt.format(value),
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                va="bottom",
                ha="center",
                fontsize=9,
                color=color,
            )


def dumbbell(
    ax,
    labels,
    start_values,
    end_values,
    start_color=CONTEXT,
    end_color=BLUE,
    start_label=None,
    end_label=None,
):
    """
    Draw a before/after dumbbell, one row per label.

    The right form when the story is "this moved" per item: the connector
    carries the size of the change, and the two dots carry the endpoints.
    Dots get a surface ring so they stay legible where they overlap.
    """

    positions = range(len(labels))

    for position, start, end in zip(
        positions,
        start_values,
        end_values,
    ):
        ax.plot(
            [start, end],
            [position, position],
            color=AXIS,
            linewidth=1.5,
            zorder=1,
            solid_capstyle="round",
        )

    ax.scatter(
        start_values,
        list(positions),
        s=80,
        color=start_color,
        zorder=2,
        edgecolors=SURFACE,
        linewidths=2,
        label=start_label,
    )

    ax.scatter(
        end_values,
        list(positions),
        s=80,
        color=end_color,
        zorder=3,
        edgecolors=SURFACE,
        linewidths=2,
        label=end_label,
    )

    ax.set_yticks(list(positions))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)


def grouped_bars(
    ax,
    categories,
    series,
    bar_width=0.26,
    gap=0.88,
):
    """
    Draw a grouped bar chart with a real surface gap between neighbours.

    Bars within a group are separated by surface showing through, not by a
    stroke drawn around each mark.

    Parameters
    ----------
    categories : list of str
        X tick labels.

    series : dict
        Ordered mapping of ``label -> (values, color)``. Values are already
        in the plotted units.

    bar_width : float
        Slot width per series.

    gap : float
        Fraction of the slot the bar fills; the remainder is the gap.
    """

    positions = np.arange(len(categories))

    count = len(series)

    offsets = (
        np.arange(count) - (count - 1) / 2
    ) * bar_width

    for offset, (label, (values, color)) in zip(
        offsets,
        series.items(),
    ):
        ax.bar(
            positions + offset,
            values,
            bar_width * gap,
            label=label,
            color=color,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(categories)

    return positions
