"""Shared figure colours: one fixed colour per method in every figure, never reassigned.

Hues are the first slots of a colour-blind-checked categorical palette. Each figure shows at most four methods; both
sets used together pass a colour-vision-deficiency check on all pairs (lightness band, chroma floor, CVD and
normal-vision separation):
  models     TIGER, SASRec-CE, SASRec-BCE, Popularity                       (main results, buckets, prefixes)
  cold start TIGER, TIGER (exact-scored unseen), Hybrid, Semantic-KNN       (cold-start figure)
The two cold-start-only colours are not checked against SASRec-CE / SASRec-BCE / Popularity: they never share a
figure. Text, reference lines and chance marks use ink colours, never a method colour.
"""

MODEL_COLORS = {
    "TIGER": "#2a78d6",  # blue
    "SASRec-CE": "#eb6834",  # orange
    "SASRec-BCE": "#1baf7a",  # aqua
    "Popularity": "#eda100",  # yellow
    "TIGER (exact-scored unseen)": "#008300",  # green (cold start only)
    "Hybrid": "#e87ba4",  # magenta (cold start only: SASRec-CE + Semantic-KNN)
    "Semantic-KNN": "#4a3aa7",  # violet (cold start only)
}
SENSITIVITY_COLOR = "#8a8980"  # evaluation-only variants, told apart by line style and marker
INK = "#1f1f1e"  # primary text, reference marks
INK_MUTED = "#6b6a63"  # secondary text, captions
GRID = "#e4e3dc"


def style_axes(ax) -> None:
    """Recessive grid and spines."""
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK)
