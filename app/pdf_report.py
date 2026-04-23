
"""
One-page landscape PDF reports for the Home tab.

All charts are rendered in matplotlib so there's no extra dependency — the
interactive Plotly versions in the app stay untouched. Each report is a
single 14x10 landscape page with:
- Purdue logo + gold accent bar in the black header
- Centered boxed horizontal legend of pitch types (pitcher report)
- Three bordered chart cards spanning the page width
- Centered title + centered summary table at the bottom

Public API
----------
- build_pitcher_pdf(...) -> bytes
- build_batter_pdf(...)  -> bytes
"""

from __future__ import annotations

import io
import os
from datetime import date as _date_type

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle, Polygon, FancyBboxPatch
import numpy as np
import pandas as pd

from constants import (
    PITCH_TYPE_COL,
    PITCH_TYPE_FIXED_COLORS,
    PITCH_TYPE_FALLBACK_COLORS,
    ZONE_LEFT,
    ZONE_RIGHT,
    ZONE_BOTTOM,
    ZONE_TOP,
    PURDUE_LOGO_FILENAME,
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
PURDUE_GOLD = "#DDB945"
PURDUE_BLACK = "#000000"
CARD_BORDER = "#c8d0da"
CARD_BG = "#ffffff"
TEXT_PRIMARY = "#1e293b"
TEXT_MUTED = "#64748b"

# Landscape — roomier than letter, still scalable for print (can be resized
# to 11x8.5 by most PDF viewers without clipping).
LETTER_LANDSCAPE = (14.0, 10.0)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pitch_colors(pitch_types) -> dict:
    """Map pitch type -> color using the same palette as the dashboard."""
    out = {}
    fallback_idx = 0
    for pt in pitch_types:
        if pd.isna(pt):
            continue
        if pt in PITCH_TYPE_FIXED_COLORS:
            out[pt] = PITCH_TYPE_FIXED_COLORS[pt]
        else:
            out[pt] = PITCH_TYPE_FALLBACK_COLORS[fallback_idx % len(PITCH_TYPE_FALLBACK_COLORS)]
            fallback_idx += 1
    return out


def _home_plate_patch(y_front: float = 0.0) -> Polygon:
    half_width = (17 / 12) / 2
    diag = 8.5 / 12
    back = 12 / 12
    pts = np.array([
        [-half_width, y_front],
        [half_width,  y_front],
        [half_width - diag, y_front - diag],
        [0, y_front - back],
        [-(half_width - diag), y_front - diag],
    ])
    return Polygon(pts, closed=True, fill=False, linewidth=1.4)


def _is_valid_pitch_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    return (
        series.notna()
        & s.ne("")
        & s.ne("undefined")
        & s.ne("other")
        & s.ne("nan")
    )


def _fmt_pct(v, digits: int = 1) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{v * 100:.{digits}f}%"


def _fmt_ba(v) -> str:
    if v is None or pd.isna(v):
        return "—"
    if v >= 1.0:
        return f"{v:.3f}"
    return f".{int(round(v * 1000)):03d}"


def _fmt_num(v, digits: int = 1) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{v:.{digits}f}"


# ---------------------------------------------------------------------------
# Header (black band + gold accent + logo + title + subtitle)
# ---------------------------------------------------------------------------
def _draw_header(fig, title: str, subtitle: str):
    # Black header spans from the very top of the page (y=1.0) down to the
    # top of the gold accent bar (y=0.935). No white strip above.
    header_top = 1.0
    header_bot = 0.935
    header_h   = header_top - header_bot  # 0.065
    header_ax = fig.add_axes([0.0, header_bot, 1.0, header_h])
    header_ax.set_facecolor(PURDUE_BLACK)
    header_ax.set_xlim(0, 1)
    header_ax.set_ylim(0, 1)
    header_ax.set_xticks([])
    header_ax.set_yticks([])
    for spine in header_ax.spines.values():
        spine.set_visible(False)

    header_ax.text(
        0.5, 0.64, title,
        ha="center", va="center",
        fontsize=20, fontweight="bold", color=PURDUE_GOLD,
        transform=header_ax.transAxes,
    )
    header_ax.text(
        0.5, 0.22, subtitle,
        ha="center", va="center",
        fontsize=11, color="#ffffff",
        transform=header_ax.transAxes,
    )

    # Purdue logo on the left, vertically centered in the header.
    logo_path = os.path.join(STATIC_DIR, PURDUE_LOGO_FILENAME)
    if os.path.isfile(logo_path):
        try:
            logo = plt.imread(logo_path)
            logo_h = 0.045
            logo_y = header_bot + (header_h - logo_h) / 2
            logo_ax = fig.add_axes([0.012, logo_y, 0.055, logo_h])
            logo_ax.imshow(logo)
            logo_ax.axis("off")
        except Exception:
            pass

    # Gold accent bar sits directly below the black header (no gap).
    accent_ax = fig.add_axes([0.0, 0.927, 1.0, 0.008])
    accent_ax.set_facecolor(PURDUE_GOLD)
    accent_ax.set_xticks([])
    accent_ax.set_yticks([])
    for spine in accent_ax.spines.values():
        spine.set_visible(False)


# ---------------------------------------------------------------------------
# Boxed horizontal legend, centered under the subtitle
# ---------------------------------------------------------------------------
def _draw_pitch_legend(fig, pitch_types: list, colors: dict,
                       y: float = 0.875, height: float = 0.035,
                       center_x: float = 0.5, width: float = 0.55):
    """Draw a centered, bordered legend of pitch types."""
    if not pitch_types:
        return
    left = center_x - width / 2
    ax = fig.add_axes([left, y, width, height])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#ffffff")
    for spine in ax.spines.values():
        spine.set_edgecolor(CARD_BORDER)
        spine.set_linewidth(1.0)

    n = len(pitch_types)
    slot = 1.0 / n
    for i, pt in enumerate(pitch_types):
        cx = slot * (i + 0.5)
        dot_x = cx - 0.04
        # Colored dot
        ax.scatter([dot_x], [0.5], s=90,
                   color=colors.get(pt, "#888"),
                   edgecolors="white", linewidths=0.6,
                   transform=ax.transAxes, clip_on=False)
        # Label
        ax.text(dot_x + 0.015, 0.5, pt,
                ha="left", va="center", fontsize=10, color=TEXT_PRIMARY,
                transform=ax.transAxes)


# ---------------------------------------------------------------------------
# Chart "card" — bordered rectangle around an inner axes
# ---------------------------------------------------------------------------
def _draw_card(fig, left: float, bottom: float, width: float, height: float,
               title: str | None = None, *, polar: bool = False):
    """
    Draw a rounded-corner card and return an inner axes positioned inside it.

    Uses a bg axes (sized to the whole card) as a mounting surface for a
    FancyBboxPatch with rounded corners. The chart axes is placed at a
    higher zorder so it renders on top of the card background.

    All coordinates are figure fraction (0..1).
    """
    # Invisible mounting axes at the full card size
    bg_ax = fig.add_axes([left, bottom, width, height], zorder=0)
    bg_ax.set_xlim(0, 1)
    bg_ax.set_ylim(0, 1)
    bg_ax.set_xticks([])
    bg_ax.set_yticks([])
    bg_ax.set_facecolor("none")
    bg_ax.tick_params(length=0)
    for spine in bg_ax.spines.values():
        spine.set_visible(False)

    # Rounded card: white fill + light gray border
    bg_ax.add_patch(FancyBboxPatch(
        (0.008, 0.008), 0.984, 0.984,
        boxstyle="round,pad=0,rounding_size=0.035",
        facecolor=CARD_BG,
        edgecolor=CARD_BORDER,
        linewidth=1.2,
        transform=bg_ax.transAxes,
        clip_on=False,
    ))

    # Title strip inside the top of the card
    if title:
        fig.text(
            left + width / 2, bottom + height - 0.020,
            title,
            ha="center", va="top",
            fontsize=11, fontweight="bold", color=TEXT_PRIMARY,
        )
        top_pad = 0.045
    else:
        top_pad = 0.015

    # Inner chart axes — taller bottom gap so x-axis labels fit inside the card
    pad_x = 0.020
    bottom_pad = 0.060   # room for x-axis tick labels + axis title
    ax_left = left + pad_x
    ax_width = width - 2 * pad_x
    ax_bottom = bottom + bottom_pad
    ax_height = height - top_pad - bottom_pad

    if polar:
        ax = fig.add_axes(
            [ax_left, ax_bottom, ax_width, ax_height],
            projection="polar", zorder=5,
        )
    else:
        ax = fig.add_axes(
            [ax_left, ax_bottom, ax_width, ax_height], zorder=5,
        )
    ax.set_facecolor("white")
    return ax


# ---------------------------------------------------------------------------
# Pitcher charts
# ---------------------------------------------------------------------------
def _draw_pitch_usage_pie(ax, usage_df: pd.DataFrame, colors: dict):
    ax.set_facecolor("white")

    if usage_df is None or usage_df.empty:
        ax.text(0.5, 0.5, "No usage data", ha="center", va="center",
                transform=ax.transAxes, color=TEXT_MUTED)
        ax.set_axis_off()
        return

    labels = usage_df[PITCH_TYPE_COL].tolist()
    pcts = usage_df["usage_pct"].values
    wedge_colors = [colors.get(pt, "#888") for pt in labels]

    ax.pie(
        pcts,
        colors=wedge_colors,
        startangle=90, counterclock=False,
        autopct=lambda p: f"{p:.1f}%" if p >= 3 else "",
        pctdistance=0.66,
        textprops={"fontsize": 9, "fontweight": "bold"},
        wedgeprops={"edgecolor": "white", "linewidth": 1.4},
    )
    ax.set_axis_off()


def _draw_pitch_locations(ax, df: pd.DataFrame, colors: dict):
    ax.set_facecolor("white")

    if df is None or df.empty:
        ax.text(0.5, 0.5, "No location data", ha="center", va="center",
                transform=ax.transAxes, color=TEXT_MUTED)
        ax.set_axis_off()
        return

    d = df.copy()
    d["PlateLocSide"] = pd.to_numeric(d["PlateLocSide"], errors="coerce")
    d["PlateLocHeight"] = pd.to_numeric(d["PlateLocHeight"], errors="coerce")
    d = d.dropna(subset=["PlateLocSide", "PlateLocHeight"])
    d = d[_is_valid_pitch_series(d[PITCH_TYPE_COL])]

    # Shadow zone — a subtle gray rectangle slightly larger than the strike zone
    shadow_pad = 0.2
    ax.add_patch(Rectangle(
        (ZONE_LEFT - shadow_pad, ZONE_BOTTOM - shadow_pad),
        (ZONE_RIGHT - ZONE_LEFT) + 2 * shadow_pad,
        (ZONE_TOP - ZONE_BOTTOM) + 2 * shadow_pad,
        facecolor="#e5e7eb", edgecolor="none", alpha=0.55, zorder=1,
    ))
    # Strike zone — solid black border
    ax.add_patch(Rectangle(
        (ZONE_LEFT, ZONE_BOTTOM),
        ZONE_RIGHT - ZONE_LEFT, ZONE_TOP - ZONE_BOTTOM,
        fill=False, linewidth=2.2, edgecolor="#1e293b", zorder=3,
    ))
    # Mid-lines across the zone
    mid_y = (ZONE_BOTTOM + ZONE_TOP) / 2
    ax.plot([ZONE_LEFT, ZONE_RIGHT], [mid_y, mid_y],
            linestyle="--", linewidth=1.0, color="#60a5fa",
            alpha=0.7, zorder=2)
    ax.plot([0, 0], [ZONE_BOTTOM, ZONE_TOP],
            linestyle="--", linewidth=1.0, color="#f97316",
            alpha=0.7, zorder=2)
    # Home plate at the bottom (point down, catcher's view)
    ax.add_patch(_home_plate_patch(y_front=0.55))

    for pt, g in d.groupby(PITCH_TYPE_COL):
        ax.scatter(
            g["PlateLocSide"], g["PlateLocHeight"],
            color=colors.get(pt, "#888"),
            s=30, alpha=0.85, edgecolors="white", linewidths=0.5,
            label=pt, zorder=4,
        )

    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.8, 5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("PlateLocSide", fontsize=9)
    ax.set_ylabel("PlateLocHeight", fontsize=9)
    ax.tick_params(labelsize=7.5, length=3, pad=3)
    ax.grid(True, which="major", linestyle="-", linewidth=0.4,
            color="#e2e8f0", zorder=0)
    for spine in ax.spines.values():
        spine.set_edgecolor("#cbd5e1")
        spine.set_linewidth(0.9)


def _draw_pitch_movements(ax, df: pd.DataFrame, colors: dict):
    ax.set_facecolor("white")

    if df is None or df.empty:
        ax.text(0.5, 0.5, "No movement data", ha="center", va="center",
                transform=ax.transAxes, color=TEXT_MUTED)
        ax.set_axis_off()
        return

    d = df.copy()
    d["HorzBreak"] = pd.to_numeric(d["HorzBreak"], errors="coerce")
    d["InducedVertBreak"] = pd.to_numeric(d["InducedVertBreak"], errors="coerce")
    d = d.dropna(subset=["HorzBreak", "InducedVertBreak"])
    d = d[_is_valid_pitch_series(d[PITCH_TYPE_COL])]

    ax.axhline(0, color="#4f83b6", linewidth=0.9, alpha=0.7)
    ax.axvline(0, color="#4f83b6", linewidth=0.9, alpha=0.7)

    for pt, g in d.groupby(PITCH_TYPE_COL):
        ax.scatter(
            g["HorzBreak"], g["InducedVertBreak"],
            color=colors.get(pt, "#888"),
            s=26, alpha=0.82, edgecolors="white", linewidths=0.4,
            label=pt,
        )

    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Horizontal break (in)", fontsize=9)
    ax.set_ylabel("Induced vertical break (in)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.15)


# ---------------------------------------------------------------------------
# Centered table (title in the middle, no gap)
# ---------------------------------------------------------------------------
def _draw_centered_table(fig, df: pd.DataFrame, title: str, *,
                         top: float, bottom: float, width: float = 0.80):
    """Draw a bordered, centered table with the title immediately above it."""
    if df is None or df.empty:
        return
    left = (1.0 - width) / 2

    # Title centered above the table
    title_y = top
    fig.text(
        left + width / 2, title_y,
        title,
        ha="center", va="top",
        fontsize=14, fontweight="bold", color=TEXT_PRIMARY,
    )

    # Table area — table sits at the TOP of its axes, hugging the title.
    table_top = title_y - 0.028
    table_bottom = bottom
    ax = fig.add_axes([left, table_bottom, width, table_top - table_bottom])
    ax.set_facecolor("white")
    ax.axis("off")

    cell_text = df.astype(str).values.tolist()
    col_labels = list(df.columns)

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="upper center",   # hug the top edge — no wasted vertical gap
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.50)

    # Header rows are taller so two-line labels ("Zone\nContact %") fit.
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(CARD_BORDER)
        if row == 0:
            cell.set_facecolor(PURDUE_BLACK)
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_height(cell.get_height() * 1.6)  # fit 2-line labels
        else:
            cell.set_facecolor("#ffffff" if row % 2 == 0 else "#f7f7f7")
            cell.set_text_props(color="#222")

    # First column left-aligned
    for row in range(len(cell_text) + 1):
        tbl[(row, 0)].set_text_props(ha="left")
        tbl[(row, 0)].PAD = 0.05


# ---------------------------------------------------------------------------
# Pitcher report
# ---------------------------------------------------------------------------
def _build_pitcher_figure(
    pitcher_df: pd.DataFrame,
    usage_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    name: str,
    hand: str,
    side_label: str,
    pitch_count: int,
    data_through: _date_type | None,
    opponent: str | None = None,
    game_date: _date_type | None = None,
):
    """Build the pitcher one-pager figure and return it (caller saves/closes)."""
    # Pitch types the pitcher actually throws (ordered by usage for the legend)
    if usage_df is not None and not usage_df.empty:
        thrown_types = usage_df[PITCH_TYPE_COL].tolist()
    elif pitcher_df is not None and PITCH_TYPE_COL in pitcher_df.columns:
        thrown_types = [
            pt for pt in pitcher_df[PITCH_TYPE_COL].dropna().unique()
            if str(pt).strip().lower() not in ("", "undefined", "other", "nan")
        ]
    else:
        thrown_types = []

    colors = _pitch_colors(thrown_types)

    fig = plt.figure(figsize=LETTER_LANDSCAPE, facecolor="white")

    # --- Header --------------------------------------------------------
    subtitle_bits = [name]
    if hand:
        subtitle_bits.append(hand)
    if side_label:
        subtitle_bits.append(side_label)
    if game_date is not None:
        subtitle_bits.append(game_date.strftime("%b %d, %Y"))
    subtitle_bits.append(f"{pitch_count} pitches")
    if opponent:
        subtitle_bits.append(f"vs {opponent}")
    if data_through is not None:
        subtitle_bits.append(
            f"through {data_through.strftime('%b %d, %Y')}"
        )
    _draw_header(fig, "Pitcher Report", "  |  ".join(subtitle_bits))

    # --- Centered boxed legend ----------------------------------------
    # Width auto-scales with number of pitch types (roomier for 5, tighter for 2)
    legend_width = min(0.08 + 0.11 * len(thrown_types), 0.70)
    _draw_pitch_legend(
        fig, thrown_types, colors,
        y=0.870, height=0.035, width=legend_width,
    )

    # --- Three bordered chart cards -----------------------------------
    card_y = 0.42
    card_h = 0.41
    card_gap = 0.012
    card_w = (1.0 - 0.04 - 2 * card_gap) / 3
    lefts = [0.02, 0.02 + card_w + card_gap, 0.02 + 2 * (card_w + card_gap)]

    pie_ax = _draw_card(fig, lefts[0], card_y, card_w, card_h, title="Pitch Usage")
    loc_ax = _draw_card(fig, lefts[1], card_y, card_w, card_h, title="Pitch Locations")
    mov_ax = _draw_card(fig, lefts[2], card_y, card_w, card_h, title="Pitch Movements")

    _draw_pitch_usage_pie(pie_ax, usage_df, colors)
    _draw_pitch_locations(loc_ax, pitcher_df, colors)
    _draw_pitch_movements(mov_ax, pitcher_df, colors)

    # --- Centered summary table (sits right below the cards) ----------
    _draw_centered_table(
        fig, summary_df, "Summary Table",
        top=0.38, bottom=0.03, width=0.80,
    )

    return fig


def build_pitcher_pdf(
    pitcher_df: pd.DataFrame,
    usage_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    name: str,
    hand: str,
    side_label: str,
    pitch_count: int,
    data_through: _date_type | None,
    opponent: str | None = None,
    game_date: _date_type | None = None,
) -> bytes:
    """Build the pitcher one-pager and return PDF bytes (single page)."""
    fig = _build_pitcher_figure(
        pitcher_df, usage_df, summary_df,
        name=name, hand=hand, side_label=side_label,
        pitch_count=pitch_count, data_through=data_through,
        opponent=opponent, game_date=game_date,
    )
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        pdf.savefig(fig, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def build_multi_pitcher_pdf(entries: list[dict]) -> bytes:
    """Build a multi-page pitcher PDF — one page per dict in ``entries``.

    Each entry is a keyword-argument mapping suitable for
    :func:`_build_pitcher_figure` (``pitcher_df``, ``usage_df``,
    ``summary_df``, ``name``, ``hand``, ``side_label``, ``pitch_count``,
    ``data_through``, ``opponent``, ``game_date``).
    """
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        for entry in entries:
            fig = _build_pitcher_figure(**entry)
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Batter charts
# ---------------------------------------------------------------------------
def _draw_out_rate_heatmap(ax, df: pd.DataFrame, *, fig=None, card_box=None):
    """Draw the zone heatmap.

    When ``fig`` and ``card_box`` (left, bottom, width, height — figure
    fraction) are provided, the axes is resized to fill the card above a
    reserved legend strip, and the color-code legend is rendered at the
    card's bottom-left via ``fig.text``. Otherwise the legend is drawn
    inside the axes for backward compatibility.
    """
    ax.set_facecolor("white")

    use_fig_legend = fig is not None and card_box is not None
    if use_fig_legend:
        card_l, card_b, card_w, card_h = card_box
        # Reserve bottom strip for the color-code legend; fill the rest.
        legend_strip = 0.18 * card_h
        # Absolute figure-fraction reserve so the card title (drawn at
        # card_b + card_h - 0.020 with va="top") isn't covered by the ax.
        top_strip = 0.045
        side_pad = 0.015 * card_w
        ax.set_position([
            card_l + side_pad,
            card_b + legend_strip,
            card_w - 2 * side_pad,
            card_h - top_strip - legend_strip,
        ])

    # Soft green shadow zone surrounding the strike zone (matches app)
    shadow_pad = 0.30
    ax.add_patch(Rectangle(
        (ZONE_LEFT - shadow_pad, ZONE_BOTTOM - shadow_pad),
        (ZONE_RIGHT - ZONE_LEFT) + 2 * shadow_pad,
        (ZONE_TOP - ZONE_BOTTOM) + 2 * shadow_pad,
        facecolor="#edf4ee", edgecolor="none", zorder=0,
    ))

    zw = (ZONE_RIGHT - ZONE_LEFT) / 3
    zh = (ZONE_TOP - ZONE_BOTTOM) / 3
    col_edges = [ZONE_LEFT + i * zw for i in range(4)]
    row_edges = [ZONE_BOTTOM + i * zh for i in range(4)]

    OUT_RESULTS = {"Out", "FieldersChoice", "Strikeout", "Sacrifice",
                   "SacrificeFly", "SacrificeBunt", "Error"}
    PA_RESULTS = {"Single", "Double", "Triple", "HomeRun", "Out", "FieldersChoice",
                  "Strikeout", "Walk", "HitByPitch", "SacrificeFly", "SacrificeBunt",
                  "Sacrifice", "Error", "CatcherInterference"}

    total_pa = 0
    total_outs = 0

    empty = df is None or df.empty
    if not empty:
        d = df.copy()
        d["PlateLocSide"] = pd.to_numeric(d["PlateLocSide"], errors="coerce")
        d["PlateLocHeight"] = pd.to_numeric(d["PlateLocHeight"], errors="coerce")
        d = d.dropna(subset=["PlateLocSide", "PlateLocHeight"])
        # Flip to catcher's view so the chart matches the app (scouting standard).
        d["PlateLocSide"] = -d["PlateLocSide"]
        pc = d["PitchCall"].astype(str).str.strip() if "PitchCall" in d.columns else pd.Series("", index=d.index)
        pr = d["PlayResult"].astype(str).str.strip() if "PlayResult" in d.columns else pd.Series("", index=d.index)
        is_terminal = pr.isin(PA_RESULTS) | pc.eq("HitByPitch")

        for ri in range(3):
            for ci in range(3):
                x0, x1 = col_edges[ci], col_edges[ci + 1]
                y0, y1 = row_edges[ri], row_edges[ri + 1]
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                in_cell = d["PlateLocSide"].between(x0, x1) & d["PlateLocHeight"].between(y0, y1)
                term_in = in_cell & is_terminal
                n_pa = int(term_in.sum())
                n_outs = int((term_in & pr.isin(OUT_RESULTS)).sum())
                total_pa += n_pa
                total_outs += n_outs

                if n_pa == 0:
                    bg = "#f5f5f5"
                else:
                    op = n_outs / n_pa
                    if op >= 0.75: bg = "#c0392b"
                    elif op >= 0.50: bg = "#f0b000"
                    elif op >= 0.25: bg = "#f5dfb3"
                    else: bg = "#ffffff"

                ax.add_patch(Rectangle((x0, y0), zw, zh, facecolor=bg,
                                        edgecolor="#999", linewidth=0.8))
                if n_pa == 0:
                    ax.text(cx, cy, "—", ha="center", va="center",
                            fontsize=12, color="#aaa")
                else:
                    op = n_outs / n_pa
                    tc = "#fff" if op >= 0.40 else "#222"
                    ax.text(cx, cy + zh * 0.18, f"{int(round(op * 100))}%",
                            ha="center", va="center", fontsize=14,
                            fontweight="bold", color=tc)
                    ax.text(cx, cy - zh * 0.22, f"{n_outs}/{n_pa}",
                            ha="center", va="center", fontsize=9, color=tc)
    else:
        for ri in range(3):
            for ci in range(3):
                x0 = col_edges[ci]; y0 = row_edges[ri]
                ax.add_patch(Rectangle((x0, y0), zw, zh, facecolor="#f5f5f5",
                                        edgecolor="#999", linewidth=0.8))

    ax.add_patch(Rectangle((ZONE_LEFT, ZONE_BOTTOM),
                            ZONE_RIGHT - ZONE_LEFT, ZONE_TOP - ZONE_BOTTOM,
                            fill=False, linewidth=2.4, edgecolor="#1e293b"))

    # Subtitle at the top of the zone (matches the app version)
    out_pct_total = f"{int(round(total_outs / total_pa * 100))}%" if total_pa > 0 else "—"
    ax.text(
        0, ZONE_TOP + 0.58,
        f"Out Rate: {out_pct_total}  |  In-Zone PA: {total_pa}",
        ha="center", va="bottom",
        fontsize=10, color="#555",
    )

    legend_entries = [
        ("Red",   "danger zone, gets out 75%+ of the time", "#c0392b"),
        ("Gold",  "gets out more often than not (50 - 75%)", "#d4a017"),
        ("Light", "moderate (25 - 50%)",                     "#b8860b"),
        ("White", "productive zone (< 25%)",                 "#888888"),
    ]
    if use_fig_legend:
        # Legend block anchored to the card's bottom-left, outside the ax.
        card_l, card_b, card_w, card_h = card_box
        line_gap = 0.025 * card_h
        top_y = card_b + 0.105 * card_h
        label_x = card_l + 0.035 * card_w
        desc_x = card_l + 0.105 * card_w
        for i, (label, desc, color) in enumerate(legend_entries):
            y = top_y - i * line_gap
            fig.text(label_x, y, label, ha="left", va="top",
                     fontsize=8, fontweight="bold", color=color)
            fig.text(desc_x, y, f": {desc}", ha="left", va="top",
                     fontsize=8, color="#555")
    else:
        # Fallback: legend inside the axes (preserves older callers).
        legend_y = ZONE_BOTTOM - 0.85
        for i, (label, desc, color) in enumerate(legend_entries):
            ax.text(
                ZONE_LEFT - 0.3, legend_y - i * 0.22,
                label, ha="left", va="center",
                fontsize=7.5, color=color, fontweight="bold",
            )
            ax.text(
                ZONE_LEFT + 0.05, legend_y - i * 0.22,
                f": {desc}", ha="left", va="center",
                fontsize=7.5, color="#555",
            )

    if use_fig_legend:
        # Tight margins — zone + shadow + subtitle, no legend below.
        ax.set_xlim(ZONE_LEFT - 0.35, ZONE_RIGHT + 0.35)
        ax.set_ylim(ZONE_BOTTOM - 0.35, ZONE_TOP + 0.70)
    else:
        ax.set_xlim(ZONE_LEFT - 0.4, ZONE_RIGHT + 0.4)
        ax.set_ylim(ZONE_BOTTOM - 1.85, ZONE_TOP + 0.75)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)


def _draw_pitch_results_scatter(ax, df: pd.DataFrame):
    ax.set_facecolor("white")

    ax.add_patch(Rectangle((ZONE_LEFT, ZONE_BOTTOM),
                            ZONE_RIGHT - ZONE_LEFT, ZONE_TOP - ZONE_BOTTOM,
                            fill=False, linewidth=1.8, edgecolor="#1e293b"))

    hp_x = [-0.71, 0.71, 0.71, 0, -0.71]
    hp_y = [0.95, 0.95, 1.15, 1.35, 1.15]
    ax.fill(hp_x + [hp_x[0]], hp_y + [hp_y[0]],
            facecolor="white", edgecolor="#555", linewidth=0.9)

    if df is not None and not df.empty:
        d = df.copy()
        d["PlateLocSide"] = pd.to_numeric(d["PlateLocSide"], errors="coerce")
        d["PlateLocHeight"] = pd.to_numeric(d["PlateLocHeight"], errors="coerce")
        d = d.dropna(subset=["PlateLocSide", "PlateLocHeight"])
        pc = d["PitchCall"].astype(str).str.strip() if "PitchCall" in d.columns else pd.Series("", index=d.index)
        pr = d["PlayResult"].astype(str).str.strip() if "PlayResult" in d.columns else pd.Series("", index=d.index)

        HIT = {"Single", "Double", "Triple", "HomeRun"}
        OUT = {"Out", "FieldersChoice", "Error", "Sacrifice",
               "SacrificeFly", "SacrificeBunt"}

        traces = [
            ("Whiff",         pc.eq("StrikeSwinging"), "x", "#E24B4A", 40),
            ("Foul",          pc.isin({"FoulBallFieldable", "FoulBallNotFieldable"}), "^", "#7B68AE", 32),
            ("Called Strike", pc.eq("StrikeCalled"), "s", "#E67E22", 28),
            ("Ball",          pc.isin({"BallCalled", "Ball"}), "o", "#aaaaaa", 22),
        ]
        for label, mask, marker, color, size in traces:
            if mask.any():
                g = d[mask]
                ax.scatter(g["PlateLocSide"], g["PlateLocHeight"],
                           marker=marker, color=color, s=size,
                           alpha=0.85, edgecolors="white", linewidths=0.35,
                           label=label)

        in_play = d[pc.eq("InPlay")]
        if not in_play.empty:
            ip_pr = in_play["PlayResult"].astype(str).str.strip()
            hits = in_play[ip_pr.isin(HIT)]
            outs = in_play[ip_pr.isin(OUT)]
            if not hits.empty:
                ax.scatter(hits["PlateLocSide"], hits["PlateLocHeight"],
                           marker="D", color="#2E8B57", s=40,
                           alpha=0.9, edgecolors="white", linewidths=0.35,
                           label="In-Play Hit")
            if not outs.empty:
                ax.scatter(outs["PlateLocSide"], outs["PlateLocHeight"],
                           marker="D", color="#8B0000", s=40,
                           alpha=0.9, edgecolors="white", linewidths=0.35,
                           label="In-Play Out")

        ax.legend(loc="upper right", fontsize=6.5, frameon=True,
                  framealpha=0.92, edgecolor="#ccc")

    ax.set_xlim(ZONE_LEFT - 1.0, ZONE_RIGHT + 1.0)
    ax.set_ylim(ZONE_BOTTOM - 0.6, ZONE_TOP + 0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)


def _draw_spray_chart(ax, df: pd.DataFrame):
    ax.set_facecolor("white")

    angles = np.linspace(np.radians(45), np.radians(135), 100)
    OF, IF = 200, 90
    ox, oy = OF * np.cos(angles), OF * np.sin(angles)
    ix, iy = IF * np.cos(angles), IF * np.sin(angles)
    ax.fill(np.append(ox, [0]), np.append(oy, [0]), color="#e8f5e9", zorder=1)
    ax.plot(np.append(ox, [0, ox[0]]), np.append(oy, [0, oy[0]]),
            color="#aaa", lw=0.9, zorder=2)
    ax.fill(np.append(ix, [0]), np.append(iy, [0]), color="#c8e6c9", zorder=1)
    ax.plot(np.append(ix, [0, ix[0]]), np.append(iy, [0, iy[0]]),
            color="#aaa", lw=0.7, zorder=2)
    for a in [45, 135]:
        r = np.radians(a)
        ax.plot([0, OF * np.cos(r)], [0, OF * np.sin(r)],
                color="#888", lw=0.9, ls="--", zorder=2)

    HIT = {"Single": "#1D9E75", "Double": "#378ADD",
           "Triple": "#7B2FBE", "HomeRun": "#BA7517"}

    if df is not None and not df.empty:
        d = df.copy()
        pr = d["PlayResult"].astype(str).str.strip() if "PlayResult" in d.columns else pd.Series("", index=d.index)
        for c in ["ExitSpeed", "Direction"]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        inp = d[pr.isin(set(HIT) | {"Out", "FieldersChoice", "Error"})].dropna(
            subset=["Direction"]
        ).copy()
        if not inp.empty:
            inp["_ang"] = np.radians(90 - inp["Direction"].astype(float))
            inp["_ev"] = inp["ExitSpeed"].fillna(70).astype(float)
            inp["_dist"] = inp["_ev"].apply(
                lambda ev: max(min(ev * 1.5, OF - 5), IF - 10)
            )
            inp["_pr"] = inp["PlayResult"].astype(str).str.strip()
            inp["_color"] = inp["_pr"].map(lambda r: HIT.get(r, "#D85A30"))
            inp["_size"] = inp["_pr"].apply(lambda r: 30 if r in HIT else 22)
            for color, grp in inp.groupby("_color", sort=False):
                ax.scatter(
                    grp["_dist"] * np.cos(grp["_ang"]),
                    grp["_dist"] * np.sin(grp["_ang"]),
                    s=grp["_size"].values,
                    color=color, alpha=0.82, edgecolors="none", zorder=4,
                )

    ax.plot(0, 0, "s", color="white", markersize=6, markeredgecolor="#555", zorder=5)
    ax.set_xlim(-220, 220); ax.set_ylim(-20, 230)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)


def _draw_ev_la(ax, df: pd.DataFrame):
    ax.set_facecolor("white")

    ax.add_patch(Rectangle((98, 26), 22, 4, color="#e8f5e9", zorder=1,
                            linewidth=0.9, edgecolor="#2e7d32", linestyle="--"))
    ax.text(109, 28, "Barrel", ha="center", va="center",
            fontsize=7, color="#2e7d32", fontweight="bold", zorder=2)

    HT = {"GroundBall": "#D85A30", "LineDrive": "#1D9E75",
          "FlyBall": "#378ADD", "Popup": "#888780"}

    if df is not None and not df.empty:
        d = df.copy()
        for c in ["ExitSpeed", "Angle"]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna(subset=["ExitSpeed", "Angle"])
        d = d[d["ExitSpeed"] > 0]
        if not d.empty and "TaggedHitType" in d.columns:
            d["_color"] = d["TaggedHitType"].astype(str).str.strip().map(
                lambda h: HT.get(h, "#aaa")
            )
            for color, grp in d.groupby("_color", sort=False):
                ax.scatter(grp["ExitSpeed"], grp["Angle"],
                           s=26, color=color, alpha=0.82,
                           edgecolors="white", linewidths=0.35, zorder=3)

    ax.axhline(0, color="#ccc", lw=0.7, ls="--")
    ax.set_xlim(20, 130)
    ax.set_ylim(-40, 90)
    ax.set_xlabel("Exit velocity (mph)", fontsize=8)
    ax.set_ylabel("Launch angle (°)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.15, linestyle="--")


def _draw_plate_discipline_radar(ax, df: pd.DataFrame):
    ax.set_facecolor("white")
    # Keep the polar plot square and give extra breathing room so the six
    # outer spoke labels ("Contact%", "Zone Sw%", etc.) never get clipped.
    pos = ax.get_position()
    side = min(pos.width, pos.height) * 0.88  # shrink just enough for spoke-label halo
    ax.set_position([
        pos.x0 + (pos.width - side) / 2,
        pos.y0 + (pos.height - side) / 2,
        side, side,
    ])

    LABELS = ["Zone\nSw%", "Contact%", "O-Contact%", "Chase%", "Whiff%", "Zone%"]
    N = len(LABELS)
    angles = [n / N * 2 * np.pi for n in range(N)] + [0]
    values = [0.0] * N

    if df is not None and not df.empty:
        d = df.copy()
        pc = d["PitchCall"].astype(str).str.strip() if "PitchCall" in d.columns else pd.Series("", index=d.index)
        SW = {"StrikeSwinging", "FoulBallFieldable", "FoulBallNotFieldable", "InPlay"}
        CT = {"FoulBallFieldable", "FoulBallNotFieldable", "InPlay"}

        if "PlateLocSide" in d.columns and "PlateLocHeight" in d.columns:
            ls = pd.to_numeric(d["PlateLocSide"], errors="coerce")
            lh = pd.to_numeric(d["PlateLocHeight"], errors="coerce")
            iz = lh.between(ZONE_BOTTOM, ZONE_TOP) & ls.between(ZONE_LEFT, ZONE_RIGHT)
        else:
            iz = pd.Series(False, index=d.index)
        oz = ~iz

        sw = pc.isin(SW)
        ct = pc.isin(CT)
        wh = pc.eq("StrikeSwinging")

        n = len(d)
        zp = int(iz.sum())
        op = int(oz.sum())
        ts = int(sw.sum())
        oz_sw = int((sw & oz).sum())

        values = [
            int((sw & iz).sum()) / zp if zp > 0 else 0,
            int(ct.sum()) / ts if ts > 0 else 0,
            int((ct & oz).sum()) / oz_sw if oz_sw > 0 else 0,
            oz_sw / op if op > 0 else 0,
            int(wh.sum()) / ts if ts > 0 else 0,
            zp / n if n > 0 else 0,
        ]

    vals = values + [values[0]]
    # Hexagonal grid rings (matches the app look) + solid spoke lines
    for r in [0.25, 0.50, 0.75, 1.0]:
        ax.plot(angles, [r] * (N + 1), color="#bfbfbf",
                lw=0.7, ls="--" if r < 1.0 else "-")
    for ang in angles[:-1]:
        ax.plot([ang, ang], [0, 1.0], color="#bfbfbf", lw=0.6)

    ax.plot(angles, vals, color="#185FA5", lw=2.0)
    ax.fill(angles, vals, color="#378ADD", alpha=0.18)
    ax.scatter(angles[:-1], values, s=26, color="#185FA5",
               edgecolors="white", lw=0.8, zorder=5)

    # Place value labels inside the hexagon so they never collide with the
    # axis spoke labels that sit just outside r=1.0.
    for ang, val in zip(angles[:-1], values):
        r = min(val + 0.11, 0.86)
        ax.text(ang, r, f"{val * 100:.0f}%",
                ha="center", va="center", fontsize=10, fontweight="bold",
                color="#185FA5", zorder=6)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(LABELS, fontsize=10)
    ax.tick_params(axis="x", pad=12)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(["25%", "50%", "75%"], fontsize=7, color="#aaa")


# ---------------------------------------------------------------------------
# Batter report
# ---------------------------------------------------------------------------
def _build_batter_figure(
    batter_df: pd.DataFrame,
    batting_line_row: dict,
    pitch_breakdown_df: pd.DataFrame,
    *,
    name: str,
    side: str,
    hand_label: str,
    pa: int,
    data_through: _date_type | None,
    opponent: str | None = None,
    game_date: _date_type | None = None,
):
    """Build the batter one-pager figure and return it (caller saves/closes)."""
    fig = plt.figure(figsize=LETTER_LANDSCAPE, facecolor="white")

    bits = [name]
    if side:
        bits.append(f"Bats: {side}")
    if hand_label:
        bits.append(hand_label)
    if game_date is not None:
        bits.append(game_date.strftime("%b %d, %Y"))
    bits.append(f"{pa} PA")
    if opponent:
        bits.append(f"vs {opponent}")
    if data_through is not None:
        bits.append(f"through {data_through.strftime('%b %d, %Y')}")
    _draw_header(fig, "Batter Report", "  |  ".join(bits))

    # --- Centered boxed batting summary table -------------------------
    batting_df = pd.DataFrame([batting_line_row])
    _draw_centered_table(
        fig, batting_df, "Batting Summary",
        top=0.895, bottom=0.80, width=0.70,
    )

    # --- By Pitch Type table (centered) — wider to fit all columns ----
    _draw_centered_table(
        fig, pitch_breakdown_df, "By Pitch Type",
        top=0.76, bottom=0.48, width=0.92,
    )

    # --- 2 chart cards — span the same width as the By Pitch Type table --
    table_width = 0.92
    card_gap = 0.025
    card_w = (table_width - card_gap) / 2  # ≈ 0.4475
    left0 = (1.0 - table_width) / 2
    lefts = [left0, left0 + card_w + card_gap]
    card_y = 0.02
    card_h = 0.45

    out_ax = _draw_card(fig, lefts[0], card_y, card_w, card_h, title="Out Rate by Zone")
    rad_ax = _draw_card(fig, lefts[1], card_y, card_w, card_h, title="Plate Discipline", polar=True)

    _draw_out_rate_heatmap(
        out_ax, batter_df,
        fig=fig, card_box=(lefts[0], card_y, card_w, card_h),
    )
    _draw_plate_discipline_radar(rad_ax, batter_df)

    return fig


def build_batter_pdf(
    batter_df: pd.DataFrame,
    batting_line_row: dict,
    pitch_breakdown_df: pd.DataFrame,
    *,
    name: str,
    side: str,
    hand_label: str,
    pa: int,
    data_through: _date_type | None,
    opponent: str | None = None,
    game_date: _date_type | None = None,
) -> bytes:
    """Build the batter one-pager and return PDF bytes (single page)."""
    fig = _build_batter_figure(
        batter_df, batting_line_row, pitch_breakdown_df,
        name=name, side=side, hand_label=hand_label, pa=pa,
        data_through=data_through, opponent=opponent, game_date=game_date,
    )
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        pdf.savefig(fig, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def build_multi_batter_pdf(entries: list[dict]) -> bytes:
    """Build a multi-page batter PDF — one page per dict in ``entries``."""
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        for entry in entries:
            fig = _build_batter_figure(**entry)
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
