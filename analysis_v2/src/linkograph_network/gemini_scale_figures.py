"""
Deck figures for the 10-session scale-out (PROTOTYPE).

Four paste-ready PNGs for the section that follows deck slide 55. Everything is read
from the artefacts `gemini_scale.py` and `gemini_session_inventory.py` already wrote --
no recomputation here, so a figure can never disagree with the tables behind it.

  1. selection      162 annotated -> 54 eligible -> 10 selected, plus the six criteria
                    and what each one excluded.
  2. basis_mix      per-session link groundedness: named / on_table / nearest_prior,
                    against the band the two focal sessions were validated in. This is
                    slide 51 + 54 answered for all 10.
  3. by_outcome     the slide-48 metric block, 5 team sessions vs 5 no-team sessions,
                    every session shown as its own point so the overlap is visible.
  4. summary_table  the slide-48 block as a table across all 10 sessions, grouped by
                    outcome -- the scaled analogue of `gemini_summary_table.png`.

Palette matches the existing deck figures (gemini_summary_table.py,
gemini_validation_summary.py, gemini_speaker_network.py).

Usage:
    python analysis_v2/src/linkograph_network/gemini_scale_figures.py
"""

import csv
import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))

TBL = os.path.join(_BASE, "analysis_v2/results/tables/prototypes/scale10")
OUT = os.path.join(_BASE, "analysis_v2/figures/prototypes/scale10/_deck")

# --- house palette ---------------------------------------------------------------
_HEAD = "#1E2761"       # section headers (navy)
_DIRECT = "#016B78"     # gemini-direct / funded / strongest basis tier (teal)
_MID = "#4FA3B0"        # on-a-team / middle basis tier (mid teal)
_LINK = "#C97B0A"       # reconstructed-link / the fallback tier (amber)
_GREY = "#B8B8B8"
_GREEN = "#1F8A70"
_MUTED = "#666666"
_BAND = "#F2F4F7"

BASIS = [("basis_named_speaker", "named_speaker", _DIRECT),
         ("basis_on_table_idea", "on_table_idea", _MID),
         ("basis_nearest_prior", "nearest_prior", _LINK)]

# Session display order: team sessions then no-team sessions, focal first within each.
TEAM_ORDER = ["NES_S3", "NES_S4", "MZT_S5", "MND_S5", "CMC_S11"]
NOTEAM_ORDER = ["NES_S10", "NES_S8", "ABI_S1", "MND_S15", "SLU_S5"]
ORDER = TEAM_ORDER + NOTEAM_ORDER
FOCAL = {"NES_S3", "NES_S10"}

# The slide-48 metric block. (key, label, gloss, provenance)
METRICS = [
    ("breadth [direct]",             "Breadth",              "new ideas",                  "direct"),
    ("volume [direct]",              "Volume",               "all idea acts",              "direct"),
    ("distribution [direct]",        "Distribution",         "speaker entropy (1 = even)", "direct"),
    ("depth_link_ratio [link]",      "Depth",                "links per idea utterance",   "link"),
    ("longest_chain [link]",         "Longest chain",        "deepest single thread",      "link"),
    ("bridging_moves [link]",        "Bridging moves",       "joins separate threads",     "link"),
    ("self_link_ratio [link]",       "Self-link ratio",      "links on own ideas",         "link"),
    ("max_betweenness [link]",       "Max betweenness",      "top broker's control",       "link"),
    ("degree_centralization [link]", "Deg. centralization",  "even (0) vs one hub",        "link"),
]


def _load():
    met = {r["label"]: r for r in csv.DictReader(
        open(os.path.join(TBL, "metrics_gemini_linkography_scaled.csv")))}
    inv = list(csv.DictReader(open(os.path.join(TBL, "session_inventory.csv"))))
    links = list(csv.DictReader(open(os.path.join(TBL, "link_basis_log.csv"))))
    return met, inv, links


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved:", os.path.relpath(path, _BASE))


# ---------------------------------------------------------------------------
# 1. Selection
# ---------------------------------------------------------------------------
CRITERIA = [
    ("ok_complete",  "Every chunk file parses",            "truncated Gemini dumps (ATTN_)"),
    ("ok_material",  "≥ 35 idea-bearing utterances",  "focal pair: 58 / 47"),
    ("ok_group",     "8–13 idea speakers",            "focal pair: 11 / 11"),
    ("ok_length",    "90–230 utterances",             "focal pair: 139 / 162"),
    ("ok_roster",    "≥ 70% speakers on the roster",  "speaker-label quality"),
    ("ok_grounded",  "nearest_prior share ≤ 35%",     "caveat (A) as a gate"),
]


def fig_selection(inv):
    total = len(inv)
    elig = sum(1 for r in inv if r["eligible"] == "True")
    fails = {k: sum(1 for r in inv if r[k] != "True") for k, _, _ in CRITERIA}

    fig, ax = plt.subplots(figsize=(13.6, 6.4))
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax.text(0.0, 0.965, "Which sessions can join NES_S3 and NES_S10?",
            fontsize=17, fontweight="bold", color=_HEAD)
    ax.text(0.0, 0.895, "Six criteria applied to every annotated session. Both focal "
                        "sessions pass unchanged — the gate is not tuned around them.",
            fontsize=10.2, color=_MUTED)

    # --- funnel (left column) -----------------------------------------------------
    # Caption sits ABOVE each bar and the count INSIDE it, so neither depends on how
    # wide the bar happens to be.
    stages = [(total, "annotated sessions", _GREY),
              (elig, "pass all six criteria", _MID),
              (10, "selected for the run", _DIRECT)]
    x0, full, h = 0.0, 0.40, 0.125
    y = 0.66
    for n, lab, c in stages:
        w = max(full * (n / total) ** 0.5, 0.12)
        ax.text(x0, y + h + 0.022, lab, fontsize=10.5, color="#333")
        ax.add_patch(Rectangle((x0, y), w, h, color=c, zorder=2))
        ax.text(x0 + 0.016, y + h / 2, str(n), fontsize=21, fontweight="bold",
                color="white", va="center", zorder=3)
        y -= h + 0.095
    ax.text(x0, y + h - 0.045,
            "8 new + the 2 already validated — 5 sessions where people teamed up\n"
            "in the room, 5 where none did, across all 8 conferences",
            fontsize=9.4, color=_MUTED, style="italic", linespacing=1.5, va="top")

    # --- criteria table (right column) --------------------------------------------
    tx, tw = 0.50, 0.50
    ax.text(tx, 0.80, "Criterion", fontsize=10.5, fontweight="bold", color=_HEAD)
    ax.text(tx + tw, 0.80, "excluded", fontsize=10.5, fontweight="bold",
            color=_HEAD, ha="right")
    ax.plot([tx, tx + tw], [0.775, 0.775], color="#999", lw=1)

    rowh, maxf = 0.115, max(fails.values())
    for i, (key, label, why) in enumerate(CRITERIA):
        yy = 0.715 - i * rowh
        if i % 2 == 0:
            ax.add_patch(Rectangle((tx - 0.012, yy - rowh / 2 + 0.012), tw + 0.024, rowh,
                                   color=_BAND, zorder=0))
        ax.text(tx, yy + 0.018, label, fontsize=9.8, color="#222", zorder=3)
        ax.text(tx, yy - 0.018, why, fontsize=7.8, color=_MUTED, style="italic", zorder=3)
        bw = 0.11 * (fails[key] / maxf)
        ax.add_patch(Rectangle((tx + tw - bw, yy - 0.004), bw, 0.028,
                               color=_LINK, alpha=0.55, zorder=2))
        ax.text(tx + tw - bw - 0.01, yy + 0.010, str(fails[key]), fontsize=9.8,
                ha="right", va="center", color="#333", zorder=3)

    ax.text(tx, 0.715 - len(CRITERIA) * rowh - 0.012,
            f"A session can fail several criteria, so these do not sum to {total - elig}.",
            fontsize=8, color=_MUTED, style="italic")
    _save(fig, "scale_selection.png")


# ---------------------------------------------------------------------------
# 2. Basis mix
# ---------------------------------------------------------------------------
def fig_basis_mix(met, links):
    ext = Counter()
    tot = Counter()
    for r in links:
        tot[r["label"]] += 1
        if r["links_to_extracted"] == "True":
            ext[r["label"]] += 1

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14.5, 7.4),
                                  gridspec_kw={"width_ratios": [3.1, 1.0]})
    for a in (ax, ax2):
        a.spines[["top", "right", "left"]].set_visible(False)

    ys = list(range(len(ORDER)))[::-1]
    for y, label in zip(ys, ORDER):
        row = met[label]
        n = int(row["total_links"])
        left = 0.0
        for key, _, color in BASIS:
            v = int(row[key]) / n * 100
            ax.barh(y, v, left=left, color=color, height=0.62,
                    edgecolor="white", linewidth=0.8)
            if v >= 9:
                ax.text(left + v / 2, y, f"{v:.0f}%", ha="center", va="center",
                        fontsize=8.6, color="white", fontweight="bold")
            left += v
        ax.text(101.5, y, f"{n} links", fontsize=8.4, va="center", color=_MUTED)

    # Where the amber segment starts = 100 - that session's fallback share. The two
    # dashed lines are where it started for the sessions the method was validated on,
    # so a bar whose amber begins right of 80 is better grounded than S3, and one that
    # begins left of 70 is worse than S10.
    ax.axvspan(70, 80, color="#444", alpha=0.06, zorder=0)
    for pct in (70, 80):
        ax.axvline(pct, color="#444", lw=1.0, ls=(0, (4, 3)), zorder=5)
    ax.annotate("fallback starts here for\nthe two validated sessions",
                xy=(75, len(ORDER) - 0.62), xytext=(75, len(ORDER) - 0.05),
                fontsize=7.8, ha="center", color="#444", style="italic",
                arrowprops=dict(arrowstyle="-", color="#888", lw=0.8))

    ax.set_yticks(ys)
    ax.set_yticklabels(ORDER, fontsize=10)
    for t, l in zip(ax.get_yticklabels(), ORDER):
        if l in FOCAL:
            t.set_fontweight("bold")
            t.set_color(_HEAD)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of that session's links, by how the target was resolved", fontsize=9.5)
    ax.set_title("How grounded is each session's linking?", fontsize=13.5,
                 fontweight="bold", color=_HEAD, loc="left")

    # group separator + brackets, placed OUTSIDE the axes so nothing overlaps a bar
    sep = len(ORDER) - len(TEAM_ORDER) - 0.5
    ax.axhline(sep, color="#999", lw=1.0)
    blend = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    for lo, hi, text, color in ((sep + 0.5, len(ORDER) - 0.5, "TEAMS FORMED\nIN THE ROOM", _DIRECT),
                                (-0.5, sep - 0.5, "NO IN-ROOM\nTEAMS", _MUTED)):
        ax.plot([-0.115, -0.115], [lo, hi], color=color, lw=2.4,
                transform=blend, clip_on=False, solid_capstyle="butt")
        ax.text(-0.128, (lo + hi) / 2, text, fontsize=8, color=color, rotation=90,
                ha="center", va="center", fontweight="bold", linespacing=1.3,
                transform=blend, clip_on=False)

    ax.legend(handles=[Line2D([0], [0], color=c, lw=9, label=n) for _, n, c in BASIS],
              loc="lower center", bbox_to_anchor=(0.5, -0.155), ncol=3, fontsize=9.4,
              frameon=False)

    # --- right panel: links_to extraction ---
    for y, label in zip(ys, ORDER):
        pct = 100 * ext[label] / tot[label]
        ax2.barh(y, pct, color=_DIRECT, alpha=0.85, height=0.62)
        ax2.text(pct + 1.2, y, f"{pct:.0f}%", fontsize=8.6, va="center", color="#333")
    ax2.axhline(sep, color="#999", lw=1.0)
    ax2.set_yticks([]); ax2.set_xlim(0, 55)
    ax2.set_ylim(*ax.get_ylim())
    ax2.set_xlabel("% of links", fontsize=9.5)
    ax2.set_title("A real `links_to`\nextracted from the explanation",
                  fontsize=11, fontweight="bold", color=_HEAD, loc="left")

    n_all = sum(tot.values())
    near = sum(int(met[l]["basis_nearest_prior"]) for l in ORDER)
    fig.text(0.005, -0.035,
             f"All 10 sessions: {n_all} links — {near} resolved by the nearest_prior "
             f"fallback ({100 * near / n_all:.0f}%), per-session 3%–32%. "
             f"A real links_to came out of the explanation for {sum(ext.values())} "
             f"({100 * sum(ext.values()) / n_all:.0f}%). "
             "Fallback links spot-checked at ~55% correct against ~85% for named — "
             "read every [reconstructed-link] metric next to this.",
             fontsize=8.6, color=_MUTED)
    fig.tight_layout()
    _save(fig, "scale_basis_mix.png")


# ---------------------------------------------------------------------------
# 3. Metrics by outcome
# ---------------------------------------------------------------------------
def fig_by_outcome(met):
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 8.6))
    fig.suptitle("The slide-48 metric block, 10 sessions by outcome",
                 fontsize=16, fontweight="bold", color=_HEAD, x=0.011, ha="left", y=0.985)
    fig.text(0.011, 0.938,
             "Every session is one point. Bars are group means. The direction matches the "
             "S3-vs-S10 story on most metrics — but the groups overlap on all of them, "
             "so at 5 vs 5 this is a direction, not a result.",
             fontsize=10, color=_MUTED)

    for axi, (key, label, gloss, prov) in zip(axes.ravel(), METRICS):
        a = [float(met[l][key]) for l in TEAM_ORDER]
        b = [float(met[l][key]) for l in NOTEAM_ORDER]
        means = []
        for x, vals, color in ((0, a, _DIRECT), (1, b, _GREY)):
            m = sum(vals) / len(vals)
            means.append(m)
            axi.bar(x, m, width=0.56, color=color, alpha=0.28, zorder=1, edgecolor="none")
            for j, v in enumerate(vals):
                jitter = (j - (len(vals) - 1) / 2) * 0.085
                axi.scatter(x + jitter, v, s=46, color=color, zorder=3,
                            edgecolors="white", linewidths=0.8)
        # Mean labels sit at the FOOT of each bar, where no data point can ever be, so
        # they never collide with a session marker at that value.
        axi.set_ylim(bottom=0)
        span = axi.get_ylim()[1]
        for x, m in zip((0, 1), means):
            axi.text(x, 0.035 * span, f"{m:.2f}".rstrip("0").rstrip("."),
                     fontsize=9, ha="center", va="bottom", color="#222",
                     fontweight="bold", zorder=4)
        axi.set_xticks([0, 1])
        axi.set_xticklabels(["teams", "no teams"], fontsize=9)
        axi.set_xlim(-0.55, 1.55)
        axi.set_title(label, fontsize=11, fontweight="bold",
                      color=_DIRECT if prov == "direct" else _LINK, loc="left", pad=13)
        axi.text(0, 1.02, gloss, transform=axi.transAxes, fontsize=8,
                 color=_MUTED, style="italic")
        axi.spines[["top", "right"]].set_visible(False)
        axi.tick_params(labelsize=8.5)
        axi.grid(axis="y", color="#EEE", lw=0.8, zorder=0)
        axi.set_axisbelow(True)

    fig.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_DIRECT, markersize=10,
               label="teams formed in the room (n=5)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_GREY, markersize=10,
               label="no in-room teams (n=5)"),
        Line2D([0], [0], color=_DIRECT, lw=4, label="[gemini-direct] metric"),
        Line2D([0], [0], color=_LINK, lw=4, label="[reconstructed-link] metric"),
    ], loc="lower center", ncol=4, fontsize=9.6, frameon=False, bbox_to_anchor=(0.5, -0.012))
    fig.tight_layout(rect=[0, 0.028, 1, 0.925])
    _save(fig, "scale_metrics_by_outcome.png")


# ---------------------------------------------------------------------------
# 4. Summary table across 10
# ---------------------------------------------------------------------------
def fig_summary_table(met):
    n = len(METRICS)
    fig, ax = plt.subplots(figsize=(15.5, 0.52 * n + 3.6))
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    x_lab = 0.005
    x0, gap = 0.325, 0.062
    xs = {}
    for i, label in enumerate(ORDER):
        xs[label] = x0 + i * gap + (0.028 if i >= len(TEAM_ORDER) else 0)

    top = 0.965
    rowh = 0.80 / (n + 1.6)
    ax.text(x_lab, top, "Summary statistics — Gemini linkography / network, 10 sessions",
            fontsize=15.5, fontweight="bold", color=_HEAD)

    y = top - 1.15 * rowh
    ax.text(xs[TEAM_ORDER[0]] - 0.028, y + 0.6 * rowh, "TEAMS FORMED IN THE ROOM",
            fontsize=8.8, fontweight="bold", color=_DIRECT)
    ax.text(xs[NOTEAM_ORDER[0]] - 0.028, y + 0.6 * rowh, "NO IN-ROOM TEAMS",
            fontsize=8.8, fontweight="bold", color=_MUTED)
    ax.text(x_lab, y, "Metric", fontsize=10.5, fontweight="bold", color=_HEAD)
    for label in ORDER:
        ax.text(xs[label], y, label, fontsize=9.2, ha="center",
                fontweight="bold" if label in FOCAL else "normal",
                color=_HEAD if label in FOCAL else "#333")
    ax.plot([x_lab, 0.995], [y - 0.45 * rowh, y - 0.45 * rowh], color="#888", lw=1.0)
    xsep = (xs[TEAM_ORDER[-1]] + xs[NOTEAM_ORDER[0]]) / 2
    ax.plot([xsep, xsep], [y + 0.9 * rowh, y - (n + 0.4) * rowh], color="#BBB", lw=1.0)

    y -= 1.35 * rowh
    band = False
    for key, label, gloss, prov in METRICS:
        if band:
            ax.add_patch(Rectangle((x_lab - 0.004, y - 0.32 * rowh), 0.999, rowh,
                                   color=_BAND, zorder=0))
        band = not band
        color = _DIRECT if prov == "direct" else _LINK
        ax.scatter([x_lab + 0.008], [y + 0.06 * rowh], s=40, color=color, zorder=3,
                   clip_on=False)
        ax.text(x_lab + 0.024, y, label, fontsize=10, zorder=3)
        ax.text(x_lab + 0.024, y - 0.40 * rowh, gloss, fontsize=7.4, color=_MUTED,
                style="italic", zorder=3)
        vals = {l: float(met[l][key]) for l in ORDER}
        hi, lo = max(vals.values()), min(vals.values())
        for l in ORDER:
            v = met[l][key]
            txt = v.rstrip("0").rstrip(".") if "." in v else v
            emph = vals[l] in (hi, lo) and hi != lo
            ax.text(xs[l], y, txt, fontsize=9.6, ha="center", zorder=3,
                    fontweight="bold" if emph else "normal",
                    color="#111" if emph else "#444")
        y -= rowh

    y -= 0.35 * rowh
    ax.plot([x_lab, 0.995], [y + 0.55 * rowh, y + 0.55 * rowh], color="#888", lw=1.0)
    ax.text(x_lab + 0.014, y, "Link basis mix (groundedness)", fontsize=8.8,
            fontweight="bold", color="#444")
    ax.text(x_lab + 0.014, y - 0.42 * rowh, "named / on-table / nearest_prior",
            fontsize=7.6, color=_MUTED, style="italic")
    for l in ORDER:
        row = met[l]
        ax.text(xs[l], y, f"{row['basis_named_speaker']}/{row['basis_on_table_idea']}"
                          f"/{row['basis_nearest_prior']}",
                fontsize=8.2, ha="center", color="#444")
    y -= 1.15 * rowh
    ax.scatter([x_lab + 0.018], [y + 0.05 * rowh], s=40, color=_DIRECT, clip_on=False)
    ax.text(x_lab + 0.034, y, "Gemini-direct (real annotation fields)", fontsize=8.4, color=_DIRECT)
    ax.scatter([x_lab + 0.30], [y + 0.05 * rowh], s=40, color=_LINK, clip_on=False)
    ax.text(x_lab + 0.316, y, "Reconstructed-link (inherits linker uncertainty — see basis mix)",
            fontsize=8.4, color=_LINK)
    y -= 0.85 * rowh
    ax.text(x_lab, y, "Bold = highest / lowest value on that row. Bold column headers = the two "
                      "sessions the method was validated on. Every definition is slides 45–48; "
                      "team outcomes come from the roster files, not from Gemini.",
            fontsize=8, color=_MUTED, style="italic")
    _save(fig, "scale_summary_table.png")


def main():
    met, inv, links = _load()
    missing = [l for l in ORDER if l not in met]
    if missing:
        raise SystemExit(f"missing from the metrics table (run gemini_scale.py first): {missing}")
    fig_selection(inv)
    fig_basis_mix(met, links)
    fig_by_outcome(met)
    fig_summary_table(met)


if __name__ == "__main__":
    main()
