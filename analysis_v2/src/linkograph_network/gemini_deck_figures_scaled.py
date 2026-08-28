"""
Deck-style linkographs + networks for the scaled session set.

`gemini_scale.py` renders the PROTOTYPE figure style (node vs move colouring, plain
sociogram). The deck uses a different, richer style for NES_S3 / NES_S10 -- speaker-
identity node colours in BOTH figures so "blue is the same person in the linkograph and
the network" (Evey's encoding), plus dashed self-link arcs, critical-move outlines,
convergence / rabbit-hole rings, team-outcome borders and in-room teammate rings.

That style lives in `analysis_v2/src/session_story/` (figures.py + viz_encodings.py) and
is driven for the two focal sessions by `session_story/gemini_deck_figures.py`, whose
SESSIONS dict is hard-coded to that pair.

This module reuses that script wholesale -- same renderer, same encodings, same legends --
and simply repoints it at the scaled run's linkography JSONs. Nothing about the styling or
the data conversion is reimplemented here; if the deck's look changes, it changes here too.

Output goes to analysis_v2/figures/scale10_deck_style/ so neither the prototype figures
nor the existing session_story_gemini/ pair is overwritten.

Usage:
    python analysis_v2/src/linkograph_network/gemini_deck_figures_scaled.py
    python analysis_v2/src/linkograph_network/gemini_deck_figures_scaled.py --all
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
BASE = _HERE.parent.parent.parent

# session_story owns the renderer; prototypes owns modules figures.py imports.
sys.path.insert(0, str(BASE / "analysis_v2/src/session_story"))
sys.path.insert(0, str(BASE / "analysis_v2/src/prototypes"))
sys.path.insert(0, str(_HERE))

import networkx as nx                              # noqa: E402
import gemini_deck_figures as G                     # noqa: E402
import figures as F                                 # noqa: E402
import speaker_network_metrics as snm               # noqa: E402


# --- determinism ------------------------------------------------------------------
# speaker_network_metrics.build_speaker_graph() collects speakers into a SET and adds
# them to the graph in set-iteration order, which varies per process under hash
# randomisation. spring_layout(seed=7) seeds the RNG but lays nodes out in insertion
# order, so the same session renders with a different layout on every run -- the
# original gemini_deck_figures.py does not reproduce its own output either.
#
# Rebuild the graph with sorted node order so these figures are reproducible. Patched
# here rather than in speaker_network_metrics.py because that module is shared with the
# Liu deck figures; the one-line fix there is `G.add_nodes_from(sorted(all_speakers))`.
_ORIG_BUILD = snm.build_speaker_graph


def _deterministic_build(session_data):
    g = _ORIG_BUILD(session_data)
    h = nx.Graph()
    h.add_nodes_from(sorted(g.nodes()))
    h.add_edges_from(g.edges(data=True))
    return h


snm.build_speaker_graph = _deterministic_build


# --- axis label -------------------------------------------------------------------
# figures.py labels the linkograph x-axis "Relevant-message sequence (irrelevant
# omitted)". That is Liu vocabulary: Liu applies a TOPIC FILTER and genuinely drops
# off-topic turns. The Gemini pipeline has no topic filter -- nothing is dropped for
# being irrelevant. What is off the graph is every turn carrying no idea-move code
# (Pronoun Framing, Knowledge Sharing, Relational Climate, Coordination, ...), which is
# the "partial coverage" caveat on deck slide 45, not a relevance judgement.
#
# gemini_linkography.draw_linkograph already words this correctly; the deck renderer
# inherited Liu's phrasing. Swap that one exact string.
import matplotlib.axes                              # noqa: E402

_LIU_XLABEL = "Relevant-message sequence (irrelevant omitted)"
_GEMINI_XLABEL = "Idea-bearing utterance sequence (turns with no idea-move code omitted)"
_ORIG_SET_XLABEL = matplotlib.axes.Axes.set_xlabel


def _set_xlabel(self, xlabel, *args, **kwargs):
    if xlabel == _LIU_XLABEL:
        xlabel = _GEMINI_XLABEL
    return _ORIG_SET_XLABEL(self, xlabel, *args, **kwargs)


matplotlib.axes.Axes.set_xlabel = _set_xlabel

# Same substitution in the linkograph subtitle, which reads "<n> relevant messages".
# "Relevant" is Liu's topic-filter word; here the count is idea-bearing utterances.
_ORIG_SET_TITLE = matplotlib.axes.Axes.set_title


def _set_title(self, label=None, *args, **kwargs):
    if isinstance(label, str) and " relevant messages · link ratio " in label:
        label = label.replace(" relevant messages · link ratio ",
                              " idea-bearing utterances · link ratio ")
    return _ORIG_SET_TITLE(self, label, *args, **kwargs)


matplotlib.axes.Axes.set_title = _set_title

SCALE_TBL = BASE / "analysis_v2/results/tables/prototypes/scale10"
# Written into each session's own folder alongside the prototype-style figures
# gemini_scale.py produces, so everything for one session lives together:
#   scale10/<LABEL>/gemini_linkography_<LABEL>.png   prototype style
#   scale10/<LABEL>/linkograph_<LABEL>_gemini.png    deck style (this script)
# The filenames differ, so neither script overwrites the other's output.
OUT_FIG = BASE / "analysis_v2/figures/prototypes/scale10"

# Same order as deck slide 62: teams first, original session first within each group.
TEAM_ORDER = ["NES_S3", "NES_S4", "MZT_S5", "MND_S5", "CMC_S11"]
NOTEAM_ORDER = ["NES_S10", "NES_S8", "ABI_S1", "MND_S15", "SLU_S5"]
FOCAL = {"NES_S3", "NES_S10"}


def _session_meta():
    """label -> (date 'MM/DD', in-room team count, funded count), from the run's own
    inventory so the titles can never drift from the numbers on the slides."""
    inv = {r["session"]: r for r in csv.DictReader(open(SCALE_TBL / "session_inventory.csv"))}
    cfg = json.load(open(_HERE / "scale_sessions.json"))
    out = {}
    for entry in cfg["sessions"]:
        sid = entry["id"] if isinstance(entry, dict) else entry
        p = sid.split("_")
        label = f"{p[3]}_{p[4]}"
        r = inv[sid]
        teams = int(r["num_teams"]) if r["has_teams"] == "True" else 0
        out[label] = (f"{p[1]}/{p[2]}", teams, int(r["num_funded_teams"]))
    return out


def _title(label, date, teams, funded):
    """Matches the phrasing already on deck slides 57-60."""
    if teams == 0:
        outcome = "0 teams formed within this room"
    else:
        noun = "team" if teams == 1 else "teams"
        outcome = f"{teams} {noun} formed WITHIN this room"
        if funded:
            outcome += f" ({funded} funded)"
    return f"{label} ({date}) — Gemini linkography · {outcome}"


def build_sessions(include_focal=False):
    """The SESSIONS dict gemini_deck_figures.build_and_register() consumes, plus the
    conference each session belongs to (needed for the team-outcome lookup)."""
    meta = _session_meta()
    cfg = json.load(open(_HERE / "scale_sessions.json"))
    conf_of = {}
    for entry in cfg["sessions"]:
        sid = entry["id"] if isinstance(entry, dict) else entry
        p = sid.split("_")
        conf_of[f"{p[3]}_{p[4]}"] = p[0] + p[3]

    sessions = {}
    for label in TEAM_ORDER + NOTEAM_ORDER:
        if label in FOCAL and not include_focal:
            continue                     # already rendered on slides 57-60
        path = SCALE_TBL / "sessions" / label / f"gemini_linkography_{label}.json"
        if not path.exists():
            raise SystemExit(f"missing linkography json (run gemini_scale.py first): {path}")
        date, teams, funded = meta[label]
        sessions[f"{label}_gemini"] = {"json": path, "conf": conf_of[label],
                                       "title": _title(label, date, teams, funded)}
    return sessions


def _bind_conference(conf):
    """figures.draw_network() calls team_status(n) and in_room_teammates(...) with no
    conference argument, so both fall back to their conf="2020NES" default. That is
    correct for the two focal sessions and silently wrong for every other conference --
    e.g. MND_S5's speakers match 0 of 8 against the 2020NES roster and 4 of 8 against
    2022MND, which would draw every node as "joined no team".

    Rebind the two lookups to this session's conference for the duration of the render.
    Patched here rather than edited in figures.py so the Liu deck figures, which rely on
    the 2020NES default, are untouched."""
    F.team_status = lambda speaker, _c=conf: _ORIG_TEAM_STATUS(speaker, _c)
    F.in_room_teammates = lambda speakers, _c=conf: _ORIG_IN_ROOM(speakers, _c)


_ORIG_TEAM_STATUS = F.team_status
_ORIG_IN_ROOM = F.in_room_teammates


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true",
                    help="also re-render NES_S3 / NES_S10 (skipped by default — they are "
                         "already on slides 57-60)")
    args = ap.parse_args()

    G.SESSIONS = build_sessions(include_focal=args.all)
    # Same substitution gemini_deck_figures.main() makes: Gemini vocabulary, not Liu N/S/M.
    F.linkograph_legend = G._gemini_linkograph_legend

    for key, cfg in G.SESSIONS.items():
        label = key.rsplit("_gemini", 1)[0]
        outdir = OUT_FIG / label
        outdir.mkdir(parents=True, exist_ok=True)
        _bind_conference(cfg["conf"])
        G.build_and_register(key)
        F.draw_linkograph(key, outdir=outdir)
        F.draw_network(key, outdir=outdir)
        # draw_* also writes a .md caption in Liu's N/S/M vocabulary, which does not
        # describe a Gemini linkograph (the same reason gemini_deck_figures.main()
        # skips export_legend_spec()). Drop them rather than ship a wrong legend.
        for stale in outdir.glob(f"*_{label}_gemini.md"):
            stale.unlink()
        print(f"  {label:8} (conf={cfg['conf']})  -> {os.path.relpath(outdir, BASE)}/")
    print(f"\n{len(G.SESSIONS)} sessions rendered under {os.path.relpath(OUT_FIG, BASE)}/<LABEL>/")


if __name__ == "__main__":
    main()
