"""
Speaker network from the Gemini code-based linkography (PROTOTYPE v1).

Counterpart to `speaker_network_metrics.py`, which builds the same speaker network
from Liu's links. This one reads a `gemini_linkography_<session>.json` (produced by
`gemini_linkography.py`) and never touches Liu -- so the whole figure originates from
the Gemini annotations, matching the linkograph.

Construction (mirrors speaker_network_metrics conventions):
  - Nodes = every unique speaker with >=1 idea-bearing utterance in the session,
    including speakers with no cross-speaker link (isolated node -- informative).
  - Each linkograph edge {from, to} is a MOVE utterance (`from`) building on an
    earlier idea utterance (`to`). Map both to their speaker.
  - Self-links (a speaker building on their OWN earlier idea) are EXCLUDED from the
    graph and counted separately as a self-link index (SLI), same as the Liu version.
  - Undirected edge weight = number of cross-speaker links between that pair.
  - Edge color = additive (blue) vs structural (red), using the same move-type split
    as the linkograph, so the two figures read together.

Usage:
    python analysis_v2/src/linkograph_network/gemini_speaker_network.py \
        --linkography-json analysis_v2/results/tables/prototypes/session_comparison/gemini_linkography_NES_S3.json \
        --out-png analysis_v2/figures/prototypes/session_comparison/gemini_speaker_network_NES_S3.png
"""

import argparse
import difflib
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx

# Same additive/structural split the linkograph uses (gemini_linkography._MOVE_COLOR).
_ADDITIVE = {"extends_existing_idea", "combines_ideas", "synthesizes_contributions",
             "connects_methods", "returns_to_earlier_idea"}
_STRUCTURAL = {"critiques_or_challenges", "raises_concern", "redirects_idea",
               "resolves_contradiction"}
_BLUE, _RED = "#4C72B0", "#C44E52"

# Team-outcome palette + ring, identical to session_story/viz_encodings.py so the
# Gemini figures mark people the same way the Liu figures do.
_OUTCOME_COLOR = {"funded": "#016B78", "team": "#4FA3B0", "none": "#B8B8B8"}
_RING_COLOR = "#C71585"   # in-room teammate ring (partnered with someone from THIS session)
_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _load_p2t(conf):
    path = os.path.join(_BASE, f"analysis_v1/data/{conf}/{conf}_person_to_team.json")
    with open(path) as f:
        return json.load(f)


def _resolve_key(speaker, p2t):
    """Fuzzy name reconciliation, same 0.85 cutoff as figures.team_status."""
    if speaker in p2t:
        return speaker
    m = difflib.get_close_matches(speaker.title(), list(p2t), n=1, cutoff=0.85)
    return m[0] if m else None


def team_status(speaker, p2t):
    """'funded' | 'team' | 'none' for one speaker (mirrors figures.team_status)."""
    key = _resolve_key(speaker, p2t)
    if key is None:
        return "none"
    return "funded" if any(t["funded_status"] for t in p2t[key]) else "team"


def in_room_teammates(speakers, p2t):
    """{speaker: set(team_ids shared with ANOTHER speaker in this same session)}."""
    resolved = {}
    for s in speakers:
        k = _resolve_key(s, p2t)
        resolved[s] = {t["team_id"] for t in p2t[k]} if k else set()
    out = {}
    for s in speakers:
        shared = set()
        for other in speakers:
            if other != s:
                shared |= resolved[s] & resolved[other]
        out[s] = shared
    return out


def build_speaker_network(link, conf):
    """Return (G, stats). G is an undirected speaker graph with weighted, colored edges,
    and each node annotated with its team outcome + in-room-teammate status."""
    idx2spk = {u["idx"]: u["speaker"] for u in link["utterances"] if u.get("speaker")}
    speakers = sorted(set(idx2spk.values()))

    p2t = _load_p2t(conf)
    outcome = {s: team_status(s, p2t) for s in speakers}
    unmatched = [s for s in speakers if _resolve_key(s, p2t) is None]
    in_room = in_room_teammates(speakers, p2t)

    G = nx.Graph()
    for s in speakers:
        G.add_node(s, outcome=outcome[s], in_room=bool(in_room[s]))

    # Aggregate cross-speaker links; track self-links separately.
    pair_w = defaultdict(int)             # (a,b) sorted -> total links
    pair_type = defaultdict(lambda: defaultdict(int))  # (a,b) -> {additive/structural: n}
    self_links = defaultdict(int)
    total_cross, total_self = 0, 0

    for e in link["edges"]:
        a, b = idx2spk.get(e["from"]), idx2spk.get(e["to"])
        if a is None or b is None:
            continue
        mv = e.get("move_name", "")
        kind = "additive" if mv in _ADDITIVE else ("structural" if mv in _STRUCTURAL else "additive")
        if a == b:
            self_links[a] += 1
            total_self += 1
            continue
        key = tuple(sorted((a, b)))
        pair_w[key] += 1
        pair_type[key][kind] += 1
        total_cross += 1

    for (a, b), w in pair_w.items():
        # Edge color = whichever link type dominates this pair (tie -> additive/blue).
        t = pair_type[(a, b)]
        color = _RED if t["structural"] > t["additive"] else _BLUE
        G.add_edge(a, b, weight=w, color=color)

    stats = {
        "session": link.get("session"),
        "conf": conf,
        "n_speakers": len(speakers),
        "n_edges": G.number_of_edges(),
        "total_cross_speaker_links": total_cross,
        "total_self_links": total_self,
        "self_link_index": round(total_self / (total_cross + total_self), 3) if (total_cross + total_self) else 0.0,
        "self_links_by_speaker": dict(self_links),
        "weighted_degree": {s: sum(G[s][n]["weight"] for n in G[s]) for s in speakers},
        "degree": {s: G.degree(s) for s in speakers},
        "outcome": outcome,
        "unmatched_speakers": unmatched,
        "in_room_teammates": {s: sorted(v) for s, v in in_room.items() if v},
    }
    return G, stats


def plot_sociogram(G, stats, out_path):
    fig, ax = plt.subplots(figsize=(11, 9))
    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "no speakers", ha="center"); fig.savefig(out_path); return

    nodes = list(G.nodes())
    pos = nx.spring_layout(G, weight="weight", seed=42, k=1.2)
    wdeg = stats["weighted_degree"]
    max_wd = max(wdeg.values()) or 1
    node_sizes = [400 + 2200 * (wdeg[n] / max_wd) for n in nodes]

    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(weights) if weights else 1
    edge_widths = [1 + 6 * (w / max_w) for w in weights]
    edge_colors = [G[u][v]["color"] for u, v in G.edges()]

    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.5, ax=ax)
    # FILL = team outcome (funded / team / none); SIZE = weighted degree.
    nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_size=node_sizes,
                           node_color=[_OUTCOME_COLOR[G.nodes[n]["outcome"]] for n in nodes],
                           edgecolors="black", linewidths=0.8, ax=ax)
    # RING = ended up on a team with ANOTHER speaker from THIS session.
    ring = [n for n in nodes if G.nodes[n]["in_room"]]
    if ring:
        nx.draw_networkx_nodes(G, pos, nodelist=ring,
                               node_size=[400 + 2200 * (wdeg[n] / max_wd) + 650 for n in ring],
                               node_color="none", edgecolors=_RING_COLOR, linewidths=3.0, ax=ax)
    # Label = name + weighted degree; sits just below each node.
    label_pos = {n: (x, y - 0.06) for n, (x, y) in pos.items()}
    nx.draw_networkx_labels(G, label_pos, font_size=8.5, ax=ax,
                            labels={n: f"{n}\n(wd={wdeg[n]})" for n in nodes})

    title = (f"Gemini idea-link speaker network: {stats['session']}\n"
             f"{stats['n_speakers']} speakers, {stats['total_cross_speaker_links']} cross-speaker links, "
             f"self-link index {stats['self_link_index']}")
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_OUTCOME_COLOR["funded"], markersize=13, label="On a FUNDED team"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_OUTCOME_COLOR["team"], markersize=13, label="On a team (not funded)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_OUTCOME_COLOR["none"], markersize=13, label="No team / not in roster"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor=_RING_COLOR,
               markeredgewidth=3, markersize=15, label="Ring = teamed with someone in THIS session"),
        Line2D([0], [0], color=_BLUE, lw=3, label="Additive link (extend/combine/synthesize)"),
        Line2D([0], [0], color=_RED, lw=3, label="Structural link (critique/redirect/concern)"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8.5, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--linkography-json", required=True)
    ap.add_argument("--out-png", required=True)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--conf", default=None,
                    help="Conference key for person_to_team lookup; inferred from session id if omitted.")
    args = ap.parse_args()

    with open(args.linkography_json) as f:
        link = json.load(f)
    # Session id like '2020_11_05_NES_S3' -> conf '2020NES'.
    conf = args.conf
    if conf is None:
        p = (link.get("session") or "").split("_")
        conf = (p[0] + p[3]) if len(p) >= 4 else "2020NES"
    G, stats = build_speaker_network(link, conf)

    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    plot_sociogram(G, stats, args.out_png)
    out_json = args.out_json or args.out_png.replace("figures", "results/tables").rsplit(".", 1)[0] + ".json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(stats, f, indent=2)

    n_funded = sum(1 for v in stats["outcome"].values() if v == "funded")
    n_team = sum(1 for v in stats["outcome"].values() if v == "team")
    n_none = sum(1 for v in stats["outcome"].values() if v == "none")
    print(f"Saved: {args.out_png}")
    print(f"Saved: {out_json}")
    print(f"  conf={conf}: {stats['n_speakers']} speakers, {stats['n_edges']} edges, "
          f"{stats['total_cross_speaker_links']} cross-speaker links, SLI={stats['self_link_index']}")
    print(f"  outcome: funded={n_funded}, team={n_team}, none={n_none}; "
          f"in-room teammates={sum(1 for n in G if G.nodes[n]['in_room'])}")
    if stats["unmatched_speakers"]:
        print(f"  UNMATCHED (rendered as 'none'): {stats['unmatched_speakers']}")


if __name__ == "__main__":
    main()
