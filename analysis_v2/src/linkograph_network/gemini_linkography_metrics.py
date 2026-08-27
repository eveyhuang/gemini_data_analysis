"""
Session-comparison metrics computed on the GEMINI code-based linkography (PROTOTYPE).

Every metric here is derived from the Gemini annotations only (the
`gemini_linkography_<session>.json` produced by `gemini_linkography.py` plus the
`<conf>_person_to_team.json` roster for speaker resolution) — no Liu pipeline. This is
the metric block behind slide 48 of the deck. Each metric is tagged in the output as
[gemini-direct] (rides on real Gemini fields) or [reconstructed-link] (computed on the
linker's reconstructed edges — inherits Link-row uncertainty).

Metrics (slide 48):
  breadth          [direct]  count of node-role idea utterances (new ideas)
  volume           [direct]  count of all idea-bearing utterances (nodes + moves)
  depth_link_ratio [link]    total_links / n_idea_utterances
  longest_chain    [link]    longest build-chain (longest path in the link DAG)
  bridging_moves   [link]    union-find: +1 when a move joins >=2 not-yet-connected threads
  distribution     [direct]  turn-taking entropy over per-speaker idea-utterance counts
  self_link_ratio  [link]    self_links / total_links
  max_betweenness  [link]    max speaker betweenness centrality over the edge graph
  n_bridges_sharing[link]    # speakers with betweenness >= 50% of the max
  builder_diversity[link]    mean over new ideas of (distinct OTHER builders / total other builds)
Also reports the `basis` mix (named_speaker / on_table_idea / nearest_prior) as the
groundedness check for every [reconstructed-link] metric.

Usage:
    python analysis_v2/src/linkograph_network/gemini_linkography_metrics.py \
        --linkography-json analysis_v2/results/tables/prototypes/session_comparison/gemini_linkography_NES_S3.json \
        --linkography-json analysis_v2/results/tables/prototypes/session_comparison/gemini_linkography_NES_S10.json \
        --out-csv analysis_v2/results/tables/prototypes/session_comparison/metrics_gemini_linkography.csv
"""

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict

import networkx as nx

_ADDITIVE = {"extends_existing_idea", "combines_ideas", "synthesizes_contributions",
             "connects_methods", "returns_to_earlier_idea"}


class _DSU:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def compute_metrics(link):
    utts = link["utterances"]
    edges = link["edges"]
    idx2spk = {u["idx"]: u.get("speaker") for u in utts}
    n_idea = len(utts)
    total_links = len(edges)

    # --- [direct] breadth / volume ---
    breadth = sum(1 for u in utts if u.get("is_node"))
    volume = n_idea

    # --- [link] depth ---
    depth_link_ratio = total_links / n_idea if n_idea else 0.0

    # --- [link] longest build-chain: longest path in the DAG (edge earlier->later) ---
    D = nx.DiGraph()
    D.add_nodes_from(u["idx"] for u in utts)
    for e in edges:
        D.add_edge(e["to"], e["from"])          # earlier idea -> the move that builds on it
    longest_chain = nx.dag_longest_path_length(D) + 1 if D.number_of_nodes() else 0

    # --- [link] bridging: union-find over the link graph in time order ---
    by_from = defaultdict(list)
    for e in edges:
        by_from[e["from"]].append(e["to"])
    dsu = _DSU()
    bridging_moves = 0
    for frm in sorted(by_from):                 # time order
        targets = by_from[frm]
        roots = {dsu.find(t) for t in targets}
        if len(roots) >= 2:                     # joins >=2 not-yet-connected threads
            bridging_moves += 1
        for t in targets:                       # merge move + all its targets into one thread
            dsu.union(frm, t)

    # --- [direct] distribution: turn-taking entropy over per-speaker idea-utterance counts ---
    spk_counts = Counter(u.get("speaker") for u in utts if u.get("speaker"))
    N = len(spk_counts)
    tot = sum(spk_counts.values())
    if N > 1 and tot:
        H = -sum((c / tot) * math.log(c / tot) for c in spk_counts.values())
        distribution = H / math.log(N)
    else:
        distribution = 0.0

    # --- [link] self-link ratio + basis mix ---
    self_links = 0
    basis_mix = Counter()
    for e in edges:
        basis_mix[e.get("basis")] += 1
        if idx2spk.get(e["from"]) and idx2spk.get(e["from"]) == idx2spk.get(e["to"]):
            self_links += 1
    self_link_ratio = self_links / total_links if total_links else 0.0

    # --- [link] speaker network: betweenness + weighted degree ---
    G = nx.Graph()
    G.add_nodes_from(v for v in idx2spk.values() if v)
    pair_w = Counter()
    for e in edges:
        a, b = idx2spk.get(e["from"]), idx2spk.get(e["to"])
        if a and b and a != b:
            pair_w[tuple(sorted((a, b)))] += 1
    for (a, b), w in pair_w.items():
        G.add_edge(a, b, weight=w)
    btw = nx.betweenness_centrality(G, weight=None) if G.number_of_nodes() > 2 else {n: 0.0 for n in G}
    max_betweenness = max(btw.values()) if btw else 0.0
    n_bridges_sharing = (sum(1 for v in btw.values() if v >= 0.5 * max_betweenness)
                         if max_betweenness > 0 else 0)
    # weighted degree = sum of a speaker's edge weights (total cross-speaker link volume)
    wdeg = {n: sum(G[n][m]["weight"] for m in G[n]) for n in G}
    wd_vals = list(wdeg.values()) or [0]
    max_wd = max(wd_vals)
    avg_wd = sum(wd_vals) / len(wd_vals)
    # degree centralization on weighted degree (0 = even, ->1 = one hub carries it all)
    denom = (len(wd_vals) - 1) * max_wd
    degree_centralization = sum(max_wd - v for v in wd_vals) / denom if denom else 0.0

    # --- [link] builder diversity: mean over new ideas of distinct-other-builders / other-builds ---
    builds_on = defaultdict(list)               # target idx -> list of builder speakers
    for e in edges:
        builds_on[e["to"]].append(idx2spk.get(e["from"]))
    ratios = []
    for u in utts:
        if not u.get("is_node"):
            continue
        owner = u.get("speaker")
        others = [b for b in builds_on.get(u["idx"], []) if b and b != owner]
        if others:
            ratios.append(len(set(others)) / len(others))
    builder_diversity = sum(ratios) / len(ratios) if ratios else 0.0

    return {
        "session": link.get("session"),
        "breadth [direct]": breadth,
        "volume [direct]": volume,
        "distribution [direct]": round(distribution, 3),
        "depth_link_ratio [link]": round(depth_link_ratio, 3),
        "longest_chain [link]": longest_chain,
        "bridging_moves [link]": bridging_moves,
        "self_link_ratio [link]": round(self_link_ratio, 3),
        "max_betweenness [link]": round(max_betweenness, 3),
        "n_bridges_sharing [link]": n_bridges_sharing,
        "max_weighted_degree [link]": max_wd,
        "avg_weighted_degree [link]": round(avg_wd, 2),
        "degree_centralization [link]": round(degree_centralization, 3),
        "builder_diversity [link]": round(builder_diversity, 3),
        "n_speakers": N,
        "total_links": total_links,
        "basis_named_speaker": basis_mix.get("named_speaker", 0),
        "basis_on_table_idea": basis_mix.get("on_table_idea", 0),
        "basis_nearest_prior": basis_mix.get("nearest_prior", 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--linkography-json", action="append", required=True,
                    help="Repeat once per session.")
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    rows = []
    for path in args.linkography_json:
        with open(path) as f:
            rows.append(compute_metrics(json.load(f)))

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    cols = list(rows[0].keys())
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # Pretty print side-by-side.
    print(f"Saved: {args.out_csv}\n")
    label_w = max(len(c) for c in cols)
    header = "metric".ljust(label_w) + "".join(f"{r['session']:>22}" for r in rows)
    print(header)
    print("-" * len(header))
    for c in cols:
        if c == "session":
            continue
        print(c.ljust(label_w) + "".join(f"{str(r[c]):>22}" for r in rows))


if __name__ == "__main__":
    main()
