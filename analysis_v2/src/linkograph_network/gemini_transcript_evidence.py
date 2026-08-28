"""
Transcript evidence tables — trace each session's linkograph and network back to what
was actually said.

Deck slide 44 does this by hand for NES_S3: a short thread shown as
`# | Speaker | Text | Gemini code | Links to`, so a reader can check the linkograph
against the transcript instead of taking it on trust. This script produces the same
thing for every session in the scaled run, plus the network counterpart.

Two tables per session:

  A. IDEA THREAD (confirms the LINKOGRAPH)
     The session's longest build-chain — the deepest path through the link DAG, which is
     the `longest_chain` metric on the slide-48 block. One row per utterance in the
     chain: index, speaker, timestamp, the quoted evidence Gemini cited, the code it
     assigned, what it links back to, and which rule resolved that link. If the chain
     reads as a real conversation, the linkograph's spine is real.

  B. SPEAKER EDGES (confirms the NETWORK)
     Every cross-speaker edge, heaviest first, with the underlying links spelled out:
     who built on whom, how many times, and a quoted example from each end. A network
     edge is an aggregate of links, so this is where an edge is checked.

The quoted text is the `evidence` field of the Gemini code — the same source slide 44
quotes from. It is Gemini's citation of the utterance, not a separate transcript file.

Usage:
    python analysis_v2/src/linkograph_network/gemini_transcript_evidence.py
    python analysis_v2/src/linkograph_network/gemini_transcript_evidence.py --all --max-edges 8
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import networkx as nx

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))

TBL = os.path.join(_BASE, "analysis_v2/results/tables/prototypes/scale10")
OUT = os.path.join(TBL, "transcript_evidence")

TEAM_ORDER = ["NES_S3", "NES_S4", "MZT_S5", "MND_S5", "CMC_S11"]
NOTEAM_ORDER = ["NES_S10", "NES_S8", "ABI_S1", "MND_S15", "SLU_S5"]
FOCAL = {"NES_S3", "NES_S10"}

BASIS_NOTE = {
    "named_speaker": "explanation named the person",
    "on_table_idea": "matched an idea on the table",
    "nearest_prior": "fallback — most recent prior idea",
}


def _quote(text, limit=150):
    """A slide-44-style quotation: trimmed, ellipsed, safe inside a markdown cell."""
    t = " ".join((text or "").split()).replace("|", "/")
    if len(t) > limit:
        t = t[:limit].rsplit(" ", 1)[0] + "…"
    return f"“{t}”" if t else "—"


def load_session(label):
    path = os.path.join(TBL, "sessions", label, f"gemini_linkography_{label}.json")
    with open(path) as f:
        link = json.load(f)
    utts = {u["idx"]: u for u in link["utterances"]}
    return link, utts


def idea_thread(link, utts):
    """The longest build-chain, as an ordered list of utterance indices. Same DAG as
    the `longest_chain` metric: an edge runs earlier-idea -> the move that builds on it."""
    D = nx.DiGraph()
    D.add_nodes_from(utts)
    for e in link["edges"]:
        if e["from"] in utts and e["to"] in utts and e["to"] != e["from"]:
            D.add_edge(e["to"], e["from"])
    if D.number_of_edges() == 0:
        return []
    return nx.dag_longest_path(D)


def thread_rows(label, link, utts, chain):
    """One row per utterance in the chain, in slide-44 column order."""
    incoming = {}
    for e in link["edges"]:
        incoming.setdefault(e["from"], []).append(e)
    rows = []
    for pos, idx in enumerate(chain):
        u = utts[idx]
        edges = [e for e in incoming.get(idx, []) if e["to"] in chain]
        e = edges[0] if edges else None
        rows.append({
            "session": label,
            "position": pos + 1,
            "utterance_id": idx,
            "speaker": u.get("speaker", ""),
            "timestamp": u.get("timestamp", ""),
            "text": " ".join((u.get("evidence") or "").split()),
            "gemini_code": ", ".join(u.get("move_names", [])),
            "links_to": f"#{e['to']} ({utts[e['to']].get('speaker','')})" if e else "— (new idea)",
            "how_resolved": BASIS_NOTE.get(e["basis"], "") if e else "",
            "links_to_name": (e or {}).get("links_to_name") or "",
        })
    return rows


def edge_rows(label, link, utts, max_edges):
    """Cross-speaker edges, heaviest first, each with a worked example."""
    pairs = defaultdict(list)
    for e in link["edges"]:
        a = utts.get(e["from"], {}).get("speaker")
        b = utts.get(e["to"], {}).get("speaker")
        if not a or not b or a == b:
            continue
        pairs[tuple(sorted((a, b)))].append(e)
    ranked = sorted(pairs.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    rows = []
    for (a, b), es in ranked[:max_edges] if max_edges else ranked:
        ex = es[0]
        rows.append({
            "session": label,
            "speaker_a": a,
            "speaker_b": b,
            "n_links": len(es),
            "example_builder": utts[ex["from"]].get("speaker", ""),
            "example_move": ex["move_name"],
            "example_text": " ".join((utts[ex["from"]].get("evidence") or "").split()),
            "builds_on_speaker": utts[ex["to"]].get("speaker", ""),
            "builds_on_text": " ".join((utts[ex["to"]].get("evidence") or "").split()),
            "how_resolved": BASIS_NOTE.get(ex["basis"], ""),
        })
    return rows


def write_markdown(label, link, threads, edges, path):
    n_idea = len(link["utterances"])
    lines = [f"# Transcript evidence — {label}", ""]
    lines.append(f"Session `{link['session']}` · {n_idea} idea-bearing utterances · "
                 f"{len(link['edges'])} links. Quoted text is the `evidence` field of the "
                 f"Gemini code — the same source deck slide 44 quotes from.")
    lines.append("")
    lines.append("## A. Longest idea thread — confirms the linkograph")
    lines.append("")
    if not threads:
        lines.append("_No multi-step chain in this session._")
    else:
        lines.append("| # | Speaker | Text | Gemini code | Links to | How the link was resolved |")
        lines.append("|---|---|---|---|---|---|")
        for r in threads:
            lines.append(f"| {r['utterance_id']} | {r['speaker']} | {_quote(r['text'])} | "
                         f"`{r['gemini_code']}` | {r['links_to']} | {r['how_resolved'] or '—'} |")
        lines.append("")
        n_self = sum(1 for r in threads if r["links_to"] != "— (new idea)"
                     and r["speaker"] and f"({r['speaker']})" in r["links_to"])
        n_fallback = sum(1 for r in threads if r["how_resolved"].startswith("fallback"))
        lines.append(f"_{len(threads)} utterances, {threads[0]['speaker']} → "
                     f"{threads[-1]['speaker']}. This is the session's deepest build-chain "
                     f"(the `longest_chain` metric)._")
        lines.append("")
        lines.append(f"_{n_self} of these steps are the same speaker building on their own "
                     f"earlier idea. Those are real linkograph links; they are excluded only "
                     f"from the speaker NETWORK, where they are counted separately as the "
                     f"self-link index. {n_fallback} step(s) were resolved by the "
                     f"`nearest_prior` fallback — the weakest tier (~55% correct), worth "
                     f"reading closely._")
    lines.append("")
    lines.append("## B. Speaker edges — confirms the network")
    lines.append("")
    lines.append("| Pair | Links | Who built on whom | Their words | Building on | How resolved |")
    lines.append("|---|---|---|---|---|---|")
    for r in edges:
        lines.append(f"| {r['speaker_a']} ↔ {r['speaker_b']} | {r['n_links']} | "
                     f"{r['example_builder']} (`{r['example_move']}`) | "
                     f"{_quote(r['example_text'], 110)} | {r['builds_on_speaker']}: "
                     f"{_quote(r['builds_on_text'], 110)} | {r['how_resolved']} |")
    lines.append("")
    lines.append("_Edge weight is the number of cross-speaker links between that pair; the "
                 "row shows one of them in full. Self-links are excluded from the network "
                 "and counted separately as the self-link index._")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true",
                    help="include NES_S3 / NES_S10 (skipped by default — slide 44 already "
                         "covers NES_S3 by hand)")
    ap.add_argument("--max-edges", type=int, default=6,
                    help="speaker edges per session in table B (0 = all)")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    labels = [l for l in TEAM_ORDER + NOTEAM_ORDER if args.all or l not in FOCAL]
    all_threads, all_edges = [], []

    for label in labels:
        link, utts = load_session(label)
        chain = idea_thread(link, utts)
        threads = thread_rows(label, link, utts, chain)
        edges = edge_rows(label, link, utts, args.max_edges)
        all_threads += threads
        all_edges += edges
        md = os.path.join(OUT, f"evidence_{label}.md")
        write_markdown(label, link, threads, edges, md)
        print(f"  {label:8} thread of {len(threads):2} utterances, "
              f"{len(edges)} edges  -> {os.path.relpath(md, _BASE)}")

    for name, rows in (("idea_threads.csv", all_threads), ("speaker_edges.csv", all_edges)):
        if not rows:
            continue
        path = os.path.join(OUT, name)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Saved: {os.path.relpath(path, _BASE)}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
