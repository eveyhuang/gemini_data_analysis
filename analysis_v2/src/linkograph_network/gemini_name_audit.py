"""
Audit: how often does Gemini's explanation say WHOSE idea is being built on?

This is the question Evey called priority one. The linker's strongest rule
(`named_speaker`) fires only when a move's `explanation` names another participant. When
it doesn't, the linker falls back to word-overlap or, failing that, to the most recent
prior idea -- and the fallback tier is the least accurate. So the rate at which Gemini
names the target sets a ceiling on how well-grounded the whole link layer can be.

Measured across every annotated session, not just the ten in the scaled run, and broken
down three ways:

  by conference   -- does the naming rate travel? (it does not: 2020NES is the best case)
  by move type    -- are some codes more likely to name a person than others?
  by session      -- distribution, so we can say whether low-naming sessions are a tail
                     or the norm

Three quantities, deliberately kept apart because they answer different questions:

  names_anyone    the explanation mentions SOME participant, including the speaker
                  themselves ("Jenny builds on her earlier point"). Upper bound on the
                  signal present.
  names_other     it mentions a participant OTHER than the speaker. This is what rule 1
                  actually looks for.
  rule_fired      `named_speaker` actually resolved the link. Lower than names_other
                  whenever the named person has no prior idea utterance to attach to --
                  the signal was there and we still could not use it.

The gap between names_other and rule_fired is recoverable signal. The gap between
names_anyone and names_other is not (self-reference cannot identify a target).

Usage:
    python analysis_v2/src/linkograph_network/gemini_name_audit.py
"""

import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gemini_linkography import (                                   # noqa: E402
    MOVE,
    extract_nodes_moves,
    link_moves,
    load_utterances,
    _first_names,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
OUT = os.path.join(_BASE, "analysis_v2/results/tables/prototypes/scale10/name_audit")


def audit_session(session_dir, conf, session):
    """One row per MOVE CODE (not per utterance -- a turn can carry several)."""
    utts = extract_nodes_moves(load_utterances(session_dir, on_bad_file=lambda p, e: None))
    speakers = sorted({u["speaker"] for u in utts if u["speaker"]})
    fn = _first_names(speakers)
    edges, _ = link_moves(utts)
    # basis per (from_idx, move_name) so we can line codes up with the rule that fired
    basis = {(e["from"], e["move_name"]): e["basis"] for e in edges}

    rows = []
    for u in utts:
        for m in u["moves"]:
            if m["role"] != MOVE:
                continue
            expl = (m["explanation"] or "").lower()
            hits = [fn[k] for k in sorted(fn) if re.search(rf"\b{re.escape(k)}\b", expl)]
            others = [h for h in hits if h != u["speaker"]]
            rows.append({
                "conf": conf,
                "session": session,
                "speaker": u["speaker"],
                "move_name": m["name"],
                "names_anyone": bool(hits),
                "names_other": bool(others),
                "named_person": others[0] if others else "",
                "rule_fired": basis.get((u["idx"], m["name"]), "unresolved"),
                "explanation": " ".join((m["explanation"] or "").split())[:300],
            })
    return rows


def pct(n, d):
    return f"{100 * n / d:.1f}%" if d else "—"


def summarize(rows, key, label, min_n=0):
    """Group rows by `key` and report the three quantities."""
    g = defaultdict(list)
    for r in rows:
        g[r[key]].append(r)
    out = []
    for k, v in sorted(g.items(), key=lambda kv: -len(kv[1])):
        if len(v) < min_n:
            continue
        n = len(v)
        out.append({
            label: k,
            "moves": n,
            "names_anyone": pct(sum(r["names_anyone"] for r in v), n),
            "names_other": pct(sum(r["names_other"] for r in v), n),
            "rule_fired": pct(sum(r["rule_fired"] == "named_speaker" for r in v), n),
            "_recoverable": sum(r["names_other"] and r["rule_fired"] != "named_speaker" for r in v),
        })
    return out


def table(rows, label):
    if not rows:
        return "  (none)\n"
    w = max(len(str(r[label])) for r in rows) + 2
    s = (f"  {label:<{w}}{'moves':>7}{'names anyone':>15}{'names other':>14}"
         f"{'rule fired':>13}{'recoverable':>13}\n")
    s += "  " + "-" * (w + 62) + "\n"
    for r in rows:
        s += (f"  {str(r[label]):<{w}}{r['moves']:>7}{r['names_anyone']:>15}"
              f"{r['names_other']:>14}{r['rule_fired']:>13}{r['_recoverable']:>13}\n")
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs-root", default=os.path.join(_BASE, "outputs"))
    args = ap.parse_args()

    rows = []
    for conf in sorted(os.listdir(args.outputs_root)):
        cdir = os.path.join(args.outputs_root, conf)
        if not os.path.isdir(cdir):
            continue
        for sd in sorted(os.listdir(cdir)):
            sdir = os.path.join(cdir, sd)
            if os.path.isdir(sdir):
                rows += audit_session(sdir, conf, sd.replace("output_", ""))

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "move_level_naming.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    anyone = sum(r["names_anyone"] for r in rows)
    other = sum(r["names_other"] for r in rows)
    fired = sum(r["rule_fired"] == "named_speaker" for r in rows)
    recoverable = sum(r["names_other"] and r["rule_fired"] != "named_speaker" for r in rows)

    lines = []
    lines.append("# Missing-names audit\n")
    lines.append(f"Every move code in every annotated session: **{n:,} moves** across "
                 f"{len({r['session'] for r in rows})} sessions and "
                 f"{len({r['conf'] for r in rows})} conferences.\n")
    lines.append("## Headline\n")
    lines.append(f"| Quantity | Count | Share |")
    lines.append(f"|---|---|---|")
    lines.append(f"| Explanation names someone (incl. the speaker) | {anyone:,} | {pct(anyone, n)} |")
    lines.append(f"| Names someone **other** than the speaker — what rule 1 needs | {other:,} | {pct(other, n)} |")
    lines.append(f"| `named_speaker` actually resolved the link | {fired:,} | {pct(fired, n)} |")
    lines.append(f"| **Named but unusable** (no prior idea by that person) | {recoverable:,} | {pct(recoverable, n)} |")
    lines.append("")
    lines.append(f"So Gemini fails to identify the target for **{pct(n - other, n)}** of moves. "
                 f"That is not rare — it is the majority case, and it is why the "
                 f"`on_table_idea` and `nearest_prior` rules carry most of the link load.\n")
    lines.append("## By conference\n")
    lines.append("```")
    lines.append(table(summarize(rows, "conf", "conference"), "conference").rstrip())
    lines.append("```\n")
    lines.append("## By move type\n")
    lines.append("```")
    lines.append(table(summarize(rows, "move_name", "move code", min_n=20), "move code").rstrip())
    lines.append("```\n")

    per_sess = summarize(rows, "session", "session", min_n=5)
    vals = sorted(float(r["names_other"].rstrip("%")) for r in per_sess)
    lines.append("## By session — is low naming a tail or the norm?\n")
    lines.append(f"- sessions measured: **{len(vals)}**")
    lines.append(f"- median naming rate: **{vals[len(vals)//2]:.1f}%**")
    lines.append(f"- worst quartile: **≤ {vals[len(vals)//4]:.1f}%**  ·  "
                 f"best quartile: **≥ {vals[3*len(vals)//4]:.1f}%**")
    lines.append(f"- sessions where Gemini names the target in **under 10%** of moves: "
                 f"**{sum(1 for v in vals if v < 10)}**")
    lines.append(f"- sessions **above 30%**: **{sum(1 for v in vals if v > 30)}**\n")

    with open(os.path.join(OUT, "NAME_AUDIT.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved: {os.path.relpath(OUT, _BASE)}/NAME_AUDIT.md and move_level_naming.csv")


if __name__ == "__main__":
    main()
