"""
Candidate-session inventory for scaling the Gemini linkography/network pipeline.

Slide 50 of the deck ("stay on two sessions before scaling") is now satisfied: the five
validation checks passed on NES_S3 / NES_S10, with two standing caveats -- (A) the
20-30% `nearest_prior` fallback and (B) session variability. This script is the first
step of scaling: it walks every annotated session under `outputs/`, computes the
quantities the selection criteria are stated in, and marks which sessions are eligible
to join the two focal ones.

It runs the SAME loading + node/move + linking code as the single-session pipeline
(`gemini_linkography.py`) -- nothing here re-defines an idea, a move, or a link. It is a
dry run whose only output is a table: no linkographs, networks or figures are written.

The criteria (each is one column in the CSV, prefixed `ok_`):

  ok_complete    every chunk json parses. A handful of raw Gemini dumps are truncated
                 (they carry an ATTN_ prefix); a session missing part of its transcript
                 is not comparable to a complete one.
  ok_material    >= MIN_IDEA idea-bearing utterances -- enough of a linkograph to carry
                 the metrics. Floor set below both focal sessions (S3 58, S10 47).
  ok_group       MIN_SPK..MAX_SPK speakers with >=1 idea utterance -- "multi-person team
                 meeting" of the same shape as the focal pair (both 11).
  ok_length      MIN_UTT..MAX_UTT total annotated utterances -- excludes the truncated
                 and the marathon outliers (focal pair: 139 and 162).
  ok_roster      >= MIN_ROSTER of the idea speakers resolve to the session roster in
                 `<conf>_session_outcomes_v2.json` under the same fuzzy match the network
                 figure uses. Below this the speaker labels are too noisy to trust the
                 nodes (the name-matching fragility flagged on deck slide 46).
  ok_grounded    `nearest_prior` share <= MAX_NEAREST. This is caveat (A) made a gate:
                 the validated band was 20% (S3) to 30% (S10), and the spot-check put
                 nearest_prior link accuracy at ~55%. A session whose links are mostly
                 fallback is not evidence, so cap it just above the validated worst case.

`eligible` = all six. Selection among eligible sessions is a separate, deliberate step
(outcome contrast + conference spread) -- see `scale_sessions.json` and the
`--selection-report` output.

Usage:
    python analysis_v2/src/linkograph_network/gemini_session_inventory.py \
        --out-csv analysis_v2/results/tables/prototypes/scale10/session_inventory.csv
"""

import argparse
import csv
import difflib
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gemini_linkography import (          # noqa: E402  (path set above)
    extract_nodes_moves,
    link_moves,
    load_utterances,
    session_files,
)

_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# --- criteria thresholds (single place to retune) ---------------------------------
MIN_IDEA = 35          # idea-bearing utterances
MIN_SPK, MAX_SPK = 8, 13
MIN_UTT, MAX_UTT = 90, 230
MIN_ROSTER = 0.70      # share of idea speakers resolvable to the session roster
MAX_NEAREST = 0.35     # share of links resolved by the nearest_prior fallback

FOCAL = ["2020_11_05_NES_S3", "2020_11_06_NES_S10"]


def _resolve(name, roster):
    """Same fuzzy speaker match the network figure uses (difflib, cutoff 0.85)."""
    if name in roster:
        return name
    m = difflib.get_close_matches(name.title(), list(roster), n=1, cutoff=0.85)
    return m[0] if m else None


def survey_session(session_dir, conf, outcomes):
    """One row of the inventory for one session directory. Runs the real pipeline
    functions, so every number here is the number the scaled run will produce."""
    session = os.path.basename(session_dir.rstrip("/")).replace("output_", "")
    bad = []
    utts = extract_nodes_moves(load_utterances(session_dir, on_bad_file=lambda p, e: bad.append(p)))
    edges, unresolved = link_moves(utts)

    idea = [u for u in utts if u["is_node"] or u["is_move"]]
    speakers = sorted({u["speaker"] for u in idea if u["speaker"]})
    basis = Counter(e["basis"] for e in edges)
    n_links = len(edges)
    n_extracted = sum(1 for e in edges if e.get("links_to_name"))

    outcome = outcomes.get(session, {})
    roster = outcome.get("all_speakers_raw") or []
    matched = [s for s in speakers if _resolve(s, roster)]
    roster_rate = len(matched) / len(speakers) if speakers else 0.0
    pct_nearest = basis["nearest_prior"] / n_links if n_links else 1.0

    row = {
        "conf": conf,
        "session": session,
        "session_dir": os.path.relpath(session_dir, _BASE),
        "n_files": len(session_files(session_dir)),
        "n_bad_files": len(bad),
        "n_utts": len(utts),
        "n_idea": len(idea),
        "n_nodes": sum(u["is_node"] for u in utts),
        "n_moves": sum(u["is_move"] for u in utts),
        "n_speakers_idea": len(speakers),
        "n_roster": len(roster),
        "roster_match_rate": round(roster_rate, 3),
        "unmatched_speakers": ";".join(s for s in speakers if not _resolve(s, roster)),
        "speaking_seconds": sum(u.get("speaking_duration_seconds") or 0 for u in utts),
        "n_links": n_links,
        "n_unresolved_moves": len(unresolved),
        "basis_named_speaker": basis["named_speaker"],
        "basis_on_table_idea": basis["on_table_idea"],
        "basis_nearest_prior": basis["nearest_prior"],
        "pct_named_speaker": round(basis["named_speaker"] / n_links, 3) if n_links else 0.0,
        "pct_nearest_prior": round(pct_nearest, 3),
        "pct_links_to_extracted": round(n_extracted / n_links, 3) if n_links else 0.0,
        "has_teams": outcome.get("has_teams"),
        "num_teams": outcome.get("num_teams"),
        "num_funded_teams": outcome.get("num_funded_teams"),
        "is_focal": session in FOCAL,
    }
    row["ok_complete"] = not bad
    row["ok_material"] = row["n_idea"] >= MIN_IDEA
    row["ok_group"] = MIN_SPK <= row["n_speakers_idea"] <= MAX_SPK
    row["ok_length"] = MIN_UTT <= row["n_utts"] <= MAX_UTT
    row["ok_roster"] = roster_rate >= MIN_ROSTER
    row["ok_grounded"] = pct_nearest <= MAX_NEAREST
    row["eligible"] = all(row[k] for k in
                          ("ok_complete", "ok_material", "ok_group", "ok_length",
                           "ok_roster", "ok_grounded"))
    return row


def build_inventory(outputs_root=None, data_root=None):
    """Survey every session under `outputs/`, newest-style conf dirs and all."""
    outputs_root = outputs_root or os.path.join(_BASE, "outputs")
    data_root = data_root or os.path.join(_BASE, "analysis_v1", "data")
    rows = []
    for conf in sorted(os.listdir(outputs_root)):
        cdir = os.path.join(outputs_root, conf)
        if not os.path.isdir(cdir):
            continue
        opath = os.path.join(data_root, conf, f"{conf}_session_outcomes_v2.json")
        outcomes = json.load(open(opath)) if os.path.exists(opath) else {}
        for sd in sorted(os.listdir(cdir)):
            sdir = os.path.join(cdir, sd)
            if os.path.isdir(sdir):
                rows.append(survey_session(sdir, conf, outcomes))
    return rows


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-csv",
                    default="analysis_v2/results/tables/prototypes/scale10/session_inventory.csv")
    args = ap.parse_args()

    rows = build_inventory()
    write_csv(rows, args.out_csv)

    elig = [r for r in rows if r["eligible"]]
    print(f"Saved: {args.out_csv}")
    print(f"{len(rows)} sessions surveyed, {len(elig)} eligible "
          f"({sum(1 for r in elig if r['is_focal'])} of them focal).")
    print("\nFailures by criterion (a session can fail several):")
    for k in ("ok_complete", "ok_material", "ok_group", "ok_length", "ok_roster", "ok_grounded"):
        print(f"  {k:14} failed by {sum(1 for r in rows if not r[k]):3d}")
    print(f"\nEligible by conference: "
          f"{dict(Counter(r['conf'] for r in elig))}")
    print(f"Eligible by has_teams: {dict(Counter(r['has_teams'] for r in elig))}")


if __name__ == "__main__":
    main()
