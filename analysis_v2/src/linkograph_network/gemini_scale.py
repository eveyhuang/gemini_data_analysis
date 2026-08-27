"""
Run the Gemini linkography/network pipeline over a LIST of sessions (PROTOTYPE v1).

This is the scale-out step the deck's slide 50 gates on ("stay on two sessions before
scaling"). Those gates are met, so the same pipeline that produced NES_S3 and NES_S10
now runs unchanged over 10 sessions.

WHAT THIS IS NOT: it is not a new method. Every definition -- what counts as an idea
unit, a node, a move, a link, a speaker edge, a metric -- is imported from the existing
single-session modules and used as-is:

    gemini_linkography.py          load -> nodes/moves -> link_moves -> report + linkograph
    gemini_speaker_network.py      linkography json -> speaker network + sociogram
    gemini_linkography_metrics.py  linkography json -> the slide-48 metric block

This module only loops, logs, and validates. Running one session through it produces
the same linkography json as `gemini_linkography.py --session-dir ...` on that session.

WHAT IT ADDS (the two caveats the validation left open):
  (A) the 20-30% `nearest_prior` fallback -> `link_basis_log.csv` records, for EVERY
      link, which of the three rules resolved it and whether a real `links_to` target
      was extractable from Gemini's explanation. Per-session basis mixes go in the
      metrics table and the manifest, so no session's numbers are read without its
      groundedness alongside.
  (B) session variability -> `spotcheck_extends_sample.csv` draws N `extends_existing_idea`
      links per newly added session, with both ends' speaker, timestamp, chunk, evidence
      and explanation, plus blank `correct_yn` / `notes` columns to fill in by hand.

Usage:
    # everything: the 10 sessions in scale_sessions.json, 4 spot-check links each
    python analysis_v2/src/linkograph_network/gemini_scale.py

    # a different sample size, stratified across the three basis tiers
    python analysis_v2/src/linkograph_network/gemini_scale.py --spotcheck-n 5 --spotcheck-stratified

    # an ad-hoc subset, ignoring the config file
    python analysis_v2/src/linkograph_network/gemini_scale.py --session 2020_11_05_NES_S3 --session 2021_10_08_CMC_S11
"""

import argparse
import csv
import difflib
import json
import os
import random
import sys
import time
import traceback
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib                                                   # noqa: E402
matplotlib.use("Agg")

from gemini_linkography import (                                    # noqa: E402
    crosstab,
    draw_linkograph,
    extract_nodes_moves,
    linkography_payload,
    link_moves,
    load_utterances,
    match_to_liu,
    summarize,
)
from gemini_linkography_metrics import compute_metrics              # noqa: E402
from gemini_speaker_network import build_speaker_network, plot_sociogram  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))

SESSIONS_JSON = os.path.join(_HERE, "scale_sessions.json")
OUT_ROOT = "analysis_v2/results/tables/prototypes/scale10"
FIG_ROOT = "analysis_v2/figures/prototypes/scale10"
SEED = 20260826          # fixed so a re-run draws the SAME spot-check sample


# ---------------------------------------------------------------------------
# Session identity -- everything is derived from the session id, so the config
# file only ever has to list ids.
# ---------------------------------------------------------------------------
def resolve_session(session_id, base=_BASE):
    """'2021_10_08_CMC_S11' -> conf '2021CMC', label 'CMC_S11', and its output dir.
    Same id -> conf convention `gemini_speaker_network.main()` uses."""
    parts = session_id.split("_")
    if len(parts) < 5:
        raise ValueError(f"unrecognised session id: {session_id!r}")
    conf = parts[0] + parts[3]
    label = f"{parts[3]}_{parts[4]}"
    session_dir = os.path.join(base, "outputs", conf, f"output_{session_id}")
    return {"session": session_id, "conf": conf, "label": label, "session_dir": session_dir}


def load_session_list(path=SESSIONS_JSON):
    """Read the configured session list. Entries may be plain ids or objects with an
    `id` key; returns (ids, {id: entry-metadata})."""
    with open(path) as f:
        cfg = json.load(f)
    ids, meta = [], {}
    for entry in cfg["sessions"]:
        if isinstance(entry, str):
            entry = {"id": entry}
        ids.append(entry["id"])
        meta[entry["id"]] = entry
    return ids, meta


# ---------------------------------------------------------------------------
# Per-link logging -- caveat (A)
# ---------------------------------------------------------------------------
LINK_LOG_COLS = [
    "session", "conf", "label", "edge_id", "move_name",
    "basis", "links_to_extracted", "links_to_name",
    "from_idx", "from_speaker", "from_timestamp", "from_chunk",
    "to_idx", "to_speaker", "to_timestamp", "to_chunk",
    "is_self_link", "is_cross_chunk", "from_evidence", "from_explanation", "to_evidence",
]


def split_speaker_candidates(speakers, cutoff=0.85):
    """Pairs of speaker labels similar enough to plausibly be ONE person written two
    ways ('Tori Hoehler' / 'Tori Hoeler'). Deck slide 46 flags this as the node-identity
    risk: a split name becomes two network nodes and halves that person's links.

    Reported, never merged. Auto-merging would also merge genuinely different people
    with similar names (the Kris Hall / Kirsten Hall case), which is the worse error --
    so this is a flag for a human to adjudicate."""
    spk = sorted(speakers)
    return [[a, b] for i, a in enumerate(spk) for b in spk[i + 1:]
            if difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() >= cutoff]


def _explanation(utt, move_name):
    """The explanation prose of the move code that produced this edge (the text the
    linker read). An utterance can carry several codes; take the matching one."""
    for m in utt["moves"]:
        if m["name"] == move_name:
            return m["explanation"]
    return ""


def link_log_rows(ident, utts, edges):
    """One auditable row per resolved link: which rule fired, whether a real links_to
    was extractable, and both ends in full so a row can be checked without the JSON."""
    by_idx = {u["idx"]: u for u in utts}
    rows = []
    for i, e in enumerate(edges):
        src, dst = by_idx[e["from"]], by_idx[e["to"]]
        rows.append({
            "session": ident["session"], "conf": ident["conf"], "label": ident["label"],
            "edge_id": i, "move_name": e["move_name"],
            "basis": e["basis"],
            "links_to_extracted": bool(e.get("links_to_name")),
            "links_to_name": e.get("links_to_name") or "",
            "from_idx": e["from"], "from_speaker": src["speaker"],
            "from_timestamp": src["timestamp"], "from_chunk": src["chunk_file"],
            "to_idx": e["to"], "to_speaker": dst["speaker"],
            "to_timestamp": dst["timestamp"], "to_chunk": dst["chunk_file"],
            "is_self_link": src["speaker"] == dst["speaker"],
            "is_cross_chunk": src["chunk_file"] != dst["chunk_file"],
            "from_evidence": " ".join(m["evidence"] for m in src["moves"])[:400],
            "from_explanation": _explanation(src, e["move_name"])[:400],
            "to_evidence": " ".join(m["evidence"] for m in dst["moves"])[:400],
        })
    return rows


# ---------------------------------------------------------------------------
# Spot-check sampling -- caveat (B)
# ---------------------------------------------------------------------------
SPOTCHECK_COLS = [
    "session", "label", "edge_id", "basis", "links_to_extracted", "links_to_name",
    "from_speaker", "from_timestamp", "from_chunk", "from_evidence", "from_explanation",
    "to_speaker", "to_timestamp", "to_chunk", "to_evidence",
    "is_self_link", "is_cross_chunk", "correct_yn", "notes",
]


def sample_extends_links(rows, n, seed_key, stratified=False):
    """Draw n `extends_existing_idea` links for manual checking.

    Random by default (what was asked for). `stratified` spreads the draw across the
    three basis tiers first -- with n=4 and a ~20% named / ~30% nearest_prior mix a
    plain random draw can miss the fallback tier entirely, which is the tier the
    spot-check exists to interrogate.

    Deterministic: seeded per session, so a re-run produces the SAME rows to check.
    Returns (sample, note) where note records any shortfall -- never a silent cap."""
    pop = [r for r in rows if r["move_name"] == "extends_existing_idea"]
    rng = random.Random(seed_key)
    if len(pop) <= n:
        note = (f"sampled all {len(pop)} extends links (fewer than the requested {n})"
                if len(pop) < n else f"sampled all {len(pop)} extends links")
        return list(pop), note

    if not stratified:
        return rng.sample(pop, n), f"random {n} of {len(pop)} extends links"

    tiers = {}
    for r in pop:
        tiers.setdefault(r["basis"], []).append(r)
    order = [t for t in ("named_speaker", "on_table_idea", "nearest_prior") if t in tiers]
    picked, seen = [], set()
    while len(picked) < n:                      # round-robin one per tier until full
        progressed = False
        for t in order:
            if len(picked) >= n:
                break
            pool = [r for r in tiers[t] if r["edge_id"] not in seen]
            if not pool:
                continue
            r = rng.choice(pool)
            picked.append(r)
            seen.add(r["edge_id"])
            progressed = True
        if not progressed:
            break
    mix = ", ".join(f"{k}={v}" for k, v in sorted(Counter(r["basis"] for r in picked).items()))
    return picked, f"stratified {len(picked)} of {len(pop)} extends links ({mix})"


# ---------------------------------------------------------------------------
# One session
# ---------------------------------------------------------------------------
def run_pipeline_on_session(session_id, out_root=OUT_ROOT, fig_root=FIG_ROOT,
                            liu_json=None, base=_BASE, log=print):
    """Run the full single-session pipeline for one session and write its outputs.

    Identical in effect to running `gemini_linkography.py` then `gemini_speaker_network.py`
    on that session: same functions, same JSON structure, same figures. Returns a result
    dict with the payload, network stats, metrics row, link-log rows and file paths."""
    ident = resolve_session(session_id, base)
    if not os.path.isdir(ident["session_dir"]):
        raise FileNotFoundError(f"no annotations directory: {ident['session_dir']}")

    label, t0 = ident["label"], time.time()
    tbl_dir = os.path.join(base, out_root, "sessions", label)
    fig_dir = os.path.join(base, fig_root, label)
    os.makedirs(tbl_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # --- annotations -> nodes/moves -> links (the pipeline proper) ---
    bad_files = []
    utts = extract_nodes_moves(
        load_utterances(ident["session_dir"],
                        on_bad_file=lambda p, e: bad_files.append(os.path.basename(p))))
    if bad_files:
        log(f"  WARNING: skipped {len(bad_files)} unparseable chunk file(s): {bad_files}")
    edges, unresolved = link_moves(utts)
    log(f"  {len(utts)} utterances -> {sum(u['is_node'] or u['is_move'] for u in utts)} "
        f"idea-bearing -> {len(edges)} links ({len(unresolved)} unresolved)")

    # --- optional Liu cross-tab, exactly as the single-session script does it ---
    matches, tab, propose_rows, extend_rows = {}, Counter(), [], []
    if liu_json and os.path.exists(liu_json):
        matches, _ = match_to_liu(utts, liu_json)
        tab, propose_rows, extend_rows = crosstab(utts, matches)
        log(f"  Liu cross-tab: matched {len(matches)} utterances")

    # --- linkography json + report + linkograph png ---
    # `session_name` is the FULL id, exactly as `gemini_linkography.main()` derives it
    # (basename minus the 'output_' prefix) -- so the json header, the report heading and
    # both figure titles read the same as a single-session run. The short `label` is used
    # only for directory and file names.
    session_name = ident["session"]
    payload = linkography_payload(session_name, os.path.relpath(ident["session_dir"], base),
                                  utts, edges, unresolved, tab)
    link_json = os.path.join(tbl_dir, f"gemini_linkography_{label}.json")
    with open(link_json, "w") as f:
        json.dump(payload, f, indent=2)
    report_md = os.path.join(tbl_dir, f"gemini_linkography_{label}_report.md")
    with open(report_md, "w") as f:
        f.write(summarize(session_name, utts, edges, unresolved, matches, tab,
                          propose_rows, extend_rows))
    link_png = os.path.join(fig_dir, f"gemini_linkography_{label}.png")
    draw_linkograph(utts, edges, link_png, title=f"Gemini code-based linkograph: {session_name}")

    # --- speaker network json + sociogram ---
    G, stats = build_speaker_network(payload, ident["conf"])
    net_png = os.path.join(fig_dir, f"gemini_speaker_network_{label}.png")
    plot_sociogram(G, stats, net_png)
    net_json = os.path.join(tbl_dir, f"gemini_speaker_network_{label}.json")
    with open(net_json, "w") as f:
        json.dump(stats, f, indent=2)
    log(f"  network: {stats['n_speakers']} speakers, {stats['n_edges']} edges, "
        f"SLI={stats['self_link_index']}")
    if stats["unmatched_speakers"]:
        log(f"  speakers not on any team in the roster (drawn as 'none'): "
            f"{stats['unmatched_speakers']}")
    splits = split_speaker_candidates(stats["outcome"])
    for a, b in splits:
        log(f"  CHECK NAME: '{a}' and '{b}' are separate nodes but may be one person")

    # --- metrics row + logging ---
    metrics = compute_metrics(payload)
    log_rows = link_log_rows(ident, utts, edges)
    n_links = len(edges) or 1
    extends = [r for r in log_rows if r["move_name"] == "extends_existing_idea"]
    ext_mix = Counter(r["basis"] for r in extends)
    log(f"  extends links: {len(extends)} "
        f"(named={ext_mix['named_speaker']}, on_table={ext_mix['on_table_idea']}, "
        f"nearest_prior={ext_mix['nearest_prior']}); "
        f"links_to extracted for {sum(1 for r in extends if r['links_to_extracted'])}")

    return {
        "ident": ident,
        "payload": payload,
        "stats": stats,
        "metrics": metrics,
        "log_rows": log_rows,
        "bad_files": bad_files,
        "seconds": round(time.time() - t0, 2),
        "session_name": session_name,
        "paths": {"linkography_json": os.path.relpath(link_json, base),
                  "report_md": os.path.relpath(report_md, base),
                  "linkograph_png": os.path.relpath(link_png, base),
                  "network_json": os.path.relpath(net_json, base),
                  "network_png": os.path.relpath(net_png, base)},
        "summary": {
            "n_utterances": len(utts),
            "n_idea_utterances": len(payload["utterances"]),
            "n_links": len(edges),
            "n_unresolved_moves": len(unresolved),
            "n_extends": len(extends),
            "basis_named_speaker": metrics["basis_named_speaker"],
            "basis_on_table_idea": metrics["basis_on_table_idea"],
            "basis_nearest_prior": metrics["basis_nearest_prior"],
            "pct_nearest_prior": round(metrics["basis_nearest_prior"] / n_links, 3),
            "pct_links_to_extracted": round(
                sum(1 for r in log_rows if r["links_to_extracted"]) / n_links, 3),
            "pct_cross_chunk": round(sum(1 for r in log_rows if r["is_cross_chunk"]) / n_links, 3),
            "n_bad_files": len(bad_files),
            "split_speaker_candidates": splits,
        },
    }


# ---------------------------------------------------------------------------
# The batch
# ---------------------------------------------------------------------------
def run_pipeline_on_sessions(session_ids, out_root=OUT_ROOT, fig_root=FIG_ROOT,
                             spotcheck_n=4, spotcheck_stratified=False,
                             skip_spotcheck=(), liu_json_for=None, seed=SEED, base=_BASE):
    """Run the pipeline over every session in `session_ids`.

    session_ids           list of `outputs/<conf>/output_<id>` session ids
    out_root/fig_root     repo-relative roots for tables and figures
    spotcheck_n           extends links to sample per session for manual checking (0 = off)
    spotcheck_stratified  spread the sample across the three basis tiers
    skip_spotcheck        session ids to exclude from the sample (the focal pair by default
                          from the CLI -- their links were already read end to end)
    liu_json_for          optional {session_id: path} for the Liu cross-tab
    seed                  fixes the spot-check draw, so a re-run yields the same rows

    A session that fails is recorded in the manifest and the batch continues; the run
    ends with a non-zero exit only through the CLI. Returns the manifest dict."""
    liu_json_for = liu_json_for or {}
    tbl_root = os.path.join(base, out_root)
    log_dir = os.path.join(tbl_root, "logs")
    os.makedirs(log_dir, exist_ok=True)

    results, failures, all_link_rows, all_spot_rows, metric_rows = [], [], [], [], []
    _, meta = ({}, {})
    try:
        _, meta = load_session_list()
    except (OSError, ValueError, KeyError):
        meta = {}

    for i, sid in enumerate(session_ids, 1):
        label = resolve_session(sid, base)["label"]
        lines = []

        def log(msg, _lines=lines):
            print(msg)
            _lines.append(str(msg))

        log(f"[{i}/{len(session_ids)}] {sid}  ({label})")
        try:
            res = run_pipeline_on_session(sid, out_root, fig_root,
                                          liu_json=liu_json_for.get(sid), base=base, log=log)
        except Exception as exc:                                # noqa: BLE001
            log(f"  FAILED: {type(exc).__name__}: {exc}")
            log(traceback.format_exc())
            failures.append({"session": sid, "error": f"{type(exc).__name__}: {exc}"})
        else:
            results.append(res)
            all_link_rows.extend(res["log_rows"])
            entry = meta.get(sid, {})
            row = {"label": res["ident"]["label"], "conf": res["ident"]["conf"],
                   "role": entry.get("role", ""), "outcome_group": entry.get("outcome", "")}
            row.update(res["metrics"])
            row.update({k: v for k, v in res["summary"].items()
                        if k not in row and not isinstance(v, list)})
            row["split_speaker_candidates"] = "; ".join(
                f"{a} / {b}" for a, b in res["summary"]["split_speaker_candidates"])
            metric_rows.append(row)

            if spotcheck_n and sid not in skip_spotcheck:
                sample, note = sample_extends_links(res["log_rows"], spotcheck_n,
                                                    seed_key=f"{seed}:{sid}",
                                                    stratified=spotcheck_stratified)
                log(f"  spot-check: {note}")
                res["spotcheck_note"] = note
                for r in sample:
                    all_spot_rows.append({**{c: r.get(c, "") for c in SPOTCHECK_COLS},
                                          "correct_yn": "", "notes": ""})
            elif spotcheck_n:
                log("  spot-check: skipped (focal session, already read end to end)")
            log(f"  done in {res['seconds']}s")
        with open(os.path.join(log_dir, f"{label}.log"), "w") as f:
            f.write("\n".join(lines) + "\n")

    # --- combined tables ---
    if metric_rows:
        cols = list(dict.fromkeys(k for r in metric_rows for k in r))
        _write_csv(os.path.join(tbl_root, "metrics_gemini_linkography_scaled.csv"),
                   cols, metric_rows)
    if all_link_rows:
        _write_csv(os.path.join(tbl_root, "link_basis_log.csv"), LINK_LOG_COLS, all_link_rows)
    if all_spot_rows:
        _write_csv(os.path.join(tbl_root, "spotcheck_extends_sample.csv"),
                   SPOTCHECK_COLS, all_spot_rows)

    manifest = {
        "n_requested": len(session_ids),
        "n_succeeded": len(results),
        "n_failed": len(failures),
        "seed": seed,
        "spotcheck_n": spotcheck_n,
        "spotcheck_stratified": spotcheck_stratified,
        "spotcheck_skipped": sorted(skip_spotcheck),
        "out_root": out_root,
        "fig_root": fig_root,
        "sessions": [{"session": r["ident"]["session"], "label": r["ident"]["label"],
                      "conf": r["ident"]["conf"], "seconds": r["seconds"],
                      "spotcheck": r.get("spotcheck_note", "skipped"),
                      **r["summary"], "paths": r["paths"]} for r in results],
        "failures": failures,
    }
    with open(os.path.join(tbl_root, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def _write_csv(path, cols, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions-json", default=SESSIONS_JSON,
                    help="config listing the sessions to run (default: scale_sessions.json)")
    ap.add_argument("--session", action="append", dest="sessions",
                    help="run this session id instead of the config list; repeatable")
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--fig-root", default=FIG_ROOT)
    ap.add_argument("--spotcheck-n", type=int, default=4,
                    help="extends links sampled per session for manual checking (0 = off)")
    ap.add_argument("--spotcheck-stratified", action="store_true",
                    help="spread the sample across named / on_table / nearest_prior")
    ap.add_argument("--spotcheck-all", action="store_true",
                    help="also sample the focal sessions (skipped by default)")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    if args.sessions:
        ids, meta = args.sessions, {}
    else:
        ids, meta = load_session_list(args.sessions_json)
    skip = () if args.spotcheck_all else tuple(
        s for s in ids if meta.get(s, {}).get("role") == "focal")

    manifest = run_pipeline_on_sessions(
        ids, out_root=args.out_root, fig_root=args.fig_root,
        spotcheck_n=args.spotcheck_n, spotcheck_stratified=args.spotcheck_stratified,
        skip_spotcheck=skip, seed=args.seed)

    print(f"\n{manifest['n_succeeded']}/{manifest['n_requested']} sessions completed.")
    if manifest["failures"]:
        print("FAILURES:")
        for f in manifest["failures"]:
            print(f"  {f['session']}: {f['error']}")
    ok = [s for s in manifest["sessions"]]
    if ok:
        tot = sum(s["n_links"] for s in ok)
        near = sum(s["basis_nearest_prior"] for s in ok)
        splits = [(s["label"], p) for s in ok for p in s["split_speaker_candidates"]]
        if splits:
            print("Speaker labels to adjudicate (one person, two nodes?):")
            for label, (a, b) in splits:
                print(f"  {label}: '{a}' / '{b}'")
        print(f"{tot} links across {len(ok)} sessions; "
              f"nearest_prior fallback {near}/{tot} ({100 * near / tot:.0f}%) — "
              f"per-session range "
              f"{min(s['pct_nearest_prior'] for s in ok):.0%}–{max(s['pct_nearest_prior'] for s in ok):.0%}")
    return 1 if manifest["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
