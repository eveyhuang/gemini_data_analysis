"""
Code-based linkography from our Gemini behavioral annotations (PROTOTYPE v1).

This is the counterpart to `liu2026_linkography_pipeline.py`. Liu runs its OWN
Claude judgments on the raw transcript and never reads our codes. This script does
the opposite: it builds the linkograph *directly from the Gemini idea-move codes*
we already have -- no topic filter, no new LLM calls -- and then answers the three
questions Evey raised:

  Q1. Do `proposes_new_idea` / `extends_existing_idea` count as a NODE (new idea)
      or a MOVE (edge) here? -> the MOVE_MAP below makes that explicit and
      sign-off-ready.
  Q2. How much do the Gemini node/move labels agree with Liu's N / S / M? -> the
      cross-tab against a Liu output JSON (`--liu-json`).
  Q3. `extends_existing_idea` records the move TYPE but not WHICH idea it extends
      (no structured `links_to`). How do we recover the target? -> the linking step
      (Part B), seeded by the code's `explanation` prose (which often names the
      person/idea being built on) plus `session_state.ideas_currently_on_table`.

Everything here is deterministic and auditable -- every node traces to a specific
code, every edge records HOW its target was resolved. No topic is generated and
nothing is dropped for being "off topic"; an utterance is a node/edge iff it carries
an idea-move code.

PROTOTYPE STATUS: the MOVE_MAP (which codes are nodes vs moves) is a *proposal* for
Evey to sign off on -- borderline cases are flagged. The linker is a transparent
heuristic (v1), not a validated method; an LLM-guided variant is a later option.

Usage:
    python analysis_v2/src/linkograph_network/gemini_linkography.py \
        --session-dir outputs/2020NES/output_2020_11_05_NES_S3 \
        --liu-json analysis_v2/results/tables/prototypes/liu2026_output_2020_11_05_NES_S3.json \
        --out-prefix analysis_v2/results/tables/prototypes/session_comparison/gemini_linkography_NES_S3
"""

import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Color families for the move types, so the linkograph reads at a glance.
_MOVE_COLOR = {
    "extends_existing_idea": "#4C72B0", "combines_ideas": "#4C72B0",
    "synthesizes_contributions": "#4C72B0", "connects_methods": "#4C72B0",
    "returns_to_earlier_idea": "#4C72B0",                       # additive/building -> blue
    "critiques_or_challenges": "#C44E52", "raises_concern": "#C44E52",
    "redirects_idea": "#C44E52", "resolves_contradiction": "#C44E52",  # structural/challenge -> red
}

# ---------------------------------------------------------------------------
# The idea-move taxonomy: which codes are IDEA WORK, and for each, is it a NODE
# (introduces a new idea) or a MOVE/edge (operates on an idea already on the table)?
#
# This is the Q1 answer, made explicit and reviewable. `proposes_new_idea` is the
# only unambiguous NODE; `extends_existing_idea` and the rest are MOVES because by
# definition they act on something already said. Codes marked BORDERLINE need Evey's
# call -- e.g. `frames_shared_problem` arguably seeds a node, `returns_to_earlier_idea`
# re-activates an OLD node rather than making a fresh one.
# ---------------------------------------------------------------------------
NODE = "node"   # introduces a new idea  (Liu 'N' analogue)
MOVE = "move"   # builds on / operates on an earlier idea  (Liu 'S'/'M' analogue)

MOVE_MAP = {
    # --- unambiguous NODE ---
    "proposes_new_idea":        NODE,
    # --- unambiguous MOVES (act on an existing idea) ---
    "extends_existing_idea":    MOVE,
    "combines_ideas":           MOVE,
    "synthesizes_contributions": MOVE,
    "connects_methods":         MOVE,
    "critiques_or_challenges":  MOVE,
    "raises_concern":           MOVE,
    "resolves_contradiction":   MOVE,
    "redirects_idea":           MOVE,
    # --- BORDERLINE (flagged for sign-off) ---
    "frames_shared_problem":    NODE,   # BORDERLINE: seeds a problem framing -> treated as a node
    "returns_to_earlier_idea":  MOVE,   # BORDERLINE: re-activates an OLD node -> treated as a move (edge back)
}
BORDERLINE = {"frames_shared_problem", "returns_to_earlier_idea"}

# Idea-move codes can appear two ways in the files (a known storage inconsistency):
#   (a) code_name == "Idea Management", subcode == "extends_existing_idea"
#   (b) code_name == "extends_existing_idea", subcode == None   (flattened)
# Plus a few move-types live under OTHER code_names (synthesizes under Integration
# Practices, critiques under Evaluation Practices, etc.). We treat any code whose
# subcode OR code_name is a key of MOVE_MAP as an idea-move. This normalization is
# the fix the handoff flagged as "needed before trusting counts".
_MOVE_KEYS = set(MOVE_MAP)

_STOP = frozenset("""a an the of and or to in on for with without by from as at is are was were be been being
this that these those it its their his her our your my we they i you he she them us here there what which who
whom how when where why not no yes than then into onto over under between among during within some any all more
most much many very just about like really sort kind thing things idea ideas way ways um uh you're i'm""".split())


# ---------------------------------------------------------------------------
# Loading & ordering
# ---------------------------------------------------------------------------
# A session's chunk files live in one of two layouts:
#   (a) nested  -- one subdirectory per recording (2020NES only):
#                  <session>/<recording>/<recording>_chunkN.json
#   (b) flat    -- every chunk directly in the session dir (all other conferences):
#                  <session>/<recording>_chunkN.json
# Either way the *recording* is the unit that has to be ordered first (a session can
# span several recordings), then chunks within it. Recording names embed the start
# timestamp in one of two formats, so we parse it rather than sorting the raw string.
_RE_MDY = re.compile(r"(\d{1,2})_(\d{1,2})_(\d{4})_(\d{1,2})_(\d{2})_(\d{2})_?(AM|PM)", re.I)
_RE_YMD = re.compile(r"(\d{4})_(\d{1,2})_(\d{1,2})_(\d{1,2})_(\d{2})_(\d{2})")


def _recording_key(path, session_dir):
    """The recording a chunk file belongs to: its subdirectory name (nested layout)
    or its basename with the _chunkN suffix stripped (flat layout)."""
    d = os.path.dirname(os.path.abspath(path))
    if d != os.path.abspath(session_dir):
        return os.path.basename(d)
    return re.sub(r"_chunk\d+$", "", os.path.splitext(os.path.basename(path))[0])


def _recording_start(key):
    """Recording start as a sortable 'YYYYMMDDhhmmss' string, or None if the name
    carries no parseable timestamp (then we fall back to sorting the name itself)."""
    m = _RE_MDY.search(key)
    if m:
        mo, dy, yr, hh, mi, ss = (int(m.group(i)) for i in range(1, 7))
        ap = m.group(7).upper()
        hh = 0 if (ap == "AM" and hh == 12) else (hh + 12 if (ap == "PM" and hh != 12) else hh)
        return f"{yr:04d}{mo:02d}{dy:02d}{hh:02d}{mi:02d}{ss:02d}"
    m = _RE_YMD.search(key)
    if m:
        yr, mo, dy, hh, mi, ss = (int(m.group(i)) for i in range(1, 7))
        return f"{yr:04d}{mo:02d}{dy:02d}{hh:02d}{mi:02d}{ss:02d}"
    return None


def _chunk_sort_key(path, session_dir=None):
    """Order a session's json files: recording start time first (name-sorted when the
    name has no timestamp), then chunk number within the recording (chunk1 < chunk2 <
    ... < chunk10 via natural sort; a chunk-less single file sorts first)."""
    key = _recording_key(path, session_dir if session_dir is not None
                         else os.path.dirname(os.path.dirname(os.path.abspath(path))))
    start = _recording_start(key)
    m = re.search(r"chunk(\d+)", os.path.basename(path))
    # ("0", start) sorts every timestamped recording ahead of the unparseable ones.
    return (("0", start) if start else ("1", ""), key, int(m.group(1)) if m else 0)


def session_files(session_dir):
    """Every chunk json for a session, in reading order, across both layouts."""
    found = set(glob.glob(os.path.join(session_dir, "*", "*.json")))
    found |= set(glob.glob(os.path.join(session_dir, "*.json")))
    return sorted(found, key=lambda p: _chunk_sort_key(p, session_dir))


def load_utterances(session_dir, on_bad_file=None):
    """Flatten every chunk's utterance_annotations into one ordered list with a
    global index, carrying each utterance's chunk-level ideas_currently_on_table.

    A handful of raw Gemini dumps are truncated / fence-wrapped and won't parse (they
    carry an ATTN_ prefix). Rather than killing a whole session's run, an unparseable
    chunk is skipped and reported through `on_bad_file(path, exception)` if supplied;
    with no callback the exception propagates, preserving the original behaviour."""
    files = session_files(session_dir)
    utts = []
    for f in files:
        try:
            d = json.load(open(f))
        except (ValueError, OSError) as exc:
            if on_bad_file is None:
                raise
            on_bad_file(f, exc)
            continue
        on_table = (d.get("session_state") or {}).get("ideas_currently_on_table") or []
        for u in d.get("utterance_annotations", []):
            utts.append({
                "speaker": (u.get("speaker") or "").strip(),
                "timestamp": u.get("timestamp"),
                "codes": u.get("codes", []),
                "idea_quality": _max_idea_quality(u.get("codes", [])),
                "ideas_on_table": on_table,
                "chunk_file": os.path.basename(f),
            })
    for i, u in enumerate(utts):
        u["idx"] = i
    return utts


def _max_idea_quality(codes):
    qs = [c.get("idea_quality") for c in codes if isinstance(c.get("idea_quality"), int)]
    return max(qs) if qs else None


def idea_moves(utt):
    """Return the idea-move codes on an utterance, normalized to
    {name, role (node/move), evidence, explanation}. Handles both storage forms."""
    out = []
    for c in utt["codes"]:
        cn, sc = c.get("code_name"), c.get("subcode")
        name = sc if sc in _MOVE_KEYS else (cn if cn in _MOVE_KEYS else None)
        if name is None:
            continue
        out.append({
            "name": name,
            "role": MOVE_MAP[name],
            "evidence": c.get("evidence", "") or "",
            "explanation": c.get("explanation", "") or "",
        })
    return out


# ---------------------------------------------------------------------------
# Part A -- nodes & moves from the codes
# ---------------------------------------------------------------------------
def extract_nodes_moves(utts):
    """Label each utterance NODE / MOVE / (both) / none from its idea-move codes.
    An utterance is a NODE if it carries any node-role code, a MOVE if it carries
    any move-role code; it can be both (e.g. proposes + extends)."""
    for u in utts:
        moves = idea_moves(u)
        u["moves"] = moves
        u["is_node"] = any(m["role"] == NODE for m in moves)
        u["is_move"] = any(m["role"] == MOVE for m in moves)
        u["move_names"] = [m["name"] for m in moves]
    return utts


# ---------------------------------------------------------------------------
# Part B -- the linking step (recover the target `extends_existing_idea` omits)
# ---------------------------------------------------------------------------
def _tokens(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP and len(t) > 2]


def _first_names(speakers):
    """Map first-name (lower) -> canonical speaker, for detecting 'builds on Phil's..'.
    Iterates a SORTED speaker list: when two participants share a first name, which one
    wins has to be a fixed choice, not set-iteration order (which varies per process
    under hash randomization)."""
    fn = {}
    for s in sorted(speakers):
        if s:
            fn.setdefault(s.split()[0].lower(), s)
    return fn


def link_moves(utts):
    """For every MOVE utterance, resolve which earlier idea-bearing utterance it
    targets. Candidate targets = earlier utterances carrying any idea-move code.
    Resolution priority (each edge records which basis fired, so it's auditable):

      1. named_speaker  -- the code's `explanation` names another participant
                           -> link to that speaker's most recent prior idea utterance.
      2. on_table_idea  -- the utterance's text best matches one of the chunk's
                           `ideas_currently_on_table` -> link to the earliest prior
                           idea utterance that introduced that idea (by token overlap).
      3. nearest_prior  -- fallback: the most recent prior idea utterance.

    Returns a list of edges {from, to, move_name, basis, links_to_name}. No edge for a
    MOVE that has no prior idea utterance to attach to (recorded as unresolved).

    `links_to_name` is the `links_to` field Gemini never wrote: the participant the
    explanation actually names, when it names one. It is RECORDED, not used -- target
    selection is unchanged -- so the share of moves with a real extracted target can be
    reported per session (the deck's slide-54 question) independently of which rule
    fired. It can be non-null on a non-named_speaker edge: the explanation named
    someone who has no prior idea utterance to link back to."""
    speakers = sorted({u["speaker"] for u in utts if u["speaker"]})
    first_names = _first_names(speakers)
    edges, unresolved = [], []

    for u in utts:
        if not u["is_move"]:
            continue
        prior = [p for p in utts if p["idx"] < u["idx"] and (p["is_node"] or p["is_move"])]
        if not prior:
            for m in u["moves"]:
                if m["role"] == MOVE:
                    unresolved.append({"from": u["idx"], "move_name": m["name"], "reason": "no_prior_idea"})
            continue

        for m in u["moves"]:
            if m["role"] != MOVE:
                continue
            target, basis, named = _resolve_target(u, m, prior, first_names)
            if target is None:
                unresolved.append({"from": u["idx"], "move_name": m["name"], "reason": "no_target",
                                   "links_to_name": named[0] if named else None})
            else:
                edges.append({"from": u["idx"], "to": target, "move_name": m["name"], "basis": basis,
                              "links_to_name": named[0] if named else None})
    return edges, unresolved


def _resolve_target(u, move, prior, first_names):
    expl = move["explanation"].lower()

    # (1) named_speaker: explanation references another participant by first name.
    # Ordered by where the name appears in the explanation ("Jenny builds on Phil's
    # idea" -> Phil, not Jenny, once the speaker herself is excluded), so the recorded
    # links_to_name is the first *other* participant the prose mentions. The scan runs
    # over a sorted key list: a dict built from a set has no stable order, and an
    # explanation naming two people would otherwise pick a different one each run.
    hits = []
    for fn in sorted(first_names):
        if first_names[fn] == u["speaker"]:
            continue
        m = re.search(rf"\b{re.escape(fn)}\b", expl)
        if m:
            hits.append((m.start(), first_names[fn]))
    named = [n for _, n in sorted(hits)]
    if named:
        for p in reversed(prior):                      # most recent first
            if p["speaker"] in named:
                return p["idx"], "named_speaker", named

    # (2) on_table_idea: match this utterance's text to a live named idea, then to the
    #     earliest prior idea utterance that best matches that same idea string.
    my_tokens = set(_tokens(move["evidence"] + " " + move["explanation"]))
    best_idea, best_overlap = None, 0
    for idea in u["ideas_on_table"]:
        it = set(_tokens(idea))
        ov = len(my_tokens & it)
        if ov > best_overlap:
            best_overlap, best_idea = ov, it
    if best_idea and best_overlap >= 1:
        scored = []
        for p in prior:
            pt = set(_tokens(" ".join(mm["evidence"] + " " + mm["explanation"] for mm in p["moves"])))
            scored.append((len(pt & best_idea), -p["idx"], p["idx"]))   # tie-break: earliest
        scored.sort(reverse=True)
        if scored and scored[0][0] >= 1:
            return scored[0][2], "on_table_idea", named

    # (3) nearest_prior fallback.
    return prior[-1]["idx"], "nearest_prior", named


# ---------------------------------------------------------------------------
# Part C -- agreement / cross-tab vs a Liu output
# ---------------------------------------------------------------------------
def _spk_key(s):
    return (s or "").strip().split()[0].lower() if s else ""


def _spk_match(a, b):
    """Tolerant speaker match across the two pipelines' name spellings. Matches on
    first name with prefix tolerance ('phil' vs 'phill') OR on last name, so
    'Phil Milner' (Gemini) and 'Phill Milner' (Liu) unify. Guards against empty."""
    ap, bp = (a or "").lower().split(), (b or "").lower().split()
    if not ap or not bp:
        return False
    fa, fb = ap[0], bp[0]
    first = fa == fb or (len(fa) >= 3 and len(fb) >= 3 and (fa.startswith(fb) or fb.startswith(fa)))
    last = len(ap) > 1 and len(bp) > 1 and ap[-1] == bp[-1]
    return first or last


def match_to_liu(utts, liu_path, min_overlap_frac=0.5, min_tokens=3):
    """Match each Gemini utterance to a Liu message by speaker (first name) +
    evidence/text token overlap. Many Gemini utterances -> one Liu turn is expected
    (Gemini segments finer). Returns {gemini_idx: liu_message} for matched ones."""
    liu = json.load(open(liu_path))
    liu_msgs = liu["messages"]

    matches = {}
    for u in utts:
        # Use the utterance's idea-move evidence if present, else all its evidence.
        ev = " ".join(m["evidence"] for m in u["moves"]) or \
             " ".join(c.get("evidence", "") for c in u["codes"])
        gtok = set(_tokens(ev))
        if len(gtok) < min_tokens:
            continue
        best, best_frac = None, 0.0
        for m in liu_msgs:
            if not _spk_match(u["speaker"], m["speaker"]):
                continue
            ltok = set(_tokens(m["text"]))
            if not ltok:
                continue
            frac = len(gtok & ltok) / len(gtok)
            if frac > best_frac:
                best_frac, best = frac, m
        if best is not None and best_frac >= min_overlap_frac:
            matches[u["idx"]] = best
    return matches, liu


def crosstab(utts, matches):
    """Contingency of Gemini label vs Liu code on matched utterances, plus the two
    headline agreements Evey asked for: propose<->N and extends/moves<->S or M."""
    def gemini_label(u):
        if u["is_node"] and u["is_move"]:
            return "node+move"
        if u["is_node"]:
            return "node"
        if u["is_move"]:
            return "move"
        return "none"

    def liu_label(m):
        codes = set(m.get("codes", []))
        if not m["relevant"]:
            return "dropped"        # Liu filtered it out (topic/relevance)
        if "N" in codes and (("S" in codes) or ("M" in codes)):
            return "N+link"
        if "N" in codes:
            return "N"
        if ("S" in codes) or ("M" in codes):
            return "link(S/M)"
        return "relevant_nocode"

    tab = Counter()
    propose_rows, extend_rows = [], []
    for u in utts:
        if u["idx"] not in matches:
            continue
        g, l = gemini_label(u), liu_label(matches[u["idx"]])
        tab[(g, l)] += 1
        if "proposes_new_idea" in u["move_names"]:
            liu_is_N = matches[u["idx"]]["relevant"] and "N" in set(matches[u["idx"]].get("codes", []))
            propose_rows.append(liu_is_N)
        if "extends_existing_idea" in u["move_names"]:
            c = set(matches[u["idx"]].get("codes", []))
            liu_is_link = matches[u["idx"]]["relevant"] and (("S" in c) or ("M" in c))
            extend_rows.append(liu_is_link)
    return tab, propose_rows, extend_rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def summarize(session_name, utts, edges, unresolved, matches, tab, propose_rows, extend_rows):
    n_nodes = sum(u["is_node"] for u in utts)
    n_moves = sum(u["is_move"] for u in utts)
    move_type_counts = Counter(n for u in utts for n in u["move_names"])
    basis_counts = Counter(e["basis"] for e in edges)

    lines = []
    lines.append(f"# Gemini code-based linkography -- {session_name}\n")
    lines.append(f"- Utterances (total): **{len(utts)}**")
    lines.append(f"- Idea-bearing utterances: **{sum(u['is_node'] or u['is_move'] for u in utts)}** "
                 f"(nodes={n_nodes}, moves={n_moves}; an utterance can be both)")
    lines.append(f"- Edges resolved by the linker: **{len(edges)}**  | unresolved moves: {len(unresolved)}")
    lines.append("")
    lines.append("## Q1 -- node vs move mapping (counts this session)")
    lines.append("| code | role | count |")
    lines.append("|---|---|---|")
    for name, cnt in move_type_counts.most_common():
        role = MOVE_MAP.get(name, "?")
        flag = " *(borderline)*" if name in BORDERLINE else ""
        lines.append(f"| `{name}` | {role}{flag} | {cnt} |")
    lines.append("")
    lines.append("## Q3 -- how the linker recovered each move's target (basis)")
    for b, c in basis_counts.most_common():
        lines.append(f"- **{b}**: {c}")
    lines.append("")

    if matches:
        lines.append(f"## Q2 -- agreement vs Liu (matched {len(matches)} of {len(utts)} utterances)")
        lines.append("Segmentation differs (Gemini fine, Liu coarse), so only utterances whose "
                     "evidence overlaps a Liu turn are matchable.\n")
        lines.append("**Contingency (Gemini rows x Liu cols):**\n")
        g_labels = ["node", "move", "node+move", "none"]
        l_labels = ["N", "link(S/M)", "N+link", "relevant_nocode", "dropped"]
        header = "| Gemini \\\\ Liu | " + " | ".join(l_labels) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(l_labels) + 1))
        for g in g_labels:
            row = [str(tab.get((g, l), 0)) for l in l_labels]
            lines.append(f"| **{g}** | " + " | ".join(row) + " |")
        lines.append("")
        total_propose = sum("proposes_new_idea" in u["move_names"] for u in utts)
        total_extend = sum("extends_existing_idea" in u["move_names"] for u in utts)
        lines.append("")
        lines.append("Agreement is measured only on the matchable slice (an utterance whose "
                     "evidence overlaps a Liu turn). Denominators below are: agreed / matched "
                     "(of TOTAL in this session) -- so the matched count is itself a subset of "
                     "the total, because Gemini segments finer and Liu dropped some turns.\n")
        if propose_rows:
            lines.append(f"- **proposes_new_idea -> Liu N:** {sum(propose_rows)}/{len(propose_rows)} "
                         f"matched agreed  (matched {len(propose_rows)} of {total_propose} total proposes)")
        if extend_rows:
            lines.append(f"- **extends_existing_idea -> Liu S/M:** {sum(extend_rows)}/{len(extend_rows)} "
                         f"matched agreed  (matched {len(extend_rows)} of {total_extend} total extends)")
    else:
        lines.append("## Q2 -- agreement vs Liu\n_No Liu output supplied (or none matched)._")
    lines.append("")
    return "\n".join(lines)


def linkography_payload(session_name, session_dir, utts, edges, unresolved, tab):
    """The on-disk linkography record for one session. Factored out of main() so the
    batch runner (`gemini_scale.py`) writes byte-for-byte the same structure as a
    single-session run -- there is one definition of this file, not two."""
    return {
        "session": session_name,
        "session_dir": session_dir,
        "move_map": MOVE_MAP,
        "utterances": [
            {"idx": u["idx"], "speaker": u["speaker"], "timestamp": u["timestamp"],
             "is_node": u["is_node"], "is_move": u["is_move"],
             "move_names": u["move_names"], "idea_quality": u["idea_quality"],
             "evidence": " ".join(m["evidence"] for m in u["moves"])[:400]}
            for u in utts if u["is_node"] or u["is_move"]
        ],
        "edges": edges,
        "unresolved_moves": unresolved,
        "liu_crosstab": {f"{g}|{l}": c for (g, l), c in tab.items()},
    }


def draw_linkograph(utts, edges, out_path, title):
    """Timeline of idea-bearing utterances (nodes/moves), arcs = resolved edges.
    Node = star (proposes/frames), move = dot. Arc color = additive (blue) vs
    structural/challenge (red). Critical utterances (most-linked) get a black ring."""
    idea = [u for u in utts if u["is_node"] or u["is_move"]]
    pos = {u["idx"]: i for i, u in enumerate(idea)}

    deg = Counter()
    for e in edges:
        if e["from"] in pos and e["to"] in pos:
            deg[e["from"]] += 1
            deg[e["to"]] += 1
    # Goldschmidt <=12% critical guideline, applied to this session's idea utterances.
    thresh = 0
    for t in sorted(set(deg.values()), reverse=True):
        if sum(1 for c in deg.values() if c >= t) / max(len(idea), 1) <= 0.12:
            thresh = t
            break
    critical = {i for i, c in deg.items() if c >= thresh and thresh > 0}

    fig, ax = plt.subplots(figsize=(16, 8))
    for e in edges:
        if e["from"] not in pos or e["to"] not in pos:
            continue
        i, j = pos[e["to"]], pos[e["from"]]
        apex_x, apex_y = (i + j) / 2, -((abs(j - i)) ** 0.6)
        ax.plot([i, apex_x, j], [0, apex_y, 0],
                color=_MOVE_COLOR.get(e["move_name"], "#8899aa"), alpha=0.5, linewidth=1.3)
    for u in idea:
        p = pos[u["idx"]]
        is_crit = u["idx"] in critical
        ax.scatter(p, 0, s=260 if is_crit else 110,
                   marker="*" if u["is_node"] else "o",
                   color="#DD8452" if u["is_node"] else "#55A868",
                   zorder=3, edgecolors="black" if is_crit else "none",
                   linewidths=1.2 if is_crit else 0)
        if is_crit:
            ax.annotate(f'{u["idx"]} ({deg[u["idx"]]})', (p, 0), textcoords="offset points",
                        xytext=(0, 12), ha="center", fontsize=7, fontweight="bold")

    legend = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#DD8452", markersize=14,
               label="Node (proposes / frames new idea)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#55A868", markersize=10,
               label="Move (extends / synthesizes / critiques ...)"),
        Line2D([0], [0], color="#4C72B0", lw=2, label="Additive edge (extend/combine/synthesize)"),
        Line2D([0], [0], color="#C44E52", lw=2, label="Structural edge (critique/redirect/concern)"),
    ]
    if thresh > 0:
        legend.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="#55A868",
                             markeredgecolor="black", markersize=10,
                             label=f"Critical (>= {thresh} links, Goldschmidt <=12%)"))
    ax.legend(handles=legend, loc="upper right", fontsize=8)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Idea-bearing-utterance sequence (non-idea utterances omitted)")
    ax.set_yticks([])
    ax.spines[["left", "top", "right"]].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session-dir", required=True)
    ap.add_argument("--liu-json", default=None)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    session_name = os.path.basename(args.session_dir.rstrip("/")).replace("output_", "")
    utts = extract_nodes_moves(load_utterances(args.session_dir))
    edges, unresolved = link_moves(utts)

    matches, tab, propose_rows, extend_rows = {}, Counter(), [], []
    if args.liu_json and os.path.exists(args.liu_json):
        matches, _ = match_to_liu(utts, args.liu_json)
        tab, propose_rows, extend_rows = crosstab(utts, matches)

    report = summarize(session_name, utts, edges, unresolved, matches, tab, propose_rows, extend_rows)

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    with open(args.out_prefix + ".json", "w") as f:
        json.dump(linkography_payload(session_name, args.session_dir, utts, edges,
                                      unresolved, tab), f, indent=2)
    with open(args.out_prefix + "_report.md", "w") as f:
        f.write(report)

    png_path = args.out_prefix.replace("results/tables", "figures") + ".png"
    draw_linkograph(utts, edges, png_path,
                    title=f"Gemini code-based linkograph: {session_name}")

    print(report)
    print(f"\nSaved: {args.out_prefix}.json")
    print(f"Saved: {args.out_prefix}_report.md")


if __name__ == "__main__":
    main()
