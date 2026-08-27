# Scaling the Gemini linkography / network pipeline to 10 sessions

Deck slide 50 ("stay on two sessions before scaling") set five gates before adding
sessions. Slides 51–55 answer all five on NES_S3 and NES_S10, with the verdict *good
enough to scale* plus two standing caveats: **(A)** the 20–30% `nearest_prior` fallback
and **(B)** session variability. This directory now runs the same pipeline over 10
sessions and carries both caveats forward as logged, checkable output.

**The method did not change.** Every definition — idea unit, node, move, link, speaker
edge, metric — is imported from the existing single-session modules, so the 8 new
sessions are built exactly the way NES_S3 and NES_S10 were.

The proof is that re-running the focal pair through the batch reproduces their committed
outputs *exactly*. For both NES_S3 and NES_S10, `scale10/sessions/<LABEL>/` matches
`session_comparison/`:

| Artefact | Result |
|---|---|
| `gemini_linkography_<LABEL>.json` | identical (bar the new `links_to_name` field, §6) |
| `gemini_linkography_<LABEL>_report.md` | identical, character for character |
| `gemini_speaker_network_<LABEL>.json` | identical |
| `gemini_linkography_<LABEL>.png` | **pixel**-identical |
| `gemini_speaker_network_<LABEL>.png` | **pixel**-identical |

Re-check it any time with the comparison in `logs/` or by diffing the two trees. If a
future edit breaks that equality, the method drifted.

---

## 1. The 10 sessions

| Session id | Label | Conf | In-room teams | Funded | Speakers | Idea utts | Links | named | nearest_prior |
|---|---|---|---|---|---|---|---|---|---|
| `2020_11_05_NES_S3` | NES_S3 | 2020NES | 3 | 1 | 11 | 58 | 46 | 28% | 20% |
| `2020_11_06_NES_S10` | NES_S10 | 2020NES | 0 | 0 | 11 | 47 | 40 | 20% | 30% |
| `2020_11_05_NES_S4` | NES_S4 | 2020NES | 3 | 2 | 12 | 62 | 55 | 31% | 14% |
| `2021_10_08_CMC_S11` | CMC_S11 | 2021CMC | 1 | 0 | 10 | 51 | 48 | 31% | 10% |
| `2021_09_30_MZT_S5` | MZT_S5 | 2021MZT | 2 | 2 | 9 | 61 | 54 | 18% | 13% |
| `2022_04_07_MND_S5` | MND_S5 | 2022MND | 2 | 1 | 8 | 51 | 42 | 36% | 21% |
| `2021_05_21_ABI_S1` | ABI_S1 | 2021ABI | 0 | 0 | 11 | 50 | 34 | 15% | 21% |
| `2021_04_23_MND_S15` | MND_S15 | 2021MND | 0 | 0 | 10 | 47 | 38 | 21% | 3% |
| `2021_06_11_SLU_S5` | SLU_S5 | 2021SLU | 0 | 0 | 12 | 56 | 47 | 36% | 6% |
| `2021_11_05_NES_S8` | NES_S8 | 2021NES | 0 | 0 | 9 | 40 | 34 | 26% | 32% |

"In-room teams" is the `has_teams` framing from deck slide 26: how many teams formed
between **two people from this room**. A 0 does not mean nobody teamed up — NES_S10 has
0 in-room teams while 9 of its people teamed up with people from other sessions.

Denominator warning: the deck says "9 of 10" for NES_S10. Computed from
`2020NES_session_outcomes_v2.json` it is **9 of 11** — the roster lists 12 names, two of
which (`Remi Bouteille` / `Remi Boutelle`) are the same person. The same defect sits in
ABI_S1's roster (`Brian Spring` / `Bryan Spring`). Reconcile which denominator is right
before either number goes on a slide; this is a defect in the outcome data, not in the
annotations.

### Outcomes in full

Team outcomes are the one thing here that does **not** come from the Gemini annotations.
They are read from `analysis_v1/data/<conf>/<conf>_person_to_team.json` and
`<conf>_session_outcomes_v2.json`. Gemini never sees team formation — that independence
is what makes it usable as the outcome variable. Headcounts below are de-duplicated for
the two roster name-splits noted above. Every team in this set is a 2-person team.

| Label | People | In-room teams (funded) | Team ids | Teamed up *anywhere* | On a funded team |
|---|---|---|---|---|---|
| NES_S3 | 11 | 3 (1) | NES11 *funded*, NES14, NES23 | 8/11 | 4/11 |
| NES_S4 | 13 | 3 (2) | NES9, NES19 *funded*, NES29 *funded* | 10/13 | 5/13 |
| MZT_S5 | 12 | 2 (2) | MZT4 *funded*, MZT5 *funded* | 7/12 | 5/12 |
| MND_S5 | 11 | 2 (1) | MND5 *funded*, MND7 | 6/11 | 4/11 |
| CMC_S11 | 13 | 1 (0) | CMC10 | 6/13 | 2/13 |
| NES_S10 | 11 | 0 | — | 9/11 | 4/11 |
| NES_S8 | 10 | 0 | — | 8/10 | 1/10 |
| ABI_S1 | 11 | 0 | — | 5/11 | 3/11 |
| MND_S15 | 12 | 0 | — | 5/12 | 2/12 |
| SLU_S5 | 13 | 0 | — | 5/13 | 1/13 |

NES_S8 is a second NES_S10-shaped case: no in-room teams, but 8 of 10 teamed up
elsewhere. ABI_S1, MND_S15 and SLU_S5 are a different shape — low in-room *and* low
elsewhere.

### How these 8 were chosen

`gemini_session_inventory.py` surveys all **162** annotated sessions and applies six
criteria. **54** sessions passed all six; both focal sessions passed unchanged, which is
the check that the gate is not tuned around them.

| Criterion | Threshold | Why | Failed |
|---|---|---|---|
| `ok_complete` | every chunk json parses | some raw Gemini dumps are truncated (`ATTN_` prefix); a session missing part of its transcript is not comparable | 29 |
| `ok_material` | ≥ 35 idea-bearing utterances | enough linkograph to carry the metrics; floor sits below both focal sessions (58 / 47) | 44 |
| `ok_group` | 8–13 idea speakers | "multi-person team meeting" of the focal pair's shape (both 11) | 35 |
| `ok_length` | 90–230 utterances | drops the truncated and marathon outliers (focal: 139 / 162) | 19 |
| `ok_roster` | ≥ 70% of idea speakers resolve to the session roster | below this the speaker labels are too noisy to trust the nodes (slide 46 name-matching risk) | 4 |
| `ok_grounded` | `nearest_prior` share ≤ 35% | caveat (A) as a gate — nearest_prior links spot-checked at only ~55% correct, so a mostly-fallback session is not evidence. Cap sits just above NES_S10's 30%, the worst case actually read through | 46 |

Selection among the 54 eligible was then deliberate, not random:

- **Outcome contrast** — 5 sessions with in-room teams, 5 without. Base rate is 21%
  no-team, so this over-samples the negative case on purpose: with n=10 the comparison
  is descriptive, and a 9–1 split would carry no contrast.
- **Conference spread** — all 8 conferences are represented, so a finding is not a
  2020NES artifact.
- **Funded variation** among the team sessions (0, 1, 1, 2, 2 funded), separating
  "formed a team" from "got funded".
- **Comparability** — every session sits inside the focal pair's range on speakers,
  idea utterances and length.
- **Groundedness spread** — 3% to 32% `nearest_prior`. NES_S8 (32%) was kept
  deliberately as the stress case for caveat (A); MND_S15 (3%) is the clean end.

One caveat worth carrying: **2021ABI_S1 is the only ABI session that clears the material
floor.** ABI sessions carry a median 26 idea utterances against 46 overall, so ABI_S1
sits at the top of its conference rather than the middle of it. If the ABI result looks
unlike the others, that is the first thing to suspect.

Per-session rationale is in `scale_sessions.json`; the full 162-row inventory with every
criterion column is `session_inventory.csv`.

---

## 2. Changing which sessions run

Edit **`analysis_v2/src/linkograph_network/scale_sessions.json`** — it is the only place the
list lives. An entry is either a bare session id or an object:

```json
{"id": "2021_10_08_CMC_S11", "role": "scale", "outcome": "teams", "why": "..."}
```

`role` and `why` are documentation, with one exception: `role: "focal"` excludes a
session from the spot-check sample by default (the focal pair's links were already read
end to end during validation). Conference, short label and annotations directory are all
derived from the id — `2021_10_08_CMC_S11` → conf `2021CMC`, label `CMC_S11`, dir
`outputs/2021CMC/output_2021_10_08_CMC_S11` — so nothing else needs updating.

To run an ad-hoc set without touching the config, pass `--session` (repeatable).

---

## 3. Running it

**Data prerequisite.** `gemini_session_inventory.py` reads
`analysis_v1/data/<conf>/<conf>_session_outcomes_v2.json`, and the speaker network reads
`<conf>_person_to_team.json`. `person_to_team` is tracked; the `session_outcomes_v2`
files are **not** — they are produced by the session-outcome work on the
`update-session-lvl-name-matching` branch. Generate or obtain them before running the
inventory on a fresh clone.

Re-survey the candidate pool (only needed after annotations change or a threshold moves):

```bash
python analysis_v2/src/linkograph_network/gemini_session_inventory.py
```

Run the pipeline over the configured sessions:

```bash
python analysis_v2/src/linkograph_network/gemini_scale.py --spotcheck-n 4 --spotcheck-stratified
```

Takes a few seconds for all 10 — no Gemini calls, no LLM, everything reads the
annotations already on disk. Useful flags:

| Flag | Default | Effect |
|---|---|---|
| `--spotcheck-n N` | 4 | extends links sampled per session for manual checking; `0` turns sampling off |
| `--spotcheck-stratified` | off | spread the sample across named / on_table / nearest_prior instead of drawing at random |
| `--spotcheck-all` | off | also sample the two focal sessions |
| `--session ID` | — | run this id instead of the config list; repeatable |
| `--seed N` | 20260826 | fixes the draw — re-running gives the **same** rows to check |
| `--out-root` / `--fig-root` | `…/prototypes/scale10` | write somewhere else |

Plain random is the default because that is the neutral draw. `--spotcheck-stratified`
is usually the better one: at n=4 with a ~20% named / ~30% nearest_prior mix, a random
draw can miss the fallback tier entirely — and that tier is the reason the spot-check
exists.

To use it from Python instead:

```python
from gemini_scale import run_pipeline_on_sessions, load_session_list
ids, meta = load_session_list()
manifest = run_pipeline_on_sessions(ids, spotcheck_n=5, spotcheck_stratified=True)
```

Signatures:

```python
run_pipeline_on_sessions(session_ids, out_root=OUT_ROOT, fig_root=FIG_ROOT,
                         spotcheck_n=4, spotcheck_stratified=False, skip_spotcheck=(),
                         liu_json_for=None, seed=SEED, base=_BASE) -> dict   # the manifest

run_pipeline_on_session(session_id, out_root=OUT_ROOT, fig_root=FIG_ROOT,
                        liu_json=None, base=_BASE, log=print) -> dict        # payload/stats/metrics/log rows

resolve_session(session_id) -> {"session", "conf", "label", "session_dir"}
load_session_list(path=SESSIONS_JSON) -> (ids, {id: entry})
sample_extends_links(rows, n, seed_key, stratified=False) -> (sample, note)
split_speaker_candidates(speakers, cutoff=0.85) -> [[a, b], ...]
```

A session that raises is recorded in the manifest and the batch continues; the CLI exits
non-zero if anything failed.

---

## 4. Where the outputs go

```
analysis_v2/results/tables/prototypes/scale10/
├── session_inventory.csv                     all 162 sessions × every criterion column
├── metrics_gemini_linkography_scaled.csv     one row per session (the slide-48 block)
├── link_basis_log.csv                        one row per link, all 438 — caveat (A)
├── spotcheck_extends_sample.csv              N extends links per new session — caveat (B)
├── run_manifest.json                         what ran, counts, timings, output paths, failures
├── logs/<LABEL>.log                          per-session console log
└── sessions/<LABEL>/
    ├── gemini_linkography_<LABEL>.json       same structure as the single-session run
    ├── gemini_linkography_<LABEL>_report.md
    └── gemini_speaker_network_<LABEL>.json

analysis_v2/figures/prototypes/scale10/<LABEL>/
    ├── gemini_linkography_<LABEL>.png
    └── gemini_speaker_network_<LABEL>.png
```

Nothing is written to `session_comparison/` — the two-session deck outputs are left
exactly as they are.

---

## 5. The logging, and what to do with it

### `link_basis_log.csv` — caveat (A), every link

One row per resolved link across all 10 sessions (438 rows, 200 of them
`extends_existing_idea`). Columns worth knowing:

- **`basis`** — which rule resolved the target: `named_speaker` (explanation named a
  participant), `on_table_idea` (word overlap with a live idea), `nearest_prior` (pure
  positional fallback). Across the 10: 117 named / 249 on_table / 72 nearest_prior —
  **16% fallback overall, 3%–32% per session**, i.e. at or below the validated band.
- **`links_to_extracted` / `links_to_name`** — whether a real `links_to` target came out
  of Gemini's explanation, and who. This is the slide-54 question answered per link
  rather than per session. It reproduces the deck exactly on the focal pair (28% S3 /
  20% S10) and runs **28% across the 10**. It can be `True` on a non-`named_speaker`
  link: the explanation named someone who had no prior idea utterance to link back to.
- **`is_cross_chunk`** — the slide-55 check: does context survive chunk boundaries?
- **`from_evidence` / `from_explanation` / `to_evidence`** — both ends in full, so a row
  can be judged without opening the JSON.

Read a session's metrics with its basis mix next to them. A `nearest_prior` link
spot-checked at ~55% correct against ~85% for `named_speaker`; treating all links as
equal evidence hides that gradient.

### `spotcheck_extends_sample.csv` — caveat (B), manual verification

4 `extends_existing_idea` links per newly added session (32 rows; the focal pair is
excluded by default). Each row has both ends' speaker, timestamp, chunk file, evidence
and explanation, plus two blank columns to fill in:

- **`correct_yn`** — does the move actually build on the idea it was linked to?
- **`notes`** — if not, what did it point at instead?

The draw is seeded, so a re-run gives the same rows and a partly-filled file stays
valid. This is the same judgment made on all 86 focal-pair edges during validation
(~70% correct, tracking the basis tier), now sampled per session so variability across
sessions is measurable rather than assumed.

### Name splits — flagged, not fixed

The run warns when two speaker labels in one session look like one person written two
ways. Two showed up in this set:

- **ABI_S1** — `Brian Spring` / `Bryan Spring`
- **SLU_S5** — `Tori Hoehler` / `Tori Hoeler`

Each becomes **two network nodes** and splits that person's links, inflating the speaker
count by one. These are not auto-merged: merging on name similarity would also merge
genuinely different people (the Kris Hall / Kirsten Hall case on slide 46), which is the
worse error. Adjudicate them before quoting ABI_S1 or SLU_S5 network numbers.

---

## 6. What changed in the shared code

Four changes to `gemini_linkography.py`, all needed for the scale-out. Verified against
the committed NES_S3 / NES_S10 outputs: utterances, edges and network stats are
byte-identical.

1. **Both directory layouts.** Only 2020NES nests chunk files in one subdirectory per
   recording; every other conference stores them flat in the session directory. The old
   glob (`*/*.json`) found **zero** files for all 145 non-2020NES sessions. `load_utterances`
   now reads both, and orders recordings by the start timestamp parsed out of the name
   (`YYYY_MM_DD_hh_mm_ss`, or 2021ABI's `M_D_YYYY_h_m_s_AM/PM`) instead of sorting the raw
   string.
2. **Unparseable chunks no longer kill a session.** 29 sessions contain at least one
   truncated Gemini dump. `load_utterances(..., on_bad_file=cb)` skips and reports them;
   with no callback it raises exactly as before. Sessions with any bad file fail
   `ok_complete` and are excluded from the 10 regardless.
3. **`links_to_name` recorded on every edge.** The participant the explanation names,
   when it names one. **Recorded, not used** — target selection is untouched — so the
   extraction rate can be reported independently of which rule fired.
4. **Determinism fix.** The named-participant scan iterated a `set`, so `links_to_name`
   differed between runs on any explanation naming two people, and a first-name collision
   between two participants resolved arbitrarily. Both now iterate sorted, and the
   recorded name is the first **other** participant the explanation mentions, by position
   in the text. Edge selection was never affected (it tested set membership), but the
   recorded field was.

`linkography_payload()` was also factored out of `main()` so the batch runner and the
single-session CLI write the same structure from one definition rather than two copies.

---

## 7. Deck figures

`gemini_scale_figures.py` renders four paste-ready PNGs into
`analysis_v2/figures/prototypes/scale10/_deck/`. They read the CSVs above rather than
recomputing anything, so a figure can never disagree with the table behind it.

```bash
python analysis_v2/src/linkograph_network/gemini_scale_figures.py
```

| File | Shows | Answers |
|---|---|---|
| `scale_selection.png` | 162 → 54 → 10 funnel + the six criteria and what each excluded | how the sessions were picked |
| `scale_basis_mix.png` | per-session named / on_table / nearest_prior, against the band the focal pair was validated in, plus `links_to` extraction rate | slides 51 + 54 at scale — caveat (A) |
| `scale_metrics_by_outcome.png` | the slide-48 block, 5 team vs 5 no-team, every session as its own point | does the S3-vs-S10 direction hold? |
| `scale_summary_table.png` | the slide-48 block as a 10-column table grouped by outcome, with the basis mix as a footer | the scaled `gemini_summary_table.png` |

The 20 per-session figures (`scale10/<LABEL>/`) are appendix material. Nothing exists yet
for the spot-check results slide — that needs `correct_yn` filled in first.

## 8. Not done here

- **Liu cross-tab at scale.** `run_pipeline_on_session` accepts a `liu_json`, and
  `run_pipeline_on_sessions` a `liu_json_for={session_id: path}` map, but no Liu outputs
  exist for the 8 new sessions — only the focal pair has them. Generate them with
  `liu2026_linkography_pipeline.py` if the slide-52 comparison should extend past NES.
- **"Bridging conversions"** is still blocked on Evey's definition (open item 1 in
  `session_story/HANDOFF.md`). The implemented `bridging_moves` metric is a different
  thing and appears in the metrics table under its own name.
- **Deck figures.** `gemini_summary_table.py` and the validation figures are still
  hard-coded two-session (S3 vs S10) layouts. They were left alone; a 10-column version
  is a separate design question.
