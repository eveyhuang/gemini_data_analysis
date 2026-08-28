# Deck section: 10 sessions — what to build

**What you did for NES_S3 / NES_S10:**

| Slides | What they are | Do they change at 10 sessions? |
|---|---|---|
| 45–48 | Component + metric **definitions** | **No.** Same definitions, same 11 codes, same 3-rule linker. Nothing to redo. |
| 49 | Metric values, `Metric \| NES_S3 \| NES_S10` | **Yes** — becomes one table with 10 columns |
| 57–60 | 2 linkographs + 2 networks | **Yes** — becomes 10 + 10 |

So the section is **two content slides plus the figures.** That's it.

Figures: `analysis_v2/figures/prototypes/scale10/_deck/` and
`analysis_v2/figures/prototypes/scale10/<LABEL>/`

---

# The minimum — build these

## 62 — The 10 sessions

**Figure:** none — table slide. Needed so the columns on slide 63 mean something.

**Columns: Session | Conference | In-room teams | Funded | People**

| Session | Conference | In-room teams | Funded | People |
|---|---|---|---|---|
| **NES_S3** *(original)* | 2020NES | 3 | 1 | 11 |
| NES_S4 | 2020NES | 3 | 2 | 13 |
| MZT_S5 | 2021MZT | 2 | 2 | 12 |
| MND_S5 | 2022MND | 2 | 1 | 11 |
| CMC_S11 | 2021CMC | 1 | 0 | 13 |
| **NES_S10** *(original)* | 2020NES | 0 | 0 | 11 |
| NES_S8 | 2021NES | 0 | 0 | 10 |
| ABI_S1 | 2021ABI | 0 | 0 | 11 |
| MND_S15 | 2021MND | 0 | 0 | 12 |
| SLU_S5 | 2021SLU | 0 | 0 | 13 |

**One line under the table:** Five sessions where two people from the room teamed up with
each other, five where none did. All 8 conferences represented.

**Notes:** "In-room teams" is the same `has_teams` framing as slide 26 — a 0 doesn't mean
nobody teamed up. NES_S8 is a second NES_S10: no in-room teams, but 8 of its 10 people
teamed up in other sessions.

---

## 63 — Summary statistics: 10 sessions

This is **slide 49 with 10 columns instead of 2** — same row labels, same order. Build it
as a native PowerPoint table the way slide 49 is. Verified: the NES_S3 and NES_S10 columns
match the values already on slide 49 on all 12 rows.

**Columns: Metric | 5 sessions with in-room teams | 5 without**
(put a divider or shading between CMC_S11 and NES_S10)

| Metric | NES_S3 | NES_S4 | MZT_S5 | MND_S5 | CMC_S11 | NES_S10 | NES_S8 | ABI_S1 | MND_S15 | SLU_S5 |
|---|---|---|---|---|---|---|---|---|---|---|
| Breadth - new ideas | 19 | 13 | 16 | 13 | 12 | 15 | 10 | 15 | 11 | 13 |
| Volume - all ideas act | 58 | 62 | 61 | 51 | 51 | 47 | 40 | 50 | 47 | 56 |
| Distribution - participation | 0.953 | 0.948 | 0.91 | 0.913 | 0.877 | 0.904 | 0.895 | 0.815 | 0.957 | 0.94 |
| Depth - link ratio | 0.793 | 0.887 | 0.885 | 0.824 | 0.941 | 0.851 | 0.85 | 0.68 | 0.809 | 0.839 |
| Longest Build Chain | 7 | 6 | 6 | 8 | 7 | 7 | 6 | 5 | 5 | 5 |
| Bridging moves - joins separate idea threads | 3 | 1 | 2 | 0 | 1 | 2 | 0 | 0 | 1 | 2 |
| Self-link ratio | 0.217 | 0.073 | 0.204 | 0.262 | 0.188 | 0.05 | 0.118 | 0.206 | 0.158 | 0.128 |
| Max betweenness | 0.221 | 0.338 | 0.317 | 0.187 | 0.343 | 0.426 | 0.131 | 0.548 | 0.26 | 0.258 |
| Bridges sharing | 2 | 1 | 1 | 4 | 2 | 1 | 3 | 1 | 2 | 5 |
| In-room teams | 3 | 3 | 2 | 2 | 1 | 0 | 0 | 0 | 0 | 0 |
| Weighted-degree centralization | 0.38 | 0.705 | 0.712 | 0.462 | 0.569 | 0.653 | 0.589 | 0.74 | 0.465 | 0.625 |
| Max weighted degree | 10 | 24 | 26 | 13 | 16 | 17 | 14 | 15 | 11 | 16 |

**Optional footer row** (the groundedness check — slide 48's summary-table figure carries
the same thing):

| Link basis mix (named / on-table / nearest_prior) | 13/24/9 | 17/30/8 | 10/37/7 | 15/18/9 | 15/28/5 | 8/20/12 | 9/14/11 | 5/22/7 | 8/29/1 | 17/27/3 |
|---|---|---|---|---|---|---|---|---|---|---|

**One line under the table:** No metric separates the two groups cleanly — the highest and
lowest value on most rows sit on the same side of the divider.

**Notes:** Twelve rows × ten columns is wide. If it won't fit legibly, either (a) split
into two slides — idea production (breadth, volume, distribution, depth, chain, bridging)
then network (self-link, betweenness, bridges sharing, centralization, max weighted
degree) — or (b) use the pre-rendered figure `scale_summary_table.png` instead, which
carries the same numbers with provenance dots and the basis mix already in the footer.
If asked about the basis mix: across all 10 sessions the `nearest_prior` fallback is
**16%**, down from 20% / 30% in the original pair; per-session range 3%–32%.

---

## 64–79 — Per-session linkographs and networks

**Only the eight NEW sessions.** NES_S3 and NES_S10 are already on slides 57–60; putting
them here too would show the same session twice.

All files live in `analysis_v2/figures/prototypes/scale10/<LABEL>/`. Each session folder
holds four PNGs — use the `*_gemini.png` pair, which is the **deck style** matching slides
57–60 (node colour = speaker in both figures, ★ new ideas, black-outlined critical moves,
dashed self-links, gold convergence rings, team-outcome borders, magenta in-room rings).
The `gemini_linkography_*` / `gemini_speaker_network_*` pair is the plainer prototype
render — same data, no speaker colouring.

The deck-style figures carry their own full title, e.g. *"NES_S4 (11/05) — Gemini
linkography · 3 teams formed WITHIN this room (2 funded) · 62 relevant messages · link
ratio 0.89 · self-link index 0.073"*, so a separate slide title is optional.

### Linkographs — slides 64–71

| # | Session | File (in `scale10/<LABEL>/`) |
|---|---|---|
| 64 | NES_S4 | `linkograph_NES_S4_gemini.png` |
| 65 | MZT_S5 | `linkograph_MZT_S5_gemini.png` |
| 66 | MND_S5 | `linkograph_MND_S5_gemini.png` |
| 67 | CMC_S11 | `linkograph_CMC_S11_gemini.png` |
| 68 | NES_S8 | `linkograph_NES_S8_gemini.png` |
| 69 | ABI_S1 | `linkograph_ABI_S1_gemini.png` |
| 70 | MND_S15 | `linkograph_MND_S15_gemini.png` |
| 71 | SLU_S5 | `linkograph_SLU_S5_gemini.png` |

### Networks — slides 72–79

| # | Session | File (in `scale10/<LABEL>/`) |
|---|---|---|
| 72 | NES_S4 | `network_NES_S4_gemini.png` |
| 73 | MZT_S5 | `network_MZT_S5_gemini.png` |
| 74 | MND_S5 | `network_MND_S5_gemini.png` |
| 75 | CMC_S11 | `network_CMC_S11_gemini.png` |
| 76 | NES_S8 | `network_NES_S8_gemini.png` |
| 77 | ABI_S1 | `network_ABI_S1_gemini.png` |
| 78 | MND_S15 | `network_MND_S15_gemini.png` |
| 79 | SLU_S5 | `network_SLU_S5_gemini.png` |

Regenerate with:

```bash
python analysis_v2/src/linkograph_network/gemini_deck_figures_scaled.py
```

**Caveat for slides 77 and 79.** ABI_S1 and SLU_S5 each render one person as two nodes —
`Brian Spring` / `Bryan Spring` and `Tori Hoehler` / `Tori Hoeler`. On SLU_S5 that splits a
weighted degree of 15 into an 8 and a 7. Note it so the sociogram isn't read as a real
structural finding.

---

# Optional — only if Evey or Josh asks

Written and ready, but not needed to say "we scaled it."

| Slide | Title | Figure | When you'd want it |
|---|---|---|---|
| — | How we picked the eight | `scale_selection.png` | If asked "why these 10 and not others" |
| — | Does the NES_S3 vs NES_S10 pattern hold? | `scale_metrics_by_outcome.png` | If asked "so what does it show" |
| — | The same pipeline, applied unchanged | none | If asked "did the method change" |

### How we picked the eight

162 annotated sessions → 54 pass six criteria → 10 selected. Both original sessions pass
unchanged. Selection was deliberate: four teams / four no-teams, all 8 conferences, funded
variation, every session inside NES_S3 / NES_S10's range on speakers and length.

### Does the pattern hold?

| Metric | Teams (n=5) | No teams (n=5) | Holds? |
|---|---|---|---|
| Volume | 56.6 | 48.0 | Yes, weakly |
| Breadth | 14.6 | 12.8 | Yes, weakly |
| Depth | 0.87 | 0.81 | Yes, weakly |
| Longest chain | 6.8 | 5.6 | Yes, weakly |
| Degree centralization | 0.57 | 0.61 | **No — runs the other way** |
| Max betweenness | 0.28 | 0.32 | **No — runs the other way** |

Every panel overlaps. At 5 vs 5 this is a direction, not a result.

### The same pipeline, applied unchanged

Re-running NES_S3 and NES_S10 through the batch reproduces their existing outputs: the
linkography json, the report md, the network json all identical, and both PNGs
**pixel**-identical. One field was added (`links_to_name`) — recorded, never used.

---

# Separate from the deck section: fixes to slides you already have

Not scaling work. Four are cosmetic; the first two are wrong numbers.

| Slide | Issue | Fix |
|---|---|---|
| 51 | Says the target is named "~37% / ~27%"; slide 54 says "~28% / ~20%" for the same thing | Use **28% / 20%** — that's what the code reproduces |
| 30 vs 49 | Self-link index **reverses**: Liu 0.375 / 0.947 ("S3 loops less") vs Gemini 0.217 / 0.05 ("S10 loops less"). Slide 52 says the methods "reach the same conclusion" | Our linker gives each move **one** target while Liu's judges every pair, so our SLI understates self-linking — it's a floor, not a comparable number. Amend slide 52 |
| 51 | Quotes *"adding transport to the list of limitations"* as a topic-only reference | The full explanation names Jeffrey and resolved as `named_speaker`. Pick a genuinely topic-only example |
| 57 | Callout clipped mid-word: `…Andrew: 2 self-repli` | Regenerate or widen the box |
| 57–60 | No slide titles — the title lives inside the PNG | Add title text |
| 59, 60 | "centralization 0.300 / 0.433" (unweighted) vs slide 49's "Weighted-degree centralization 0.38 / 0.65" | Relabel the figure titles "unweighted degree centralization" |
