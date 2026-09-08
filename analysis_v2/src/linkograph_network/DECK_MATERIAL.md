# Raw material for the synthesized deck

Not slides. Content to draw from — more than you'll use.

---

## 1. The finding, three ways

**One sentence.** We can now reconstruct, automatically and from video annotations alone,
who built on whose ideas in a brainstorm — and across ten sessions, none of the idea-flow
or network measures cleanly separates the rooms where people formed teams from the rooms
where they didn't.

**One paragraph.** Gemini annotates each turn of a Scialog brainstorm with behaviour codes
but never records which earlier idea a speaker is building on. We reconstruct that link
with a deterministic three-rule procedure, stamp every link with which rule produced it,
and from those links build both a linkograph (how ideas developed) and a speaker network
(who built on whom). Applied to ten sessions — five where two people from the room went on
to form a team, five where none did — idea production points weakly in the expected
direction, but network structure does not, and every measure overlaps between the two
groups. The most evenly-participating session in the whole set produced no team; the
second-most-centralized produced two funded ones.

**Half a page.** The question is why some brainstorm rooms produce research collaborations
and others don't. Prior work gives us two lenses: linkography, which treats a design
conversation as ideas linked to earlier ideas, and interaction networks, which treat a
meeting as people linked by who responds to whom. Both need a link between contributions,
and Gemini's annotations don't contain one — so the methodological contribution here is a
transparent, auditable way to recover it, with a confidence tier attached to every link.
Applied at n=10 with a deliberately balanced outcome split, the result is largely
negative: volume, breadth, depth and chain length run slightly higher in team-forming
sessions, but the groups overlap on all of them, and the network measures — centralization
and betweenness — run the *opposite* way to what the two-session pilot suggested. The
honest reading is that at this sample size the method demonstrates it can measure the
constructs, not that the constructs predict the outcome. Two things follow: the sample
needs to grow, and the outcome variable may need rethinking, because "two people from this
specific room teamed up" has only 34 positive cases in the entire corpus of 162 sessions.

---

## 2. Candidate findings, ranked by how well the data supports them

**Strong — I'd put these on a slide.**

*The link layer can be reconstructed and audited.* 438 links across 10 sessions, each
tagged with which of three rules produced it and traceable to the specific utterances
behind it. Nothing about this is a black box. Confidence: high, it's a property of the
implementation.

*Link accuracy holds outside the conference where it was developed.* Reading 32 sampled
links across all 8 conferences and all three rules: named-speaker ~85%, word-match ~75%,
fallback ~65% — essentially matching the estimates from the original two-session
hand-check. Confidence: moderate. 32 links is a small sample and it's my read, not an
independent one.

*Gemini rarely says whose idea is being built on.* 78% of 6,157 moves across 157 sessions
give no usable target. It varies enormously by move type — 52% for
`resolves_contradiction`, **2.9%** for `raises_concern`. Confidence: high, it's a
straightforward count of the full corpus.

**Weak — true in the data, but I'd hedge or leave out.**

*Idea production is slightly higher in team-forming sessions.* Volume 56.6 vs 48.0,
breadth 14.6 vs 12.8, depth 0.87 vs 0.81, longest chain 6.8 vs 5.6. Every one overlaps
between groups. n=5 per side. Confidence: low — a direction, not a result.

**Runs against the earlier story — say it yourself.**

*Network centralization does not behave as the pilot suggested.* In NES_S3 vs NES_S10,
centralization was the headline: the successful room was distributed, the unsuccessful one
funnelled through Wilson Smith. Across ten sessions it reverses — 0.57 for team-forming
sessions against 0.61 for the others, and betweenness 0.28 against 0.32. **MND_S15 has the
most even participation of all ten (0.957) and produced no team. MZT_S5 is the second most
centralized (0.712) and produced two funded teams.** Confidence: the numbers are solid;
the interpretation is that the pilot pattern doesn't generalize, not that it reverses.

*No metric separates in-room-team sessions from the rest.* The two sessions with confirmed
in-room teammates sit at opposite ends of self-linking (0.073 and 0.262). Confidence:
high that nothing separates them here; low that this means nothing does.

---

## 3. Which figures earn their place

**Worth showing (2–3 max):**

- `scale10/_deck/scale_metrics_by_outcome.png` — the slide-48 block, team vs no-team, each
  session as its own point. **Shows the overlap honestly**, which is the finding.
- `scale10/_deck/scale_basis_mix.png` — per-session link groundedness against the band the
  method was validated in. **This is the methodological credibility slide** — it shows
  quality didn't degrade at scale and in some sessions improved.
- One paired linkograph + network, e.g. `scale10/MND_S15/` — as an illustration of what the
  method produces, not as evidence of anything.

**Decorative — leave out:** the remaining 14 per-session figure pairs (appendix at most),
`scale_selection.png` (process, not finding), `scale_summary_table.png` (same content as
the by-outcome figure but harder to read).

---

## 4. The method, three lengths

**One sentence.** We turn Gemini's per-utterance behaviour codes into a graph of which
idea built on which, then aggregate that into a network of who built on whom.

**One paragraph.** An utterance enters the analysis if it carries one of 11 idea-move
codes — two that open a new thread, nine that act on an existing one. Gemini never records
*which* earlier idea a move acts on, so we reconstruct it with three rules in priority
order: the explanation names a person; the words overlap an idea currently on the table;
or, failing both, the most recent earlier idea. Every link records which rule fired, so
groundedness travels with the number. Links between different speakers aggregate into
weighted network edges; links from a person to their own earlier idea are excluded from
the network and counted separately as a self-link index.

**A slide's worth.** Use `EDGE_METHOD.md` — it's written for exactly this and is one page.

---

## 5. Quotable examples

*The mechanism working in plain view* — NES_S8. Wilson Smith: **"back to Linsey's point…
having local identification under reaction conditions."** He names his target out loud;
the linker resolved it to Linsey Seitz. This is the ~85% tier.

*Three people building one list* — MND_S15. Ying-Hui Chou: **"like traumatic brain injury,
that could be a stress."** Ali Keshavarzian: **"vaginal environment is under tremendous
amount of stress as well."** Stavroula Hatzios: **"and antibiotics too."** Textbook
additive building — in a session that produced no team.

*A thread widening across three people* — NES_S4. Chong Liu: **"nature has enzymes to do
methane to liquid… maybe they just need an energy source like electrochemistry."** Michael
Nippe extends it to cofactor generation. Katie Knowles then: **"build off an idea that
Michael just said… using electrochemistry more broadly."**

*Disagreement that stays on-thread* — CMC_S11. Gulcin Pekkurnaz: **"if size matter that
much, why neurons are the exception?"** Wenjing Wang: **"I want to argue against that the
size of the neuron is so different."**

*What failure looks like* — CMC_S11. Judith Su suggests **"you could do some modeling to
get the diffusion over a particular length."** The fallback rule linked it to Davide
Donadio's **"I'm a physicist but I don't buy that"** — because the rule takes the previous
turn without checking whether it contains an idea. Good slide if you want to show you know
where the method breaks.

---

## 6. Literature hooks

Already cited in your deck and directly relevant:

- **Goldschmidt (1994, 2014)** — linkography, the move as the base unit, link density and
  critical moves. Our node/move split maps onto this.
- **van der Lugt (2000, 2003)** — self-links and the self-link index; link types
  (supplementary / modification). Our SLI is his, with a different denominator.
- **Sauer & Kauffeld (2013)** — meetings as networks, participants as nodes, weighted
  degree as the measure. Our speaker network follows this convention.
- **Freeman (1979)** — degree centralization and betweenness.
- **Burt (2004) / Fleming (2001)** — structural holes and brokerage; the basis for
  bridging.
- **Shah et al. (2003)** — ideation effectiveness metrics: fluency, variety. Our breadth
  and volume.
- **Hatcher et al. (2018)** — using linkography to compare group ideation methods, and the
  warning that link judgment is subjective.

Gap worth naming: none of these connect idea-flow structure to *downstream team formation*.
That's the novel move here, and it's also why there's no established expectation for which
metric should predict it.

---

## 7. Questions to expect, and honest answers

**"How do you know who's building on whom?"** Three rules, priority order, every link
tagged with which fired. 21% by name, 57% by word-overlap with live ideas, 16% positional
fallback. Accuracy ~85 / ~75 / ~65 respectively.

**"How did you check that without reading transcripts?"** We read all 86 edges in the two
pilot sessions by hand, and 32 sampled links across all 8 conferences. Both ends of every
link are in `link_basis_log.csv` with the quoted text, so any link can be checked without
opening code.

**"What's the dependent variable?"** Whether two people *from the same room* formed a
team, and whether it was funded. Only 34 of 162 sessions have any in-room team — worth
flagging that this is a rare outcome.

**"So what predicts team formation?"** Nothing we can show at n=10. Idea production points
weakly the right way; network structure doesn't. That's the current state.

**"Why did centralization reverse?"** Two sessions aren't a sample. The pilot pattern was
real for those two rooms and doesn't hold across ten.

**"Can you scale to everything?"** The pipeline runs on ~130 of 162 in seconds. Two things
need doing first: divide the count metrics by session length (they currently correlate 0.6
with it, masked because our ten are all similar length), and decide whether sessions where
most links are fallback come in with a flag or stay out.

**"What about the sessions Gemini barely coded?"** CMC_S7 has 168 turns and only 17 idea
codes. It wasn't a quiet meeting — it was doing knowledge-sharing and coordination, which
our 11 idea codes don't capture. About 60% of all turns are invisible to the linkograph.
That's a definitional boundary, not missing data.

**"What don't you know?"** Whether link accuracy holds up on a bigger sample than 32, and
whether "in-room team formation" is the right outcome given how rare it is.
