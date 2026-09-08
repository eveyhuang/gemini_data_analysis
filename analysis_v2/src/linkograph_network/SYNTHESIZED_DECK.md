# Synthesized deck — for collaborators and senior audiences

Ten slides. Story first; every figure is evidence for a claim already stated in words.
Build order follows how Brian reads: problem → why it matters → variables → what we did →
what we found → what it rests on.

Figures referenced live in `analysis_v2/figures/prototypes/scale10/`.

---

## 1 · Title

**Do brainstorms that build on ideas better produce more research teams?**
Measuring idea flow and collaboration structure in Scialog sessions from multimodal LLM
annotations

Max Chalekson · with Evey Huang · [date]

---

## 2 · The problem

**Scialog puts strangers in a room for an hour and hopes teams come out.** Some rooms
produce funded collaborations. Most don't.

- Across 162 annotated sessions, only **34** produced a team between two people **from
  that room**
- Nobody can currently say what distinguishes the rooms that did
- The conventional answers — who talked most, how long they met — aren't measurable at
  scale from recordings, and haven't been tested

**Why it's interesting:** if the difference is *how ideas got built on* rather than *who
was in the room*, that's actionable. You can design a session; you can't redesign the
people.

**Speaker note:** the honest hook is that this is a well-funded intervention running for
years with no measurement of its core mechanism.

---

## 3 · The question and the variables

**Question.** Does the structure of idea-building in a session predict whether people from
that session go on to form a research team?

| | |
|---|---|
| **DV** | Team formation. Primary: number of teams formed between two people *from the same room* (0–3). Secondary: share of the room joining any team; share on a funded team. |
| **IV — idea flow** | How many distinct new ideas; how much total idea work; how deeply ideas were built on; how long the longest build-chain ran |
| **IV — collaboration structure** | Who built on whom; how evenly participation was spread; how concentrated the building was on one person; who bridged otherwise-separate parts of the room |
| **IV — roles** | Who initiates, builds, synthesizes, evaluates — derived from which move codes each person uses. And **role versatility**: what share of the room plays more than one role |
| **Unit** | One session (n = 10 analysed; 162 available) |

**Say out loud:** a "0" on the DV does **not** mean nobody formed a team. NES_S10 has zero
in-room teams and 9 of its 11 people teamed up in *other* sessions. We're measuring
whether *this room* produced a pairing.

---

## 4 · What we did

**Google Gemini annotates each turn of the recorded session** — who spoke, when, and what
they were doing (proposing an idea, extending one, challenging one, coordinating).

**We turn those annotations into two pictures of the session:**

- a **linkograph** — each idea moment as a point, arcs joining a move to the earlier idea
  it built on
- a **speaker network** — people as nodes, connected when one built on the other's idea

**The hard part:** Gemini records *that* someone built on an earlier idea but never *which
one*. We reconstruct that with three rules in priority order, and tag every link with
which rule produced it, so confidence travels with the number.

**Applied to 10 sessions** — five where teams formed in the room, five where none did,
across all 8 conferences.

**Figure:** one paired linkograph + network, e.g. `MND_S15/linkograph_MND_S15_gemini.png`.
Use it to *show what the output looks like*, not to prove anything.

---

## 5 · Finding 1 — denser idea work, more teams

**Rooms where a larger share of the conversation was idea work produced more teams.**

| Session | In-room teams | Ideas per turn |
|---|---|---|
| MZT_S5 | 2 | **0.48** |
| NES_S3 | 3 | **0.42** |
| NES_S4 | 3 | **0.40** |
| SLU_S5 | 0 | 0.35 |
| NES_S10 | 0 | 0.29 |
| CMC_S11 | 1 | 0.28 |
| ABI_S1 | 0 | 0.28 |
| MND_S15 | 0 | 0.28 |
| MND_S5 | 2 | 0.25 |
| NES_S8 | 0 | 0.24 |

**r = 0.63** (0.74 before adjusting for session length). Robust to dropping any single
session (0.66–0.85).

**It isn't just longer meetings.** Session length correlates **−0.32** with team formation
— the longer rooms did slightly *worse*. It's density, not duration.

**The exception:** MND_S5 formed two teams at the second-lowest density. Show it rather
than hide it.

---

## 5b · Finding 2 — rooms where people played more than one role

**Brian's question — do individuals play multiple roles? — turns out to be the strongest
single signal we have.**

Each move code maps to a role: *initiator* (proposes, frames), *builder* (extends,
combines, connects), *synthesizer* (synthesizes, returns, resolves), *evaluator*
(critiques, raises concern). We count someone as playing a role if they use it at least
twice.

| Session | In-room teams | Share of room playing 2+ roles |
|---|---|---|
| MND_S5 | 2 | **0.75** |
| MZT_S5 | 2 | **0.67** |
| NES_S4 | 3 | **0.58** |
| NES_S3 | 3 | **0.55** |
| NES_S10 | 0 | 0.55 |
| MND_S15 | 0 | 0.40 |
| SLU_S5 | 0 | 0.36 |
| CMC_S11 | 1 | 0.30 |
| ABI_S1 | 0 | 0.30 |
| NES_S8 | 0 | 0.22 |

**r = 0.68**, leave-one-out 0.64–0.77.

**It is not the role *mix* that matters.** The share of the conversation spent initiating,
building, synthesizing or evaluating correlates with nothing (−0.20 to +0.20). What
matters is whether the *same people* do several of those things.

**And it is not just density restated** — the two correlate at 0.46, so they overlap but
measure different things. Nor is it room size (−0.12).

**Exceptions to show:** NES_S10 has high versatility (0.55) and no in-room teams;
CMC_S11 has the second lowest (0.30) and formed one.

---

## 6 · Finding 3 — but network structure doesn't predict it

**We expected network shape to matter. It doesn't.**

| | Correlation with team formation |
|---|---|
| Degree centralization | −0.18 |
| Max betweenness | −0.22 |

Our two-session pilot suggested the opposite — the successful room looked distributed, the
unsuccessful one funnelled through one person. **Across ten sessions that doesn't hold.**

- **MND_S15** has the most even participation of all ten (0.957) and produced **no team**
- **MZT_S5** is the second most centralized (0.712) and produced **two funded teams**

**Figure:** `_deck/scale_metrics_by_outcome.png` — each session as its own point, so the
overlap is visible.

**Speaker note:** this is the slide that buys credibility. Say it before anyone finds it.

---

## 7 · What that suggests

**What people *do* matters more than how the network is *shaped*.**

The two things that track team formation — how densely the room does idea work, and how
many people play more than one role — are both about behaviour. The things that don't —
centralization, betweenness — are about structure.

A room can run through one dominant person and still produce teams, as long as the
conversation is densely about ideas and people are doing more than one kind of idea work.
A room can be perfectly egalitarian and produce nothing.

**Two readings, and we can't yet separate them:**
- Dense idea work *causes* people to find collaborators
- Rooms with compatible people *produce* dense idea work and teams independently

Distinguishing them needs more sessions and probably the temporal data we haven't used —
whether density rises or falls across the hour.

---

## 8 · How much to trust the links

**Every link is tagged with how it was resolved**, so the numbers carry their own
confidence:

| Rule | When it fires | Share | Accuracy |
|---|---|---|---|
| Gemini names the person | "builds on Chong's idea" | 21% | ~85% |
| Words match a live idea | — | 57% | ~75% |
| Fallback: the previous idea | — | 16% | ~65% |

**Checked by hand:** all 86 links in the two pilot sessions, plus 32 sampled across all 8
conferences and all three rules. The accuracy tiers held outside the conference where the
method was developed.

**Figure:** `_deck/scale_basis_mix.png`.

**One example, to make it concrete.** Wilson Smith: *"back to **Linsey's** point… having
local identification under reaction conditions."* He names his target out loud; the link
resolved to Linsey Seitz.

---

## 9 · What we can't see yet

**Gemini names whose idea is being built on only 22% of the time.** Measured across 6,157
moves in 157 sessions. It varies by what people are doing — 52% when resolving a
contradiction, **3%** when raising a concern. So most links rest on weaker inference, and
that's the ceiling on the whole approach.

**About 60% of conversation isn't idea-coded at all** — coordination, knowledge-sharing,
relational talk. One session had 168 turns and only 17 idea moments. It wasn't a quiet
room; it was doing something our codes don't capture.

**We have never validated against the raw transcripts** — Gemini's output doesn't include
the utterance text, only the snippets it chose to quote. Closing that needs the source
recordings.

**n = 10.** The correlation is exploratory, not confirmed.

---

## 10 · What's next

1. **Scale to more sessions** — the pipeline runs on ~130 of the 162 in seconds. The
   finding exists; it now needs testing on sessions that didn't generate it.
2. **Validate against transcripts** — needs the source recordings alongside the
   annotations.
3. **Ask Gemini for the target directly.** If the annotation prompt asked *which* idea is
   being built on, the entire reconstruction layer and its uncertainty would mostly
   disappear. This is the single highest-leverage change.
4. **Reconsider the outcome.** "Two people from this room teamed up" has only 34 positive
   cases in 162 sessions. Share of the room joining any team varies more and may be the
   better dependent variable.

---

## Appendix, if asked

- Method in full: `EDGE_METHOD.md`
- Missing-names audit: `NAME_AUDIT.md`
- Per-session profiles: `SESSION_PROFILES.md`
- Every link with both ends quoted: `link_basis_log.csv`
- Literature: Goldschmidt (1994, 2014) linkography; van der Lugt (2000, 2003) self-links;
  Sauer & Kauffeld (2013) meetings as networks; Freeman (1979) centralization; Burt (2004)
  brokerage; Shah et al. (2003) ideation metrics
