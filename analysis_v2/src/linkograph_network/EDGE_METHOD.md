# How an edge is built

One page, for the question "how did you decide who is building on whom?"

## The problem this solves

Gemini annotates each utterance with behaviour codes. When someone builds on an earlier
idea it records **that** they did — `extends_existing_idea`, `synthesizes_contributions`,
`critiques_or_challenges` and six others — but never **which** idea. There is no
`links_to` field anywhere in the annotations. Every link in our linkographs and every
edge in our networks is therefore *reconstructed*, not read off the data. This is the one
layer of the pipeline with no Gemini field behind it, and everything downstream inherits
whatever error it carries.

## Step 1 — which utterances are in play

An utterance enters the linkograph if any of its codes is one of 11 idea-move codes
(deck slide 45). Two are **node** codes that open a thread — `proposes_new_idea`,
`frames_shared_problem`. Nine are **move** codes that act on something already said —
`extends_existing_idea`, `combines_ideas`, `synthesizes_contributions`,
`connects_methods`, `critiques_or_challenges`, `raises_concern`,
`resolves_contradiction`, `redirects_idea`, `returns_to_earlier_idea`.

A turn can carry up to three codes, so it can be a node and a move at once (~3% are).
Turns carrying none of the 11 — coordination, knowledge-sharing, relational talk, roughly
60% of the transcript — are not in the linkograph at all.

Fields used per utterance: `speaker` (who), `timestamp` (order), each code's `evidence`
(what was said) and `explanation` (Gemini's prose account of what the person was doing).

## Step 2 — resolving each move to a target

For every move code, candidate targets are all **earlier** utterances that carry any
idea-move code. Three rules are tried in priority order, and every link records which one
fired, in a field called `basis`:

**1. `named_speaker`.** The code's `explanation` names another participant — "Michael
builds directly on Chong's idea of using electrochemistry with enzymes." Link to that
person's most recent prior idea utterance. Matching is on first names against the
session's own speaker list, excluding the speaker themselves. **Fires for 21% of moves;
~85% correct.**

**2. `on_table_idea`.** No name given. Take the move's own words and score them against
each idea in the chunk's `session_state.ideas_currently_on_table`; take the best-matching
live idea, then link to the earliest prior utterance whose words best match that same
idea. Token overlap, stopwords removed. **57% of links; ~75% correct.**

**3. `nearest_prior`.** Neither matched. Link to the most recent prior idea utterance.
This is a positional guess. **16% of links; ~65% correct.**

A move with no prior idea utterance to attach to produces no edge and is recorded as
unresolved (1 of 438 in the ten-session run).

The linker is deterministic and rule-based. No LLM is involved in the linking step — the
same input always produces the same links, and every link can be traced by hand.

## Step 3 — links become network edges

Each link is `{from, to, move_name, basis, links_to_name}`, where `from` is the move and
`to` is the earlier idea. Map both ends to their speaker:

- **Different speakers** → add or increment an undirected edge between them. Edge weight
  is the number of links between that pair; edge colour is whichever type dominates,
  additive (extend/combine/synthesize/connect) or structural (critique/redirect/concern).
- **Same speaker** → excluded from the network and counted separately as the **self-link
  index**. Someone building on their own earlier idea is real idea work but it is not
  collaboration, and folding it into the network would inflate a person's centrality for
  talking to themselves.

Every speaker with at least one idea utterance becomes a node, including speakers with no
cross-speaker link — an isolated node is informative.

## What this method cannot do

**One target per move.** Real linkography lets a move link back to several earlier moves.
Ours picks one. This caps the link ratio near 1 and under-represents fan-in — it is why
our "depth" is 0.68–0.94 while Liu's, which allows multi-links, is 2.38 and 4.25.

**Link quality is capped by explanation quality.** The strongest rule only fires when
Gemini's prose happens to name someone. It does so for 21% of moves (see `NAME_AUDIT.md`),
so 79% of links rest on the weaker two rules.

**Direction is dropped at the network level.** Links are directional (move → earlier
idea); edges are not. Eight links between two people could be mutual building or one
person repeatedly building on the other.

**Node identity is only as good as the speaker labels.** One person written two ways
becomes two nodes. Two such cases were found and merged explicitly; see the
`speaker_aliases` block in `scale_sessions.json`.

## How to check any of it

`scale10/link_basis_log.csv` has one row per link across all ten sessions — which rule
fired, whether a name was extractable, whether it crossed a chunk boundary, and both ends'
quoted evidence and explanation. Any edge on any figure can be traced back to the specific
utterances behind it without opening the code.

Per-session worked examples are in `scale10/transcript_evidence/evidence_<LABEL>.md`.
