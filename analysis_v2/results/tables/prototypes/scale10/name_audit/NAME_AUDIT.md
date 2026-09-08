# Missing-names audit

Every move code in every annotated session: **6,157 moves** across 157 sessions and 8 conferences.

## Headline

| Quantity | Count | Share |
|---|---|---|
| Explanation names someone (incl. the speaker) | 3,869 | 62.8% |
| Names someone **other** than the speaker — what rule 1 needs | 1,345 | 21.8% |
| `named_speaker` actually resolved the link | 1,263 | 20.5% |
| **Named but unusable** (no prior idea by that person) | 82 | 1.3% |

So Gemini fails to identify the target for **78.2%** of moves. That is not rare — it is the majority case, and it is why the `on_table_idea` and `nearest_prior` rules carry most of the link load.

## By conference

```
  conference  moves   names anyone   names other   rule fired  recoverable
  -----------------------------------------------------------------------
  2021SLU      992          61.4%         24.6%        24.1%            5
  2021MZT      967          61.9%         23.8%        23.2%            6
  2020NES      809          59.5%         26.2%        25.0%           10
  2021NES      768          61.8%         20.8%        19.1%           13
  2021CMC      762          69.8%         18.0%        16.3%           13
  2021MND      715          62.8%         22.7%        21.0%           12
  2022MND      595          59.7%         23.4%        21.5%           11
  2021ABI      549          67.2%         11.1%         8.9%           12
```

## By move type

```
  move code                    moves   names anyone   names other   rule fired  recoverable
  -----------------------------------------------------------------------------------------
  extends_existing_idea         2571          63.6%         27.1%        25.8%           33
  synthesizes_contributions     1296          74.5%         24.0%        21.8%           29
  raises_concern                 786          45.9%          2.9%         2.5%            3
  critiques_or_challenges        590          63.4%         19.2%        18.0%            7
  returns_to_earlier_idea        313          64.2%         27.5%        26.2%            4
  redirects_idea                 197          52.3%          5.1%         4.1%            2
  combines_ideas                 184          60.3%         26.6%        26.6%            0
  connects_methods               157          48.4%         14.6%        12.7%            3
  resolves_contradiction          63          68.3%         52.4%        50.8%            1
```

## By session — is low naming a tail or the norm?

- sessions measured: **157**
- median naming rate: **21.1%**
- worst quartile: **≤ 13.6%**  ·  best quartile: **≥ 27.3%**
- sessions where Gemini names the target in **under 10%** of moves: **28**
- sessions **above 30%**: **26**


---

## What this means

**It is common, not rare.** Gemini fails to identify whose idea is being built on for
**78% of moves**. Any policy that treats missing names as an edge case is wrong.

**Most of the apparent signal is self-reference.** 63% of explanations name *someone*, but
only 22% name someone *other* than the speaker. The 41-point gap is Gemini writing
"Jenny builds on her earlier point" — grammatically a name, useless as a target.

**There is almost nothing left on the table.** Only **1.3%** of moves name a usable person
that the linker failed to resolve. So improving rule 1's matching buys ~1 percentage
point. The signal genuinely is not in the annotation.

**It varies by move type far more than by conference.** `resolves_contradiction` names a
target 52% of the time; `raises_concern` does so **2.9%** of the time across 786 moves,
and `redirects_idea` 5.1%. Concern-raising is about content, not attribution — Gemini
describes what the worry is, never whose idea provoked it. Conference range is narrower
(ABI 11% is the outlier; everything else sits between 18% and 26%).

**Low-naming sessions are not a tail.** Median 21%; 28 of 157 sessions are under 10%.

## Recommended policy

**1. Do not exclude on naming rate.** Excluding would drop most of the corpus and would
correlate with move type — you would systematically lose sessions where people raise
concerns.

**2. Keep reporting the basis mix per session.** Already done. It is the honest way to
carry the uncertainty rather than hide it.

**3. Fix the fallback's failure mode, which is specific and not about names at all.**
Reading 32 sampled links across 8 conferences, both errors were the same: `nearest_prior`
links to the previous idea utterance *without checking whether it contains anything to
build on*. It attached a modelling proposal to "I'm a physicist but I don't buy that."

A length threshold is a tempting fix and a bad one — the shortest fallback targets include
both "It's really tough." (no content) and "it's going to require the FEL presumably" (a
real contribution). Only 5% of fallback targets are under 40 characters, so length would
catch few of the bad cases and some good ones.

**Better: let rule 3 abstain.** It currently always returns a link. Allowing it to return
nothing — recording the move as unresolved, which the pipeline already supports — trades a
small number of links for removing the tier most likely to be wrong. At the observed rate
that is ~16% of links, of which perhaps a third are wrong. Worth testing behind a flag and
comparing metrics before and after.

**4. The real fix is upstream.** If the annotation prompt asked Gemini to name what each
move builds on, this entire layer would mostly disappear — along with the basis tiers, the
fallback, and the uncertainty inherited by 8 of the 11 metrics. Expensive (re-annotation),
but it is the actual answer rather than a better guess.
