# Transcript evidence — MND_S5

Session `2022_04_07_MND_S5` · 51 idea-bearing utterances · 42 links. Quoted text is the `evidence` field of the Gemini code — the same source deck slide 44 quotes from.

## A. Longest idea thread — confirms the linkograph

| # | Speaker | Text | Gemini code | Links to | How the link was resolved |
|---|---|---|---|---|---|
| 72 | Faranak Fattahi | “are we in the phase where we should be screening for a larger set of molecules or metabolites or factors that could be coming from the microbiome.” | `proposes_new_idea` | — (new idea) | — |
| 86 | Erin Longbrake | “So to go back to Faranak's question, like what would be a model system to figure out if and how bacteria microbes impact let's just stick with the…” | `synthesizes_contributions, proposes_new_idea` | #72 (Faranak Fattahi) | explanation named the person |
| 146 | Rosa Krajmalnik-Brown | “maybe we can also add a bullet point of there are some lower throughput screening methods” | `extends_existing_idea` | #86 (Erin Longbrake) | matched an idea on the table |
| 167 | Iliyan Iliev | “But because you had the results of the open label, right? That wouldn't have happened without the results” | `critiques_or_challenges` | #146 (Rosa Krajmalnik-Brown) | explanation named the person |
| 169 | Iliyan Iliev | “And also like the institution will not like that because there is no... anything that you bring to the institution” | `raises_concern` | #167 (Iliyan Iliev) | fallback — most recent prior idea |
| 183 | Rosa Krajmalnik-Brown | “I think, um, we mentioned fungi at some point, and I don't I don't see it anywhere.” | `returns_to_earlier_idea` | #169 (Iliyan Iliev) | fallback — most recent prior idea |
| 190 | Erin Longbrake | “They don't like funding anything that's risky and there's a bias towards hypothesis... So who's going to fund the observations?” | `critiques_or_challenges` | #183 (Rosa Krajmalnik-Brown) | fallback — most recent prior idea |
| 192 | Erin Longbrake | “there's a fair amount of, um, snobbery when it comes to being NIH funded versus not NIH funded.” | `critiques_or_challenges` | #190 (Erin Longbrake) | fallback — most recent prior idea |

_8 utterances, Faranak Fattahi → Erin Longbrake. This is the session's deepest build-chain (the `longest_chain` metric)._

_2 of these steps are the same speaker building on their own earlier idea. Those are real linkograph links; they are excluded only from the speaker NETWORK, where they are counted separately as the self-link index. 4 step(s) were resolved by the `nearest_prior` fallback — the weakest tier (~55% correct), worth reading closely._

## B. Speaker edges — confirms the network

| Pair | Links | Who built on whom | Their words | Building on | How resolved |
|---|---|---|---|---|---|
| Iliyan Iliev ↔ Tom Mansell | 4 | Iliyan Iliev (`raises_concern`) | “there is a co-founder effect here, right? Because the bacteria and the fungi that have actually siderophores…” | Tom Mansell: “studies that models that involved soluble or insoluble chelators like EDTA right is a thing that's in food…” | explanation named the person |
| Erin Longbrake ↔ Faranak Fattahi | 3 | Erin Longbrake (`synthesizes_contributions`) | “So to go back to Faranak's question, like what would be a model system to figure out if and how bacteria…” | Faranak Fattahi: “are we in the phase where we should be screening for a larger set of molecules or metabolites or factors that…” | explanation named the person |
| Erin Longbrake ↔ Iliyan Iliev | 3 | Erin Longbrake (`critiques_or_challenges`) | “Would the molecule need to be appropriately sized though? I mean, in a sense, if you have inflammatory…” | Iliyan Iliev: “how big molecules can go depending on your genetic defect.” | matched an idea on the table |
| Erin Longbrake ↔ Rosa Krajmalnik-Brown | 3 | Erin Longbrake (`synthesizes_contributions`) | “it sounds plausible that certain microbes either make small metabolites that impact the barrier or make…” | Rosa Krajmalnik-Brown: “the brain barrier for me also it's distant... but I feel that the gut permeability is still a very important…” | matched an idea on the table |
| Iliyan Iliev ↔ Rosa Krajmalnik-Brown | 3 | Iliyan Iliev (`extends_existing_idea`) | “how big molecules can go depending on your genetic defect.” | Rosa Krajmalnik-Brown: “we have people that are focused on the chemicals, organoids, the gut, the brain, immunology.” | fallback — most recent prior idea |
| Andrew Feig ↔ Iliyan Iliev | 2 | Andrew Feig (`connects_methods`) | “so that would allow you to at least isolate that barrier separate from the intestinal barrier.” | Iliyan Iliev: “how big molecules can go depending on your genetic defect.” | explanation named the person |

_Edge weight is the number of cross-speaker links between that pair; the row shows one of them in full. Self-links are excluded from the network and counted separately as the self-link index._
