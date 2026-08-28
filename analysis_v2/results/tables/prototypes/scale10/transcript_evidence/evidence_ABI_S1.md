# Transcript evidence — ABI_S1

Session `2021_05_21_ABI_S1` · 50 idea-bearing utterances · 34 links. Quoted text is the `evidence` field of the Gemini code — the same source deck slide 44 quotes from.

## A. Longest idea thread — confirms the linkograph

| # | Speaker | Text | Gemini code | Links to | How the link was resolved |
|---|---|---|---|---|---|
| 26 | Carolyn Bayer | “It strikes me that that's a common strategy, right? Sacrificing spatial resolution to improve temporal resolution.” | `synthesizes_contributions` | — (new idea) | — |
| 27 | Brian Pogue | “And can you have both? Can you have high spatial, high temporal?” | `synthesizes_contributions` | #26 (Carolyn Bayer) | matched an idea on the table |
| 34 | Brian Pogue | “in the microscopy world, this is where it's done a lot” | `extends_existing_idea` | #27 (Brian Pogue) | matched an idea on the table |
| 37 | Katharine White | “I would have no idea in terms of that” | `raises_concern` | #34 (Brian Pogue) | fallback — most recent prior idea |
| 41 | Katharine White | “how well those work off the shelf can vary quite significantly” | `critiques_or_challenges` | #37 (Katharine White) | fallback — most recent prior idea |

_5 utterances, Carolyn Bayer → Katharine White. This is the session's deepest build-chain (the `longest_chain` metric)._

_2 of these steps are the same speaker building on their own earlier idea. Those are real linkograph links; they are excluded only from the speaker NETWORK, where they are counted separately as the self-link index. 2 step(s) were resolved by the `nearest_prior` fallback — the weakest tier (~55% correct), worth reading closely._

## B. Speaker edges — confirms the network

| Pair | Links | Who built on whom | Their words | Building on | How resolved |
|---|---|---|---|---|---|
| Brian Pogue ↔ Katharine White | 5 | Katharine White (`raises_concern`) | “I would have no idea in terms of that” | Brian Pogue: “in the microscopy world, this is where it's done a lot” | fallback — most recent prior idea |
| Brian Pogue ↔ Shannon Quine | 3 | Shannon Quine (`synthesizes_contributions`) | “how do we have a molecular dynamics situation for imaging data” | Brian Pogue: “And can you have both? Can you have high spatial, high temporal?” | matched an idea on the table |
| Brian Pogue ↔ Brian Spring | 2 | Brian Spring (`extends_existing_idea`) | “I keep thinking of MRI or PET, something guiding a microscopic tool or photoacoustic tool.” | Brian Pogue: “CT is useless, you know. It gives you great images, but it's really insensitive to molecules.” | explanation named the person |
| Brian Pogue ↔ Josh Brake | 2 | Brian Pogue (`extends_existing_idea`) | “The thing that came to my mind was sort of palm and storm microscopy, right, where you've got probes that…” | Josh Brake: “I wonder if there's a hybrid modality where you have a front end which is maximally performant in space and…” | matched an idea on the table |
| Carolyn Bayer ↔ Katharine White | 2 | Carolyn Bayer (`extends_existing_idea`) | “a way to approach this might be to identify the imaging modality that currently does the best job of scaling” | Katharine White: “thinking about how you could do multi-modal imaging to identify a larger spatial or longer temporal scale” | matched an idea on the table |
| Katharine White ↔ Katy Keenan | 2 | Katy Keenan (`extends_existing_idea`) | “That feels like an opportunity for some of the orthogonal imaging approaches also.” | Katharine White: “how to determine... by combining imaging modality analysis” | matched an idea on the table |

_Edge weight is the number of cross-speaker links between that pair; the row shows one of them in full. Self-links are excluded from the network and counted separately as the self-link index._
