# Transcript evidence — MND_S15

Session `2021_04_23_MND_S15` · 47 idea-bearing utterances · 38 links. Quoted text is the `evidence` field of the Gemini code — the same source deck slide 44 quotes from.

## A. Longest idea thread — confirms the linkograph

| # | Speaker | Text | Gemini code | Links to | How the link was resolved |
|---|---|---|---|---|---|
| 75 | Ying-Hui Chou | “Like traumatic brain injury, that could be a stress.” | `proposes_new_idea` | — (new idea) | — |
| 80 | Abhishek Shrivastava | “changes in oxygen are definitely one type of stress” | `extends_existing_idea` | #75 (Ying-Hui Chou) | matched an idea on the table |
| 118 | Lisa Ryno | “how would you interface that with one of these organoid models” | `combines_ideas` | #80 (Abhishek Shrivastava) | matched an idea on the table |
| 119 | Abhishek Shrivastava | “what if the more prominent phyla was not Bacteroidetes, what if it was Proteobacteria... that could easily be genetically modified” | `extends_existing_idea` | #118 (Lisa Ryno) | explanation named the person |
| 125 | Carolina Tropini | “I think it's key to find the right model for the question that is asked.” | `synthesizes_contributions` | #119 (Abhishek Shrivastava) | matched an idea on the table |

_5 utterances, Ying-Hui Chou → Carolina Tropini. This is the session's deepest build-chain (the `longest_chain` metric)._

_0 of these steps are the same speaker building on their own earlier idea. Those are real linkograph links; they are excluded only from the speaker NETWORK, where they are counted separately as the self-link index. 0 step(s) were resolved by the `nearest_prior` fallback — the weakest tier (~55% correct), worth reading closely._

## B. Speaker edges — confirms the network

| Pair | Links | Who built on whom | Their words | Building on | How resolved |
|---|---|---|---|---|---|
| Abhishek Shrivastava ↔ Carolina Tropini | 2 | Carolina Tropini (`critiques_or_challenges`) | “interestingly, a lot of the microbiota is not motile... a lot of the fiber degraders just do not have…” | Abhishek Shrivastava: “what does changes in the motility of the gut mean to the microbes? You're increasing the fluid flows” | matched an idea on the table |
| Abhishek Shrivastava ↔ Lisa Ryno | 2 | Lisa Ryno (`combines_ideas`) | “how would you interface that with one of these organoid models” | Abhishek Shrivastava: “changes in oxygen are definitely one type of stress” | matched an idea on the table |
| Abhishek Shrivastava ↔ Ying-Hui Chou | 2 | Abhishek Shrivastava (`extends_existing_idea`) | “changes in oxygen are definitely one type of stress” | Ying-Hui Chou: “Like traumatic brain injury, that could be a stress.” | matched an idea on the table |
| Aida Ebrahimi ↔ Ali Keshavarzian | 2 | Ali Keshavarzian (`critiques_or_challenges`) | “The first point, Aida, was the microbial associated, not host derived.” | Aida Ebrahimi: “when the host is under stress, we know a lot of biochemicals in our saliva for example, it is changing and we…” | explanation named the person |
| Ali Keshavarzian ↔ Ying-Hui Chou | 2 | Ali Keshavarzian (`extends_existing_idea`) | “vaginal environment is under tremendous amount of stress as well... continuous stress because of proximity” | Ying-Hui Chou: “Like traumatic brain injury, that could be a stress.” | matched an idea on the table |
| Carolina Tropini ↔ Ying-Hui Chou | 2 | Carolina Tropini (`extends_existing_idea`) | “I was thinking more on the cellular stress... is it I feel stressed out today or is it heat stress” | Ying-Hui Chou: “Like traumatic brain injury, that could be a stress.” | matched an idea on the table |

_Edge weight is the number of cross-speaker links between that pair; the row shows one of them in full. Self-links are excluded from the network and counted separately as the self-link index._
