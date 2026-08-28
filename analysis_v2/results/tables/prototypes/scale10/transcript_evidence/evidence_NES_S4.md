# Transcript evidence — NES_S4

Session `2020_11_05_NES_S4` · 62 idea-bearing utterances · 55 links. Quoted text is the `evidence` field of the Gemini code — the same source deck slide 44 quotes from.

## A. Longest idea thread — confirms the linkograph

| # | Speaker | Text | Gemini code | Links to | How the link was resolved |
|---|---|---|---|---|---|
| 32 | Charles McCrory | “doing that as low temperature ammonia synthesis... low temperature hydrogenation for distributed ammonia” | `proposes_new_idea` | — (new idea) | — |
| 46 | Adam Holewinski | “since no single commodity chemical is gonna make a giant dent... the number was still small” | `raises_concern` | #32 (Charles McCrory) | matched an idea on the table |
| 47 | Charles McCrory | “I think the one stat that I remember from there was the decarbonizing polymerization chemistry” | `extends_existing_idea` | #46 (Adam Holewinski) | explanation named the person |
| 49 | Charles McCrory | “there's also the diol oxidation chemistry” | `extends_existing_idea` | #47 (Charles McCrory) | matched an idea on the table |
| 50 | Charles McCrory | “from some sort of oxalate CO2 CO2 coupling sort of intermediates.” | `extends_existing_idea` | #49 (Charles McCrory) | fallback — most recent prior idea |
| 51 | Sen Zhang | “Not just oxidation for fewer applications about the polymer building blocks, it can be also be reduced, right, and generate useful fuels directly” | `extends_existing_idea` | #50 (Charles McCrory) | fallback — most recent prior idea |

_6 utterances, Charles McCrory → Sen Zhang. This is the session's deepest build-chain (the `longest_chain` metric)._

_2 of these steps are the same speaker building on their own earlier idea. Those are real linkograph links; they are excluded only from the speaker NETWORK, where they are counted separately as the self-link index. 2 step(s) were resolved by the `nearest_prior` fallback — the weakest tier (~55% correct), worth reading closely._

## B. Speaker edges — confirms the network

| Pair | Links | Who built on whom | Their words | Building on | How resolved |
|---|---|---|---|---|---|
| Charles McCrory ↔ Katie Knowles | 6 | Katie Knowles (`synthesizes_contributions`) | “taking C1 to C2, C3, C4, like small things” | Charles McCrory: “build a little bit on what was said... Adam and Carlos are right on” | fallback — most recent prior idea |
| Charles McCrory ↔ Marta Hatzell | 5 | Marta Hatzell (`critiques_or_challenges`) | “looking at like direct nitrogen oxidation, so you can skip the nitrogen to ammonia I'm hesitant to say that…” | Charles McCrory: “doing that as low temperature ammonia synthesis... low temperature hydrogenation for distributed ammonia” | matched an idea on the table |
| Adam Holewinski ↔ Michael Nippe | 3 | Adam Holewinski (`extends_existing_idea`) | “I'd be interested in building on that one a little bit” | Michael Nippe: “we could be talking about electrochemical approaches to chiral synthesis” | matched an idea on the table |
| Adam Holewinski ↔ Charles McCrory | 2 | Adam Holewinski (`raises_concern`) | “since no single commodity chemical is gonna make a giant dent... the number was still small” | Charles McCrory: “doing that as low temperature ammonia synthesis... low temperature hydrogenation for distributed ammonia” | matched an idea on the table |
| Adam Holewinski ↔ Chong Liu | 2 | Adam Holewinski (`synthesizes_contributions`) | “So is the wish list chemistry methane to methanol at ambient conditions where you could do it at a…” | Chong Liu: “another thing I want to point out is where to get those methane. The flaring happens... because it's not…” | explanation named the person |
| Adam Holewinski ↔ Daniel Yawitz | 2 | Adam Holewinski (`raises_concern`) | “if the question is we really need to, negative emission is the only answer, then the beyond CO2 conversation…” | Daniel Yawitz: “tying it back to the negative emissions goal, I've been trying to kind of bucket what I'm hearing” | matched an idea on the table |

_Edge weight is the number of cross-speaker links between that pair; the row shows one of them in full. Self-links are excluded from the network and counted separately as the self-link index._
