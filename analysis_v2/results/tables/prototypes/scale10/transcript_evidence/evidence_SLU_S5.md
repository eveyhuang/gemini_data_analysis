# Transcript evidence — SLU_S5

Session `2021_06_11_SLU_S5` · 56 idea-bearing utterances · 47 links. Quoted text is the `evidence` field of the Gemini code — the same source deck slide 44 quotes from.

## A. Longest idea thread — confirms the linkograph

| # | Speaker | Text | Gemini code | Links to | How the link was resolved |
|---|---|---|---|---|---|
| 27 | Maggie Osburn | “one of the biggest battles we fight is sort of the abiotic system mimicking the biological system in terms of the isotopic fractionations” | `frames_shared_problem` | — (new idea) | — |
| 32 | Amanda Stockton | “connectivity and patterns of connectivity that leads to molecules and then perhaps to spatial distributions” | `extends_existing_idea` | #27 (Maggie Osburn) | matched an idea on the table |
| 90 | Jay Forsythe | “I think it'll be really difficult to assign... when you have these really complex mixtures” | `raises_concern` | #32 (Amanda Stockton) | matched an idea on the table |
| 91 | Andro Rios | “there are small reactive organic compounds in the absence of life would just degrade very quickly... that would be very exciting to expand with that…” | `proposes_new_idea, extends_existing_idea` | #90 (Jay Forsythe) | explanation named the person |
| 94 | Ziming Yang | “I think that's really challenging for remote sensing.” | `raises_concern` | #91 (Andro Rios) | fallback — most recent prior idea |

_5 utterances, Maggie Osburn → Ziming Yang. This is the session's deepest build-chain (the `longest_chain` metric)._

_0 of these steps are the same speaker building on their own earlier idea. Those are real linkograph links; they are excluded only from the speaker NETWORK, where they are counted separately as the self-link index. 1 step(s) were resolved by the `nearest_prior` fallback — the weakest tier (~55% correct), worth reading closely._

## B. Speaker edges — confirms the network

| Pair | Links | Who built on whom | Their words | Building on | How resolved |
|---|---|---|---|---|---|
| Andro Rios ↔ Maggie Osburn | 4 | Andro Rios (`extends_existing_idea`) | “a continuity of precursor molecules so that you see that same isotopic signature in the same precursor…” | Maggie Osburn: “one of the biggest battles we fight is sort of the abiotic system mimicking the biological system in terms of…” | matched an idea on the table |
| Eliza Kempton ↔ Maggie Osburn | 4 | Eliza Kempton (`extends_existing_idea`) | “thinking about trends over many objects... what would be the smoking gun of life if you could see a…” | Maggie Osburn: “one of the biggest battles we fight is sort of the abiotic system mimicking the biological system in terms of…” | matched an idea on the table |
| Maggie Osburn ↔ Morgan Raven | 4 | Morgan Raven (`extends_existing_idea`) | “I mean even the Martian sulfur isotopes right it really drives home that you can get these really large…” | Maggie Osburn: “one of the biggest battles we fight is sort of the abiotic system mimicking the biological system in terms of…” | explanation named the person |
| Morgan Raven ↔ Tori Hoeler | 4 | Tori Hoeler (`synthesizes_contributions`) | “the extent of recycling of resources is actually a characteristic of a biological system that doesn't exist…” | Morgan Raven: “the offset in sulfur isotopes between those two can tell you something... creative utilization and…” | explanation named the person |
| Amanda Stockton ↔ Andro Rios | 3 | Amanda Stockton (`synthesizes_contributions`) | “to kind of come back to what Andro was saying... we would expect to see more enrichment of lighter isotopes” | Andro Rios: “a continuity of precursor molecules so that you see that same isotopic signature in the same precursor…” | explanation named the person |
| Eliza Kempton ↔ Tori Hoeler | 3 | Tori Hoeler (`synthesizes_contributions`) | “take a statistical view of things... and I see it evolving in the direction that you guys are describing…” | Eliza Kempton: “observing many planets... are there trends that one could look for if if one had the option of observing 100…” | matched an idea on the table |

_Edge weight is the number of cross-speaker links between that pair; the row shows one of them in full. Self-links are excluded from the network and counted separately as the self-link index._
