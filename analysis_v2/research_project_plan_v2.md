# Multimodal AI Annotation of Team Meetings: Research Project Plan (v2)

---

## Theoretical framing and positioning

This project sits at the intersection of team science and multimodal AI, motivated by a foundational tension in organizational research: we have known for decades that micro-interactions predict team outcomes, yet have lacked instruments capable of measuring those interactions at scale.

The conceptual anchor is Gottman and Levenson's (2000) finding that fifteen minutes of observing a couple's interaction predicted divorce with over 90% accuracy across fourteen years — not from what couples said about their relationship, but from specific observable micro-behaviors: contempt, defensiveness, stonewalling, and criticism visible in facial expressions, vocal tone, and conversational dynamics. This insight — that relational futures are encoded in present micro-behaviors — extends to teams. Woolley et al. (2010) showed that collective intelligence, a property predicting group performance across diverse tasks, correlates more strongly with equality of conversational turn-taking and members' social sensitivity than with individual intelligence. Jung (2016) demonstrated that five-minute observations predicted team performance weeks later with 91% accuracy. The pattern is consistent: micro-interactions contain macro-predictive signal.

Yet team science has continued to measure teams primarily through retrospective surveys — capturing what teams articulate about themselves rather than what their interactions actually reveal. The reason is infrastructure, not ignorance. Behavioral coding required trained experts investing ten hours per hour of footage. Reliability demanded multiple coders per interaction. The labor economics of careful observation constrained what could be studied.

That constraint is lifting. Multimodal large language models can now process video, audio, and text simultaneously at approximately one dollar per hour of footage — compared to hundreds of dollars for equivalent expert human coding. This is not incremental improvement; it is the arrival of a new instrument that may do for team science what the microscope did for biology.

This project operationalizes that argument. The Scialog conference setting — scientists from diverse disciplines, meeting virtually, with team formation and grant funding as concrete downstream outcomes — provides an ideal naturalistic test case. Scialog meetings combine the intellectual dynamics of interdisciplinary science (cross-disciplinary bridging, idea generation and convergence) with the social dynamics of nascent team formation (commitment signals, relational climate, participation equity). Both dimensions are theoretically grounded and measurable with multimodal AI.

The project contributes to three streams of the broader research agenda outlined in the companion paper:

**The thin slice threshold question.** How much observation is needed to reliably predict outcomes? The chunk-level architecture (5–10 minute segments, beginning/middle/end temporal structure) allows systematic mapping of the prediction curve — does beginning-segment behavior suffice, or does the full session add signal? This directly tests the Gottman/Jung premise in a team science context.

**The building versus blocking question.** Whether team members integrate others' contributions into their own thinking is central to theories of collaborative cognition (Bales' interaction process analysis; collective intelligence literature) but has been notoriously difficult to code reliably at scale. The utterance-level behavioral codebook operationalizes this directly through Idea Management, Integration Practices, and Evaluation Practices codes.

**The affective trajectory question.** Current measurement misses how affect evolves within and across meetings — when tension emerges, whether it resolves, how enthusiasm trajectories relate to team formation. Vocal enthusiasm, shared affect, and collective behavioral responsiveness across the beginning/middle/end temporal structure directly address this.

The article's core claim — that multimodal AI offers team science new resolution into participation architecture, affective dynamics, and temporal patterns — is the theoretical claim this project empirically tests and validates.

---

## What this project is trying to answer

This project asks four core questions:

1. Can a multimodal LLM pipeline reliably annotate behavioral, vocal, and affective signals from recorded virtual team science meetings at scale, with demonstrated agreement against trained human expert coders — and which categories of behavior are most reliably measured?
2. Which AI-annotated behavioral patterns are associated with team formation and funding outcomes — specifically, which features from the participation architecture, building/blocking, and affective trajectory dimensions emphasized in team science theory show the strongest signal?
3. How predictive are these behavioral features in properly validated ML models with leave-one-out cross-validation and conference-level clustering controls — and does predictive performance match the "thin slice" premise that early observation windows carry disproportionate signal?
4. Does the temporal trajectory of behavioral features (beginning → convergence → commitment) predict outcomes beyond any static snapshot, consistent with the theoretical claim that team formation is a dynamic process?

---

## Targeted outlet and audience

1. Nature Machine Intelligence
2. PNAS

The framing for both outlets positions this as a methodological contribution (a validated, scalable pipeline for behavioral measurement) with substantive team science findings. For Nature MI, the emphasis is on the AI methodology, reliability validation, and the multimodal-over-transcript contribution. For PNAS, the emphasis is on the team science findings: which behaviors predict interdisciplinary team formation and what that reveals about the micro-foundations of scientific collaboration.

---

## Theoretical grounding for annotation categories

The codebook and annotation targets are not arbitrary — each maps to a construct from the team science literature with established theoretical and predictive significance.

**Participation architecture** (speaking time Gini, dominant speaker flag, turn-taking): Woolley et al. (2010) established equality of conversational turn-taking as a key predictor of collective intelligence. Pentland's sociometric work (2012) showed that energy, engagement, and exploration in interaction — measurable from speaking patterns — predict team performance. The Gini coefficient of speaking time is a direct operationalization of participation equality.

**Building and blocking behaviors** (Idea Management, Integration Practices, Evaluation Practices codes): Bales' (1950) Interaction Process Analysis established the idea-building/blocking distinction as foundational. The specific subcodes in the pipeline (proposes_new_idea, extends_existing_idea, synthesizes_contributions, supports_or_validates vs. critiques_or_challenges) operationalize this at utterance level.

**Psychological safety markers** (Relational Climate codes, hesitation_flag, interruption quality): Edmondson's (1999) work showed that psychological safety — the shared belief that the team is safe for interpersonal risk-taking — is the strongest predictor of team learning and performance. Observable behavioral markers include how dissent is received (competitive vs. collaborative interruptions), whether speakers hesitate before critical claims (hesitation_flag), and relational climate behaviors (manages_tension, expresses_appreciation).

**Idea trajectory (divergent → convergent)**: Group creativity research (Nijstad & Stroebe, 2006) shows that effective idea generation requires a divergent phase before convergence. Teams that never shift from divergent to convergent fail to coalesce around a direction. The chunk-level `idea_trajectory` code and its evolution across the beginning/middle/end of sessions tests this premise directly.

**Cross-disciplinary bridging**: Specific to the Scialog context, but grounded in the science of science literature (Uzzi et al., 2013 on atypical combinations; Wuchty et al., 2007 on team science). Team formation in interdisciplinary science is driven by perceived complementarity of frameworks, not just affinity. Explicitly bridging disciplinary vocabularies is a behavioral indicator of perceived complementarity.

**Commitment signals** (explicit_commitment_signal, future-oriented language codes): These are proximal behavioral precursors of the outcome of interest. Their presence is the most direct behavioral analog to team formation intent measurable from a single meeting.

**Affective dynamics** (vocal_enthusiasm, shared_affect, nod_count, audible_backchannel): Barsade (2002) showed that emotional contagion in groups affects cooperation and task performance. Collective positive affect — observable through synchronized smiling, shared laughter, and high vocal enthusiasm — predicts group cohesion. These are the team-level operationalizations of affective trajectory.

**Problem specificity and decision crystallization**: Grant success depends on the intellectual sharpness of the proposed project. A meeting that ends with a vague shared interest produces a different trajectory than one that ends with a specific question and approach. The 1–4 problem specificity and decision crystallization ratings track this arc across the session and are expected to be among the strongest predictors of grant outcomes specifically (not just team formation).

**Complementarity recognition**: Team formation in science is driven by perceived complementarity — the sense that "you have what I lack." Lazarsfeld and Merton's (1954) work on homophily and complementarity in relationships, extended to scientific collaboration by Uzzi et al. (2013), suggests that atypical combinations — including of complementary expertise — produce the highest-impact science. Explicit complementarity recognition and skill gap identification operationalize this at the behavioral level.

**Shared vision and pronoun framing**: Conversation analysis research (Goodwin, 1981; Clark & Schaefer, 1989) documents how joint projects emerge through shifts in linguistic frame — from individual ownership ("my work," "your approach") to collective ownership ("our project," "we could test this"). The pronoun shift from individual to joint framing is a reliable behavioral marker of the moment team formation crystallizes.

**Ambition level and intellectual risk-taking**: Uzzi et al.'s (2013) "atypical combinations" finding established that novel cross-disciplinary combinations predict both higher impact and higher failure rates in science. Scialog specifically funds high-risk, high-reward research. The ambition level code (novel_combination, paradigm_challenging) and the risk acknowledgment + enthusiasm signal operationalize this directly. Teams that propose safe, incremental work within a Scialog context are less competitive regardless of their interpersonal dynamics.

**Setback response and psychological safety**: Edmondson's (1999) work showed that psychological safety — observable in how teams respond to disagreement and obstacles — is the strongest predictor of team learning. The setback response subcodes (explores vs. defends vs. redirects vs. accepts_and_builds) and dissent_response_quality provide direct behavioral operationalization of this construct, distinct from stated beliefs about safety measured by surveys.

**Interpersonal rapport signals** (personal disclosure, laughter quality, energy matching): Pentland's (2012) sociometric research showed that energy, engagement, and exploration in social interaction — partially captured through vocal synchrony and positive affect — predict team performance. Personal disclosure signals trust and genuine engagement. Appreciative laughter — responding to an idea as clever or surprising — is a more specific indicator of intellectual rapport than social laughter. Energy matching (vocal enthusiasm synchrony across speakers) is the team-level analog of emotional contagion.

**Absence signals** (no_convergence_flag, parallel_monologue_index, unresolved_tension_flag): Some of the most predictive signals may be what doesn't happen. Teams that never shift from divergent exploration to convergent commitment, that take turns presenting their own work without building on others', or that leave dissent unresolved are less likely to form and less likely to produce competitive proposals. These negative flags operationalize failure modes that are invisible to survey-based measurement.

**Grant context signals** (funding_awareness, prior_relationship, broader_significance): Teams that demonstrate awareness of funding mechanisms, prior familiarity with each other's work, and the ability to articulate why their proposed research matters to reviewers are better positioned to convert collaboration intent into funded proposals.

---

## Key methodological improvements over preliminary pipeline

The preliminary pipeline had several structural limitations that this redesign corrects:

- **No inter-rater reliability validation**: The new pipeline includes a mandatory human expert validation stage covering ~15–20% of chunks. Human-AI agreement (Cohen's kappa, ICC) is computed per annotation category before any feature is included in downstream models.
- **Cherry-picked split reporting**: The prior pipeline reported best-split AUC across 20 random splits. The new pipeline uses leave-one-session-out or leave-one-conference-out cross-validation and reports mean ± SD of performance metrics.
- **Conference-level clustering ignored**: All regression models now include conference fixed effects or mixed-effects structure with conference as a random intercept.
- **Narrow context window (±3 utterances)**: Utterance-level annotation now uses the full chunk transcript as context (5–10 min segments), not a ±3 utterance window.
- **Behavioral coding and quality scoring consolidated into one pass**: Quality is captured where it matters through subcodes, chunk-level ratings, and a lightweight `idea_quality` field on three categories — eliminating a redundant second inference pass without losing predictive signal.
- **Missing theoretically motivated features**: New annotation targets include idea trajectory, cross-disciplinary bridging, explicit commitment signals, behavioral responsiveness (replacing gaze-direction engagement), vocal enthusiasm, and collaborative artifact manipulation — all grounded in prior team science literature.
- **Invalid gaze-direction engagement measure**: In virtual Zoom meetings, participants face their camera regardless of where their attention is directed, making "looking toward the speaker" an uninterpretable signal. The new pipeline replaces this with Zoom-valid behavioral proxies for engagement: camera-on status, visible off-screen distraction, head nods, facial responsiveness, shared affect, and backchannel vocalizations — signals that are genuinely observable and interpretable in a video-conferencing context.

---

## Data context

- **N = 157 meeting sessions** across multiple Scialog conferences.
- Each session is a 60–90 minute recorded Zoom meeting among scientists from diverse disciplinary backgrounds.
- Sessions have already been **chunked into 5–10 minute video segments** for processing.
- Each session has associated **outcome variables**: `num_teams`, `num_funded_teams`, `has_teams` (binary), `has_funded_teams` (binary).
- Raw files available per session: `.mp4` video chunks, `.vtt` transcript files, participant roster CSVs, outcome spreadsheets.
- Conference membership is known for all sessions (conference = a clustering variable to be controlled).

---

## Pipeline overview


Stage 0: Preprocessing & Chunk Registry
        ↓
Stage 1: Multimodal Annotation 
        ↓
Stage 2: Schema Validation & Quality Flagging
        ↓
Stage 3: Human Expert Validation & Agreement Computation
        ↓
Stage 4: Feature Engineering
        ↓
Stage 5: Descriptive Analysis & Visualization
        ↓
Stage 6: Inferential Modeling (Regression + ML)
        ↓
Stage 7: Temporal Segment Analysis
        ↓
Stage 8: Reproducibility Packaging & Zenodo Deposit

        ↓
Stage 9: Reproducibility Packaging


Each stage corresponds to one or more notebooks or scripts described below.

---

## Annotation details

### Layer 1: Chunk-level summary annotations (per 5–10 min segment)

These are holistic assessments made by the model over the full chunk after processing all audio, video, and transcript content.

**1.1 Participation structure**
- For each participant visible in the chunk, estimate percentage of total speaking time.
- Compute and store raw per-participant speaking seconds from the model output.
- Derived downstream: Gini coefficient of speaking time within chunk.
- Flag if any single participant accounts for >50% of speaking time (`dominant_speaker_flag`: Yes/No).

**1.2 Idea trajectory**
- Classify the chunk's primary mode as one of: `divergent` (generating, exploring, brainstorming), `convergent` (narrowing, synthesizing, deciding, committing), `procedural` (logistics, housekeeping), or `ambiguous`.
- Provide a one-sentence justification.
- This is the model's holistic read of the chunk's epistemic function.

**1.3 Collective behavioral responsiveness**
- Rate (1–4 scale) the overall behavioral responsiveness of non-speaking participants throughout the chunk, based on signals that are genuinely observable in Zoom video.
- **Important**: Do NOT use gaze direction as a signal of engagement in Zoom meetings. In a video-conference context, all participants face their cameras regardless of where their attention is actually directed — "looking at the screen" is therefore not a valid indicator of engagement. Only code signals that are unambiguously observable.
- Valid signals to observe (listed from highest to lowest interpretive confidence):
  - **Head nods** from non-speakers (clearly observable, strong engagement signal)
  - **Facial expressions**: smiles, raised eyebrows, frowning, visible reactions to what is being said
  - **Backchannel vocalizations** audible from non-speakers: "mm-hmm", "yeah", laughter, brief affirmations
  - **Camera-on status**: participants who turn their camera off mid-chunk (flag this)
  - **Visible off-screen distraction**: participant is visibly looking far away from their screen (not just a brief glance), using a phone in frame, or appears to be talking to someone off-camera
  - **Active leaning in or out**: visible postural shift toward or away from the camera
- Rating rubric:
  - 1 = Most visible non-speakers show no behavioral responsiveness: no nods, no visible facial reactions, no audible backchannels; or multiple cameras off; or visible off-screen distraction.
  - 2 = Mixed: some participants show responsiveness (nods, brief smiles), others do not. Inconsistent across the chunk.
  - 3 = Most visible non-speakers show consistent responsiveness: regular nods, audible backchannels, visible expressions of engagement.
  - 4 = Strong collective responsiveness throughout: frequent nods, shared laughter or smiling, active backchannel vocalizations from multiple participants, visible enthusiasm.
- This field requires simultaneous audio + video processing and must be set to null in transcript-only mode.

**1.4 Cross-disciplinary bridging**
- Binary flag (`Yes/No`): Did any participant in this chunk explicitly connect their disciplinary framing to another participant's perspective? (e.g., "What you're calling X is essentially what we in Y refer to as Z", or "In my field we approach this differently — your approach is complementary because…")
- If Yes: note which speaker made the bridging move and a brief phrase describing it.

**1.5 Explicit commitment signals**
- Binary flag (`Yes/No`): Did any participant express interest in continuing to work together, propose a specific future collaboration, reference a next step that involves multiple participants, or suggest forming a team?
- If Yes: note the speaker, the chunk index, and a verbatim quote (≤20 words) of the commitment signal.
- This is a proximal behavioral indicator of the outcome of interest.

**1.6 Collaborative artifact engagement**
- If a screen is shared during this chunk: binary flag for whether participants actively interact with the artifact (e.g., live edits, pointing, referencing specific elements of a document or diagram in their speech).
- `screenshare_active` (Yes/No), `artifact_interaction` (Yes/No/NA), brief description of artifact type (document, diagram, data, slides, code, other).

**1.7 Problem specificity level**
- Rate (1–4) how specific the problem the group is working on is by the end of this chunk.
  - 1 = Topic-level only ("we're both interested in X")
  - 2 = Domain narrowed but no specific question ("something about X in the context of Y")
  - 3 = A specific research question articulated ("can X mechanism explain Y phenomenon in Z system?")
  - 4 = A specific question with hypothesis and approach ("we think X causes Y via Z, testable by W")
- If the chunk is procedural and no research problem is discussed, write `NA`.
- Provide a one-sentence justification referencing the specific language in the transcript.

**1.8 Decision crystallization level**
- Rate (1–4) how crystallized the group's sense of a shared direction is by the end of this chunk.
  - 1 = No shared direction; still at open exploration
  - 2 = Shared interest identified but no specific project
  - 3 = A specific project idea named and agreed upon by at least two participants
  - 4 = A specific project with at least two of: a research question, an approach, a timeline, a division of roles, or a named next step
- This is especially important for the **final chunk** of each session, where the value directly predicts team formation. Provide a one-sentence justification.

**1.9 Explicit complementarity recognition**
- Binary flag (`Yes/No`): Did any participant explicitly articulate how their expertise or approach *complements* another's — not just that the fields are related, but that the combination is more capable than either alone?
- Examples that qualify: "I could never do the experimental side — that's exactly what I'd need someone like you for." / "My modeling work has been missing exactly the kind of empirical grounding you're describing."
- Examples that do NOT qualify: Simply noting that you work on related topics, or agreeing that a connection exists.
- If Yes: note speaker and a verbatim phrase (≤20 words).

**1.10 Skill or resource gap identification**
- Binary flag (`Yes/No`): Did any participant identify a specific methodological, knowledge, or resource gap in the proposed project and implicitly or explicitly connect that gap to what another participant in the room could provide?
- This is the "you're the missing piece" signal. Code only explicit gap-to-person connections, not general acknowledgments that the project is hard.
- If Yes: describe the gap and who was identified as filling it (1 sentence).

**1.11 Shared vision indicator**
- Binary flag (`Yes/No`): Did the conversation in this chunk shift from participants describing their *own separate* work toward discussing a *shared* project that belongs jointly to the group? Key signal: participants use "our," "we," or "together" when referring to the proposed work (not just social politeness).
- This is distinct from commitment signals (which are explicit proposals to collaborate) — shared vision can be implicit in how the project is described.
- If Yes: note the phrase or exchange that most clearly marks the shift.

**1.12 Pronoun shift flag**
- Binary flag (`Yes/No`): Did this chunk show a notable shift from individual framing ("my work," "your work," "in my field") toward joint framing ("our idea," "we could," "together we") when discussing the proposed research?
- Code `Yes` only when the shift is observable within this chunk — not simply because joint language exists. If joint language has been present since the beginning of the session, do not re-flag.

**1.13 Personal disclosure**
- Binary flag (`Yes/No`): Did any participant share something personal — a research frustration, a career aspiration, a domain passion, a past failure, an unexpected finding — that goes beyond professional role presentation?
- Personal disclosure signals genuine engagement and trust. Do not flag polite self-introductions as disclosure. Only flag disclosures that are personally revealing or emotionally candid.

**1.14 Ambition level of proposed ideas**
- Classify the most ambitious idea proposed in this chunk (if any):
  - `incremental`: extends existing work in a straightforward or expected direction
  - `novel_application`: applies well-established methods to a new domain
  - `novel_combination`: combines two fields, methods, or frameworks in a way neither field has previously done
  - `paradigm_challenging`: questions a foundational assumption of one or both fields involved
  - `not_applicable`: no specific research idea was proposed in this chunk
- This directly operationalizes the Uzzi et al. (2013) "atypical combinations" construct that predicts high-impact science.

**1.15 Risk acknowledgment with enthusiasm**
- Binary flag (`Yes/No`): Did any participant explicitly acknowledge that the proposed project is risky, uncertain, or hard — AND respond to that acknowledgment with excitement or positive affect rather than retreat?
- The combination of risk acknowledgment + positive affect toward the risk is the key signal. Either alone does not qualify.
- If Yes: provide a verbatim phrase (≤20 words) capturing the risk + enthusiasm combination.

**1.16 Dissent response quality**
- When any participant expresses disagreement, raises a concern, or proposes a contrarian view in this chunk, how does the group respond?
  - 1 = Dismissive or defensive: speaker is interrupted, idea is dropped without engagement, or response signals the dissent was unwelcome.
  - 2 = Neutral: dissent is acknowledged but not deeply engaged with.
  - 3 = Curious and exploratory: dissent is met with follow-up questions, elaboration requests, or genuine engagement. Disagreement is treated as information.
  - `NA` = No dissent or contrarian view was expressed in this chunk.
- Operationalizes Edmondson's (1999) psychological safety construct at the behavioral level.

**1.17 Laughter quality**
- If laughter or shared humor occurred in this chunk, classify its primary function:
  - `tension_release`: laughter occurring at or immediately after a moment of disagreement, awkwardness, or challenge — signals relief
  - `shared_humor`: laughter in response to a joke or playful remark — signals social warmth
  - `appreciative`: laughter or amusement in direct response to an idea being clever, surprising, or elegant — signals intellectual rapport
  - `social_lubricant`: light laughter with no clear trigger — background social comfort
  - `none`: no laughter occurred in this chunk
- Appreciative laughter is the most predictive of intellectual engagement and team formation.

**1.18 Meeting structure quality**
- Rate the degree to which this chunk has an implicit or explicit structure:
  - `unstructured`: conversation is associative, topic jumps without apparent plan
  - `loosely_structured`: participants have a general shared sense of what they're doing, but no explicit agenda
  - `structured`: participants explicitly reference phases, topics to cover, or what has been accomplished; discussion is organized

**1.19 Funding and grant awareness**
- Binary flag (`Yes/No`): Did any participant mention a specific funding mechanism, program priority, grant deadline, review criterion, or funding agency relevant to the proposed work?
- This signals grant-writing sophistication and converts abstract collaboration interest into concrete proposal intent.
- If Yes: note the specific funding reference (1 sentence).

**1.20 Prior relationship signal**
- Binary flag (`Yes/No`): Did any participant mention prior familiarity with another participant — having read their work, met at a conference, collaborated before, or known of each other?
- Prior familiarity reduces coordination costs and predicts both team formation and proposal quality.
- If Yes: note the nature of the prior connection (1 sentence).

**1.21 Notable observation**
- A brief (1–3 sentence) free-text note on anything surprising, unusual, or potentially important that occurred in the chunk that is not captured by any other field above but is relevant to predicting team formation or grant proposal quality.
- Examples: an unexpected moment of laughter that broke tension, a participant abruptly reversing position, a striking personal anecdote, an unusually creative leap, or a visible moment of disengagement that seems significant.
- Set to `None` if nothing stands out.

---

### Layer 2: Utterance-level behavioral codes (single pass — behavior + selective inline quality)

Applied to each speaker turn within the chunk. The full chunk transcript is provided as context.

**Why a single pass:** The original design used a separate quality-scoring pass to prevent anchoring bias. That separation is no longer necessary because:
- Most quality distinctions are already captured by subcode choice (e.g., `setback_response_accepts_builds` vs. `setback_response_defends` encodes a quality difference directly)
- The expanded chunk-level summary (problem_specificity_level, decision_crystallization_level, dissent_response_quality, ambition_level) captures holistic quality at the chunk level without per-utterance redundancy
- Vocal enthusiasm, hesitation, and pace capture delivery quality at utterance level through the multimodal signals

**Inline quality rating (three categories only):** For three categories where meaningful quality variation exists *within* a single subcode — and where that variation is likely to predict outcomes independently — a lightweight `idea_quality` field (0/1/2) is included directly in the code object:
- **Idea Management**: a `proposes_new_idea` can range from an undeveloped throwaway to a richly elaborated hypothesis
- **Integration Practices**: a `synthesizes_contributions` can range from a superficial list to a generative new framing
- **Knowledge Sharing**: shared knowledge can range from tangential to filling a specific, important gap

For all other categories, the subcode alone captures the necessary quality distinction. Do not add `idea_quality` to other categories.

**Codebook categories (from `code_book_v4`, refined — now 16 total):**

Original 8: Idea Management, Information Seeking, Knowledge Sharing, Evaluation Practices, Relational Climate, Participation Dynamics, Coordination and Decision Practices, Integration Practices

New categories added in this version:
- **Idea Ownership / Attribution**: Does the speaker claim or explicitly attribute an idea to a specific person?
- **Future-Oriented Language**: Does the utterance reference future joint work, shared plans, or next steps?
- **Epistemic Bridging**: Does the speaker explicitly translate a concept across disciplinary frames?
- **Pronoun Framing** *(new)*: Is the speaker framing the proposed work individually or jointly?
- **Complementarity Articulation** *(new)*: Does the speaker explicitly name how their expertise complements another's?
- **Role Anticipation** *(new)*: Does the speaker map out who would do what in the proposed collaboration?
- **Broader Significance** *(new)*: Does the speaker articulate why the proposed work matters beyond the immediate research question?
- **Idea Novelty Signal** *(new)*: Does any participant explicitly mark an idea as surprising or unlike prior approaches?

**Rules:**
- Apply codes based on what is explicitly observed in the utterance, not inferred intent.
- If no code applies (e.g., filler words like "yep", "mm-hmm", short acknowledgments), write `None`.
- Assign no more than 3 codes per utterance.
- **Pronoun Framing is applied to every substantive utterance** that references the proposed research — it is not optional like other codes.
- Add `idea_quality` (0/1/2) only when `code_name` is Idea Management, Integration Practices, or Knowledge Sharing.
- For each code, provide: `code_name`, `subcode`, `evidence` (verbatim quote), `explanation` (1–2 sentences), and `idea_quality` if applicable.

---

### Layer 3: Multimodal-only signals per utterance (audio + video)

These are extracted during Pass 1 and require the video to be present. They are not requested from transcript-only inputs.

**4.1 Distraction status**
- `visible_off_screen_distraction` (Yes/No): Is any non-speaking participant visibly and clearly showing sustained off-screen distraction — looking far off-screen for more than a few seconds, using a phone in frame, or visibly engaged with someone off-camera? Do NOT flag brief eye movements or natural head shifts. When in doubt, write No.
- `distracted_participant_count` (integer): number of non-speakers showing clearly visible, sustained off-screen distraction. Set to 0 if none.

> **Note**: Camera on/off counts (`cameras_on_count`, `cameras_off_count`) were removed from the per-utterance annotation schema. Camera status is captured at the chunk level via the `collective_engagement_level` rating, which penalizes camera-off behavior in its rubric. Per-utterance camera counts were producing sparse, low-signal data that did not improve downstream modeling and increased annotation noise.

**4.2 Backchannel responses from non-speakers**
- Count of visible head nods from non-speakers during this turn: `nod_count` (integer, 0 if none).
- Any visible shared laughter or smiling from ≥2 participants simultaneously: `shared_affect` (Yes/No).
- Presence of at least one non-speaker smiling: `any_smile_other` (Yes/No).
- Audible backchannel vocalization from any non-speaker: `audible_backchannel` (Yes/No). Includes "mm-hmm", "yeah", "right", laughter, or any brief verbal response that does not constitute a full turn.

**4.3 Speaker vocal affect**
- Vocal enthusiasm rating (1–4):
  - 1 = Flat, monotone, minimal energy
  - 2 = Moderate energy, conversational
  - 3 = Noticeably energetic, engaged
  - 4 = High energy, passionate, emphatic
- Notable hesitation or long pauses before key claims: `hesitation_flag` (Yes/No).
- Notable speaking pace: `pace` (fast/normal/slow).

**4.4 Interruption quality**
- If this turn was an interruption, classify type:
  - `collaborative_completion`: listener finishes speaker's sentence in agreement
  - `elaborative_jump_in`: listener adds content before speaker finishes, in support
  - `competitive_interruption`: listener cuts off to redirect or contradict
  - `not_interruption`: this turn was not an interruption
- This replaces the prior binary Yes/No interruption code.

**Confidence flagging**: For any multimodal field where video quality is poor (low resolution, participant off-camera, dark frame), append `[low_confidence]` to that field's value.

---

### Prompt Library

These are the exact, production-ready prompts to be saved as plain `.txt` files in the `prompts/` directory. They are referenced verbatim in the annotation scripts. Any change to a prompt text must be saved as a new file version and re-hashed in `prompt_manifest.json`.

`prompts/pass1_chunk_prompt.txt` — Multimodal behavioral annotation 

---

### Session context preamble (cross-chunk continuity)

- Chunk annotation is sequential within each session.
- `build_chunk_prompt()` prepends a SESSION CONTEXT block before each chunk.
- `extract_session_state()` carries forward prior chunk state and repairs malformed `chunk_summaries` outputs.
- On resumed runs, previously saved chunk JSON files are loaded so context continues across already-completed chunks.

**One implementation detail to retain:** for the first chunk (`session_state is None`), the script prepends a short bracket note:
`[SESSION CONTEXT: This is chunk 1 ... No prior context ...]`
instead of the full prior-context template.

---

# DETAILED STEPS AND STAGES (implementation-aligned)

## Stage 0: Video preprocessing and `path_dict` creation

**Implemented in:** `src/analyze_video.py` via `main()`

### 0.1 Discover and normalize source videos

The script scans the target directory and supports videos in nested session folders or at root.
Supported extensions: `.mp4`, `.mkv`, `.avi`, `.mov`, `.flv`, `.wmv`.

Implementation behavior:
- `.mkv` is converted to `.mp4` when needed (`convert_mkv_to_mp4`).
- Videos longer than 10 minutes are split into 10-minute chunks (`split_video`).
- Existing split folders (`split-<recording_name>/`) are reused.

### 0.2 Build/update `path_dict`

`create_or_update_path_dict()` builds one ordered chunk list per session key (session folder name).

Each chunk entry uses this list schema:
`[chunk_file_name, chunk_path, gemini_file_name, analyzed_flag, model_used]`

Key properties of the current implementation:
- Session recordings are timestamp-sorted (`extract_recording_timestamp`) before chunk expansion.
- Split chunks are chunk-number sorted (`extract_chunk_number`).
- Existing progress is preserved by loading prior `<folder_name>_path_dict.json` and matching by `chunk_path`.
- Duplicate chunk paths are removed while preserving order.

### 0.3 Persist progress

`<folder_name>_path_dict.json` is written in the current working directory and updated during annotation after each chunk attempt.

### 0.4 Build chunk registry

**Notebook: `0-build_registry.ipynb`**

The `path_dict` tracks file paths and analysis status but carries no study-level metadata (conference identity, session outcome, temporal position within a session). The chunk registry is a flat, tabular version of the same information enriched with those fields. It is the master index used in Stage 3 for stratified sampling and in Stage 4–6 for joining features to outcomes.

**Build the registry after `path_dict` is finalized and before running `analyze_video.py` on the full dataset** — this is what makes it possible to select the human validation set without peeking at AI outputs.

#### Step 0.4.1 — Load outcomes and session metadata

Outcomes are stored in per-conference JSON files:
`analysis_v1/data/<conferenceID>/<conferenceID>_session_outcomes.json`

Each file maps `session_group` (e.g. `2020_11_05_NES_S1`) → `{teams: {team_id: {members, funded_status}}}`.
The `session_group` is the first path component of the `session_key` in the path_dict
(e.g. `2020_11_05_NES_S1` from `2020_11_05_NES_S1/1_DAC_Simulations_Zoom_Meeting_2020_11_05_12_09_10`).

```python
import pandas as pd
import json
from pathlib import Path

GEMINI_CODE = Path('../../gemini_code')          # relative to analysis_v2/notebooks/
ANALYSIS_V1 = GEMINI_CODE / 'analysis_v1/data'

path_dict_files = sorted(GEMINI_CODE.glob('*_path_dict.json'))

def parse_session_outcomes(outcomes_path):
    with open(outcomes_path) as f:
        raw = json.load(f)
    result = {}
    for session_group, data in raw.items():
        teams = data.get('teams', {})
        num_funded = sum(1 for t in teams.values() if t.get('funded_status', 0) == 1)
        result[session_group] = {
            'num_teams':        len(teams),
            'num_funded_teams': num_funded,
            'has_teams':        len(teams) > 0,
            'has_funded_teams': num_funded > 0,
        }
    return result
```

#### Step 0.4.2 — Expand path_dict entries into chunk rows

Each path_dict entry maps `session_key → list of chunks`. Expand into one row per chunk and derive temporal position.

```python
def chunk_position_label(chunk_index, n_chunks):
    """Assign beginning / middle / end label based on chunk index."""
    if n_chunks == 1:
        return 'whole'
    if chunk_index == 0:
        return 'beginning'
    if chunk_index == n_chunks - 1:
        return 'end'
    return 'middle'

rows = []
for pd_file in path_dict_files:
    with open(pd_file) as f:
        path_dict = json.load(f)

    for session_key, chunks in path_dict.items():
        n_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            # chunk schema: [chunk_file_name, chunk_path, gemini_file_name, analyzed_flag]
            # (4 elements; model_used is not stored in the current analyze_video.py implementation)
            chunk_file_name = chunk[0]
            chunk_path      = chunk[1]
            analyzed_flag   = bool(chunk[3])

            rows.append({
                'chunk_id':         f'{conference_id}__{session_key}__{chunk_file_name}',
                'session_key':      session_key,
                'chunk_file_name':  chunk_file_name,
                'chunk_path':       chunk_path,
                'chunk_index':      i,
                'n_chunks_in_session': n_chunks,
                'chunk_position':   chunk_position_label(i, n_chunks),
                'analyzed':         bool(analyzed_flag),
            })

registry = pd.DataFrame(rows)
```

#### Step 0.4.3 — Join study-level metadata

Outcomes are joined inline while iterating the path_dict (see notebook Step 4). Conference ID is derived from the path_dict filename. Session group is the first component of the session_key. After building `rows` and constructing the DataFrame, add the convenience stratification flag:

```python
registry = pd.DataFrame(rows)
registry['outcome_has_funded_teams'] = registry['has_funded_teams'].astype('boolean')

# Report any sessions with no outcome match
unmatched = [(c, s) for c, s in unmatched_sessions]
if unmatched:
    print(f'WARNING: {len(unmatched)} session groups had no outcome entry')
```

#### Step 0.4.4 — Add validation flags (empty; filled in Stage 3a)

```python
registry['human_validation_set']   = False
registry['utterance_validation_set'] = False
registry['oversampled_for']        = None
```

#### Step 0.4.5 — Validate and save

```python
# Sanity checks
assert registry['chunk_id'].is_unique, 'chunk_id must be unique'
assert registry['conference_id'].notna().all(), 'every chunk must have a conference_id'

print(f'Registry built: {len(registry)} chunks across {registry["session_key"].nunique()} sessions')
print(registry.groupby(['conference_id', 'chunk_position']).size().unstack(fill_value=0))

# Save
registry.to_parquet('data/chunk_registry_v1.parquet', index=False)
registry.to_csv('data/chunk_registry_v1.csv', index=False)   # human-readable backup
print('Saved data/chunk_registry_v1.parquet')
```

**Expected columns in `chunk_registry_v1.parquet`:**

| Column | Type | Description |
|---|---|---|
| `chunk_id` | str | `<conference_id>__<session_key>__<chunk_file_name>` — globally unique |
| `session_key` | str | Folder name used as session identifier |
| `chunk_file_name` | str | Filename of the chunk video (e.g. `chunk_003.mp4`) |
| `chunk_path` | str | Absolute path to the video file |
| `chunk_index` | int | 0-based position within the session |
| `n_chunks_in_session` | int | Total chunks in this session |
| `chunk_position` | str | `beginning` / `middle` / `end` / `whole` |
| `analyzed` | bool | Whether AI annotation has run (from path_dict) |
| `conference_id` | str | Conference identifier (clustering variable) |
| `has_teams` | bool | Session-level outcome |
| `has_funded_teams` | bool | Session-level outcome |
| `num_teams` | int | Number of teams formed |
| `num_funded_teams` | int | Number of funded teams formed |
| `outcome_has_funded_teams` | bool | Convenience copy for stratification key |
| `human_validation_set` | bool | Filled in Stage 3a (initially `False`) |
| `utterance_validation_set` | bool | Filled in Stage 3a (initially `False`) |
| `oversampled_for` | str / None | Which rare flag triggered oversampling (Stage 3a) |

---

## Stage 1: Multimodal chunk annotation

**Implemented in:** `analyze_video()` + `gemini_analyze_video()`

### 1.1 Prompt and model invocation

Current runtime flow per chunk:
1. Build prompt with `build_chunk_prompt(prompt, session_state, chunk_index)`.
2. Resolve/upload Gemini file with `get_gemini_video()`.
3. Call `gemini_analyze_video()`.

Model fallback order in code:
1. `gemini-3.1-pro-preview`
2. `gemini-3.1-flash-lite-preview`
3. `gemini-3-flash-preview`
4. `gemini-2.5-pro`

Retry behavior:
- Up to 3 attempts per model for non-fatal errors.
- Fatal API errors (400/403/413/429/503) trigger model-switch; if all models fail fatally, run stops with `FatalAPIError` after saving progress.

### 1.2 Output persistence

For each successful chunk response:
- Response text is saved to JSON via `save_to_json()`.
- JSON parsing is attempted directly; if malformed, `parse_json_garbage()` tries salvage.
- If salvage fails, raw text is saved as `ATTN_<chunk>.json` for manual inspection.

Output directory pattern:
`outputs/<input_folder_name>/output_<session_key>/<chunk_name>.json`

### 1.3 Session-state continuity during reruns

If a chunk is already marked analyzed, the script loads the saved chunk JSON and rehydrates `session_state` via `extract_session_state()` before continuing to later chunks.

This enables safe resume without losing cross-chunk context.

---

## Stage 2: Current guardrails implemented in code

The following quality/error guardrails are implemented today:
- API fatal error classification and early stop with progress save.
- Per-model fallback and retry loop.
- JSON salvage path (`parse_json_garbage`) and manual-attention outputs (`ATTN_*.json`).
- Invalid/empty JSON detection in `load_json_files()` via `InvalidJsonContentError`.



---

## Stage 3: Human expert validation and agreement computation

**Notebooks:** `3a-sample_validation_set.ipynb` · `3b-export_coding_materials.ipynb` · `3c-compute_agreement.ipynb`

This stage is the methodological centerpiece. It must be completed before any features are used in predictive models. The core challenge is that the annotation scheme has ~25 chunk-level dimensions and ~10 utterance-level fields per turn — no human rater can attend to all of these simultaneously with acceptable reliability. The solution is instrument decomposition with multiple passes per chunk, stratified sampling to ensure rare events are covered, and a tiered priority system that determines which dimensions must be validated before model use.

---

### 3.1 Construct the human validation set

**Notebook: `3a-sample_validation_set.ipynb`**

**Principle:** The sample must be drawn before AI annotation runs so that selection cannot be influenced by model performance. Save the `human_validation_set` flag to the registry before running `analyze_video.py`.

#### Step 3.1.1 — Define dimension priority tiers

Assign each chunk-level dimension to a validation tier. Only Tier 1 dimensions are blockers for model use; Tier 2 are important but can proceed with caveats; Tier 3 are descriptive and can remain AI-only.

```python
TIER_1 = [  # Must validate before any predictive model use
    'idea_trajectory',
    'collective_engagement_level',
    'explicit_commitment_signal',
    'decision_crystallization_level',
    'pronoun_shift_flag',
    'cross_disciplinary_bridging',
    'shared_vision_indicator',
]
TIER_2 = [  # Should validate; include with caveat if kappa 0.40–0.59
    'problem_specificity_level',
    'ambition_level',
    'laughter_quality',
    'dissent_response_quality',
    'risk_acknowledgment_with_enthusiasm',
    'personal_disclosure',
    'meeting_structure_quality',
]
TIER_3 = [  # Descriptive only; AI-only annotation acceptable
    'screenshare_active',
    'artifact_interaction',
    'funding_awareness_signal',
    'prior_relationship_signal',
    'explicit_complementarity_recognition',
    'skill_gap_identification',
]
UTTERANCE_PRIORITY = [  # Utterance-level: validate only these categories
    'Idea_Management',
    'Integration_Practices',
    'Pronoun_Framing',
    'interruption_type',
]
```

#### Step 3.1.2 — Stratified random sampling

Target: **20% of chunks for chunk-level coding** (all Tier 1 + 2 dimensions); **50 chunks for utterance-level coding** (utterance priority categories only).

Stratify the chunk-level sample on four dimensions to ensure coverage:

```python
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# Load registry built in Stage 0
registry = pd.read_parquet('data/chunk_registry_v1.parquet')

# Derive a combined stratification key
registry['strat_key'] = (
    registry['conference_id'].astype(str) + '__' +
    registry['chunk_position'] + '__' +            # beginning / middle / end
    registry['outcome_has_funded_teams'].astype(str)
)

# 20% stratified sample (chunk-level coding)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
_, val_idx = next(sss.split(registry, registry['strat_key']))
registry['human_validation_set'] = False
registry.loc[registry.index[val_idx], 'human_validation_set'] = True

print(registry['human_validation_set'].value_counts())
print(registry[registry['human_validation_set']].groupby(
    ['conference_id', 'chunk_position', 'outcome_has_funded_teams']
).size())
```

#### Step 3.1.3 — Oversample rare-event chunks

Random stratified sampling will underrepresent rare binary flags. Ensure the validation set contains at least **15 positive examples** of each Tier 1 binary dimension by inspecting AI annotations and adding targeted chunks where needed.

```python
RARE_FLAGS = [
    'explicit_commitment_signal',
    'cross_disciplinary_bridging',
    'risk_acknowledgment_with_enthusiasm',
]
MIN_POSITIVE = 15

# Load all AI-annotated chunk summaries (from Stage 1 outputs)
ai_summaries = load_all_chunk_summaries('outputs/')  # returns DataFrame

for flag in RARE_FLAGS:
    n_positive_in_val = (
        ai_summaries
        .loc[ai_summaries['chunk_id'].isin(
            registry.loc[registry['human_validation_set'], 'chunk_id']
        ), flag]
        .eq('Yes').sum()
    )
    if n_positive_in_val < MIN_POSITIVE:
        # Find positive chunks not yet in validation set
        positive_not_in_val = ai_summaries.loc[
            (ai_summaries[flag] == 'Yes') &
            (~ai_summaries['chunk_id'].isin(
                registry.loc[registry['human_validation_set'], 'chunk_id']
            ))
        ]
        n_needed = MIN_POSITIVE - n_positive_in_val
        extra_ids = positive_not_in_val.sample(
            min(n_needed, len(positive_not_in_val)), random_state=42
        )['chunk_id']
        registry.loc[registry['chunk_id'].isin(extra_ids), 'human_validation_set'] = True
        registry.loc[registry['chunk_id'].isin(extra_ids), 'oversampled_for'] = flag
        print(f'{flag}: added {len(extra_ids)} oversampled chunks')

# Save updated registry
registry.to_parquet('data/chunk_registry_v2.parquet', index=False)
```

**Note:** Track `oversampled_for` so that population-level statistics (e.g., prevalence estimates) are computed on the non-oversampled random sample only; oversampled chunks are used only for kappa computation.

#### Step 3.1.4 — Select utterance-level subsample

```python
# 50-chunk subsample for utterance-level coding, drawn from within the chunk-level set
utterance_val = (
    registry[registry['human_validation_set']]
    .groupby(['conference_id', 'chunk_position'], group_keys=False)
    .apply(lambda g: g.sample(frac=0.25, random_state=42))
    .head(50)
)
registry['utterance_validation_set'] = registry['chunk_id'].isin(utterance_val['chunk_id'])
registry.to_parquet('data/chunk_registry_v2.parquet', index=False)
```

---

### 3.2 Coding instrument design and rater materials

**Notebook: `3b-export_coding_materials.ipynb`**

**Core principle:** Decompose the full annotation scheme into three thematically coherent instruments. Each rater watches the same video chunk up to three times, once per instrument. This keeps per-viewing attention load to 6–8 dimensions, which is within reliable human coding capacity.

| Instrument | Dimensions coded | Viewing focus |
|---|---|---|
| **A — Intellectual trajectory** | `idea_trajectory`, `problem_specificity_level`, `decision_crystallization_level`, `ambition_level`, `cross_disciplinary_bridging`, `explicit_commitment_signal` | What is being discussed; where the group is headed |
| **B — Social & relational dynamics** | `pronoun_shift_flag`, `shared_vision_indicator`, `laughter_quality`, `personal_disclosure`, `dissent_response_quality`, `risk_acknowledgment_with_enthusiasm`, `meeting_structure_quality` | How participants relate; affective and relational signals |
| **C — Behavioral responsiveness** | `collective_engagement_level` (and its sub-signals: nods, facial expressions, backchannels, cameras-off) | Non-speaker behavior only; can be watched at reduced playback speed |

**Utterance instrument (subset of chunks only):** `Idea_Management` subcode, `Integration_Practices` subcode, `Pronoun_Framing` subcode, `interruption_type`. Raters code all utterances in the chunk but only for these four categories.

#### Step 3.2.1 — Generate per-chunk coding sheets

For each chunk in the validation set, export one coding sheet per instrument as a CSV (one row per field, with AI annotation hidden):

```python
INSTRUMENT_A_FIELDS = [
    'idea_trajectory',
    'idea_trajectory_justification',
    'problem_specificity_level',
    'problem_specificity_justification',
    'decision_crystallization_level',
    'decision_crystallization_justification',
    'ambition_level',
    'cross_disciplinary_bridging',
    'cross_disciplinary_bridging_description',
    'explicit_commitment_signal',
    'commitment_signal_quote',
]
INSTRUMENT_B_FIELDS = [
    'pronoun_shift_flag',
    'shared_vision_indicator',
    'shared_vision_quote',
    'laughter_quality',
    'personal_disclosure',
    'dissent_response_quality',
    'risk_acknowledgment_with_enthusiasm',
    'risk_enthusiasm_quote',
    'meeting_structure_quality',
]
INSTRUMENT_C_FIELDS = [
    'collective_engagement_level',
    'collective_engagement_justification',
]

def export_coding_sheet(chunk_id, instrument_fields, output_dir):
    rows = [{'field': f, 'rater_code': '', 'notes': ''} for f in instrument_fields]
    df = pd.DataFrame(rows)
    df.to_csv(f'{output_dir}/{chunk_id}__instrument_{instrument_fields[0][:1]}.csv', index=False)

val_chunks = registry.loc[registry['human_validation_set'], 'chunk_id'].tolist()
for chunk_id in val_chunks:
    export_coding_sheet(chunk_id, INSTRUMENT_A_FIELDS, 'data/human_coding/materials/A')
    export_coding_sheet(chunk_id, INSTRUMENT_B_FIELDS, 'data/human_coding/materials/B')
    export_coding_sheet(chunk_id, INSTRUMENT_C_FIELDS, 'data/human_coding/materials/C')
```

#### Step 3.2.2 — Prepare rater packages

- Copy chunk video files (`.mp4`) for all validation chunks to `data/human_coding/videos/`.
- Do **not** include AI annotations in any materials delivered to raters.
- Export codebook anchors for each instrument as a reference PDF.
- Deliver via secure shared folder (e.g., Google Drive with individual rater folders). Each rater receives their own copy so annotations remain independent.

---

### 3.3 Rater training protocol

- Minimum 2 raters with background in team science, organizational behavior, or interaction analysis.
- **Calibration phase:** 3 practice chunks coded together with discussion per instrument. Calibration chunks are drawn from sessions **outside** the formal validation sample. Do not use validation chunks for calibration.
- **Pilot reliability phase:** 5 chunks coded independently per instrument. Compute preliminary kappa per dimension. If kappa < 0.50 on any Tier 1 dimension, revise codebook anchors and repeat pilot. Pilot chunks are excluded from final validation statistics.
- **Main validation phase:** Remaining validation chunks coded independently. No discussion between raters until after all coding is complete for the batch.
- Raters are assigned **one instrument at a time** (complete all chunks for Instrument A before starting B) to minimize context switching.

---

### 3.4 Ingest and reconcile human codes

**Notebook: `3c-compute_agreement.ipynb`**

#### Step 3.4.1 — Load and merge rater files

```python
import pandas as pd
import glob

def load_rater_codes(rater_dir, instrument):
    files = glob.glob(f'{rater_dir}/instrument_{instrument}/*.csv')
    dfs = []
    for f in files:
        chunk_id = os.path.basename(f).replace(f'__instrument_{instrument}.csv', '')
        df = pd.read_csv(f)
        df['chunk_id'] = chunk_id
        dfs.append(df)
    return pd.concat(dfs).pivot(index='chunk_id', columns='field', values='rater_code')

rater1_A = load_rater_codes('data/human_coding/rater1', 'A')
rater2_A = load_rater_codes('data/human_coding/rater2', 'A')
rater1_B = load_rater_codes('data/human_coding/rater1', 'B')
rater2_B = load_rater_codes('data/human_coding/rater2', 'B')
rater1_C = load_rater_codes('data/human_coding/rater1', 'C')
rater2_C = load_rater_codes('data/human_coding/rater2', 'C')

# Stack all instruments
all_fields = INSTRUMENT_A_FIELDS + INSTRUMENT_B_FIELDS + INSTRUMENT_C_FIELDS
rater1_all = pd.concat([rater1_A, rater1_B, rater1_C], axis=1).reindex(columns=all_fields)
rater2_all = pd.concat([rater2_A, rater2_B, rater2_C], axis=1).reindex(columns=all_fields)
```

#### Step 3.4.2 — Resolve disagreements

```python
# Majority vote (2 raters: use rater1 as default; flag disagreements for reconciliation)
resolved = rater1_all.copy()
disagreements = []

for field in all_fields:
    mismatch = rater1_all[field] != rater2_all[field]
    disagreements.append({
        'field': field,
        'n_disagreements': mismatch.sum(),
        'pct_disagreements': mismatch.mean()
    })
    # Flag disagreements for manual adjudication
    resolved.loc[mismatch, field] = 'DISPUTED'

pd.DataFrame(disagreements).to_csv('data/human_coding/disagreement_summary.csv', index=False)
# Adjudicated values entered manually into resolved_codes.csv
resolved.to_csv('data/human_coding/resolved_codes.csv')
```

---

### 3.5 Compute inter-rater reliability (human vs. human)

```python
from sklearn.metrics import cohen_kappa_score
from pingouin import intraclass_corr

ORDINAL_FIELDS = [
    'collective_engagement_level',
    'problem_specificity_level',
    'decision_crystallization_level',
    'dissent_response_quality',
]
BINARY_FIELDS = [
    'explicit_commitment_signal', 'cross_disciplinary_bridging',
    'pronoun_shift_flag', 'shared_vision_indicator',
    'personal_disclosure', 'risk_acknowledgment_with_enthusiasm',
]
CATEGORICAL_FIELDS = [
    'idea_trajectory', 'ambition_level', 'laughter_quality', 'meeting_structure_quality',
]

irr_results = []

for field in BINARY_FIELDS + CATEGORICAL_FIELDS:
    paired = rater1_all[[field]].join(rater2_all[[field]], lsuffix='_r1', rsuffix='_r2').dropna()
    kappa = cohen_kappa_score(paired[f'{field}_r1'], paired[f'{field}_r2'])
    irr_results.append({'field': field, 'metric': 'cohen_kappa', 'value': kappa, 'n': len(paired)})

for field in ORDINAL_FIELDS:
    # ICC(2,1) two-way mixed for ordinal ratings
    scores_long = pd.concat([
        rater1_all[[field]].rename(columns={field: 'rating'}).assign(rater='rater1').reset_index(),
        rater2_all[[field]].rename(columns={field: 'rating'}).assign(rater='rater2').reset_index(),
    ])
    scores_long['rating'] = pd.to_numeric(scores_long['rating'], errors='coerce')
    scores_long = scores_long.dropna(subset=['rating'])
    icc_result = intraclass_corr(data=scores_long, targets='chunk_id',
                                  raters='rater', ratings='rating')
    icc_val = icc_result.loc[icc_result['Type'] == 'ICC2', 'ICC'].values[0]
    irr_results.append({'field': field, 'metric': 'ICC2', 'value': icc_val,
                        'n': scores_long['chunk_id'].nunique()})

irr_df = pd.DataFrame(irr_results)
irr_df.to_csv('data/validation/human_irr_results.csv', index=False)
print(irr_df.sort_values('value'))
```

---

### 3.6 Compute human-AI agreement

Use the reconciled `resolved_codes.csv` as the ground-truth reference.

```python
# Load AI chunk summaries for validation chunks only
ai_rows = []
for chunk_id in val_chunks:
    chunk_json = load_chunk_json(chunk_id, 'outputs/')   # loads saved JSON
    summary = chunk_json.get('chunk_summary', {})
    summary['chunk_id'] = chunk_id
    ai_rows.append(summary)

ai_df = pd.DataFrame(ai_rows).set_index('chunk_id')
human_df = pd.read_csv('data/human_coding/resolved_codes.csv', index_col='chunk_id')

# Align on shared chunks
shared = human_df.index.intersection(ai_df.index)
human_aligned = human_df.loc[shared]
ai_aligned = ai_df.loc[shared]

hai_results = []
for field in BINARY_FIELDS + CATEGORICAL_FIELDS:
    if field not in ai_aligned.columns:
        continue
    paired = human_aligned[[field]].join(
        ai_aligned[[field]], lsuffix='_human', rsuffix='_ai'
    ).dropna()
    kappa = cohen_kappa_score(paired[f'{field}_human'], paired[f'{field}_ai'])
    hai_results.append({'field': field, 'metric': 'cohen_kappa_human_ai',
                        'value': kappa, 'n': len(paired)})

for field in ORDINAL_FIELDS:
    if field not in ai_aligned.columns:
        continue
    scores_long = pd.concat([
        human_aligned[[field]].rename(columns={field: 'rating'}).assign(source='human').reset_index(),
        ai_aligned[[field]].rename(columns={field: 'rating'}).assign(source='ai').reset_index(),
    ])
    scores_long['rating'] = pd.to_numeric(scores_long['rating'], errors='coerce')
    scores_long = scores_long.dropna(subset=['rating'])
    icc_result = intraclass_corr(data=scores_long, targets='chunk_id',
                                  raters='source', ratings='rating')
    icc_val = icc_result.loc[icc_result['Type'] == 'ICC2', 'ICC'].values[0]
    hai_results.append({'field': field, 'metric': 'ICC2_human_ai',
                        'value': icc_val, 'n': scores_long['chunk_id'].nunique()})

hai_df = pd.DataFrame(hai_results)
hai_df.to_csv('data/validation/human_ai_agreement.csv', index=False)
print(hai_df.sort_values('value'))
```

---

### 3.7 Feature inclusion decision rule

```python
KAPPA_INCLUDE   = 0.60   # full inclusion
KAPPA_CAVEAT    = 0.40   # include with caveat; flag in paper
# < KAPPA_CAVEAT = exclude from predictive models

decisions = []
for _, row in hai_df.iterrows():
    field = row['field']
    k = row['value']
    tier = (
        'tier1' if field in TIER_1 else
        'tier2' if field in TIER_2 else
        'tier3'
    )
    if k >= KAPPA_INCLUDE:
        decision = 'include'
    elif k >= KAPPA_CAVEAT:
        decision = 'include_with_caveat'
    else:
        decision = 'exclude'
    decisions.append({'field': field, 'tier': tier,
                      'human_ai_kappa': k, 'decision': decision})

feature_decisions = pd.DataFrame(decisions)
feature_decisions.to_csv('data/validation/feature_inclusion_decision.csv', index=False)

# Print summary
print(feature_decisions.groupby(['tier', 'decision']).size())
excluded = feature_decisions[feature_decisions['decision'] == 'exclude']
if not excluded.empty:
    print('\nExcluded features (require prompt revision before use):')
    print(excluded[['field', 'human_ai_kappa']].to_string())
```

**Decision rules:**
- `kappa ≥ 0.60`: **include** — feature used in all predictive models.
- `kappa 0.40–0.59`: **include with caveat** — feature included but flagged as "moderate reliability" in the paper; sensitivity analyses must be run excluding these features.
- `kappa < 0.40`: **exclude** — feature dropped from predictive models. If theoretically important, revise prompt and re-annotate before re-running validation. Document exclusion explicitly in paper.
- Any Tier 1 feature with `kappa < 0.40` triggers a prompt revision cycle before Stage 4 can proceed.

---

### 3.8 Outputs

```
data/
  human_coding/
    materials/
      A/         ← per-chunk CSV coding sheets, Instrument A
      B/         ← per-chunk CSV coding sheets, Instrument B
      C/         ← per-chunk CSV coding sheets, Instrument C
    videos/      ← copied .mp4 files for validation chunks
    rater1/      ← completed coding sheets, Rater 1
    rater2/      ← completed coding sheets, Rater 2
    disagreement_summary.csv
    resolved_codes.csv    ← adjudicated ground truth
  validation/
    human_irr_results.csv          ← kappa / ICC2 per dimension, human vs. human
    human_ai_agreement.csv         ← kappa / ICC2 per dimension, human vs. AI
    feature_inclusion_decision.csv ← include / caveat / exclude per feature
  chunk_registry_v2.parquet        ← updated with human_validation_set and oversampled_for flags
```

The `human_ai_agreement.csv` and `feature_inclusion_decision.csv` tables together become **Table 1** in the paper (annotation reliability and validity summary).

---

## Stage 4: Feature engineering

**Notebook: `4-feature_engineering.ipynb`**


### 4.1 Chunk-level derived features

For each chunk, compute from validated Pass 1 outputs:

```python
# ── Participation ──────────────────────────────────────────────────────────
speaking_times = chunk_summary['speaking_time_seconds']
gini_coefficient        = compute_gini(list(speaking_times.values()))
dominant_speaker_flag   = int(max(speaking_times.values()) /
                               sum(speaking_times.values()) > 0.50)

# ── Idea trajectory ────────────────────────────────────────────────────────
idea_trajectory = chunk_summary['idea_trajectory']
is_convergent   = int(idea_trajectory == 'convergent')
is_divergent    = int(idea_trajectory == 'divergent')
is_procedural   = int(idea_trajectory == 'procedural')

# ── Collective engagement (multimodal; null in transcript-only sessions) ───
collective_engagement_score = chunk_summary['collective_engagement_level']  # 1–4 or null

# ── Cross-disciplinary bridging ────────────────────────────────────────────
cross_disciplinary_bridging = int(chunk_summary['cross_disciplinary_bridging'] == 'Yes')

# ── Commitment signal ──────────────────────────────────────────────────────
commitment_signal = int(chunk_summary['explicit_commitment_signal'] == 'Yes')

# ── Artifact engagement ────────────────────────────────────────────────────
screenshare_active  = int(chunk_summary['screenshare_active'] == 'Yes')
artifact_interaction = int(chunk_summary['artifact_interaction'] == 'Yes')

# ── Intellectual quality (NEW) ─────────────────────────────────────────────
problem_specificity_level     = chunk_summary['problem_specificity_level']     # 1–4 or NA→null
decision_crystallization_level = chunk_summary['decision_crystallization_level']  # 1–4

# Ambition level: encode as ordinal (0=not_applicable, 1=incremental, …, 4=paradigm_challenging)
ambition_map = {'not_applicable': 0, 'incremental': 1, 'novel_application': 2,
                'novel_combination': 3, 'paradigm_challenging': 4}
ambition_level_ordinal = ambition_map[chunk_summary['ambition_level']]
is_novel_combination   = int(chunk_summary['ambition_level'] in
                              ['novel_combination', 'paradigm_challenging'])

# ── Complementarity and shared vision (NEW) ────────────────────────────────
explicit_complementarity = int(chunk_summary['explicit_complementarity_recognition'] == 'Yes')
skill_gap_identified     = int(chunk_summary['skill_gap_identification'] == 'Yes')
shared_vision_present    = int(chunk_summary['shared_vision_indicator'] == 'Yes')
pronoun_shift_occurred   = int(chunk_summary['pronoun_shift_flag'] == 'Yes')

# ── Interpersonal signals (NEW) ────────────────────────────────────────────
personal_disclosure = int(chunk_summary['personal_disclosure'] == 'Yes')

# Laughter quality: separate binary flags for each type
laughter_q = chunk_summary['laughter_quality']
laughter_appreciative   = int(laughter_q == 'appreciative')
laughter_shared_humor   = int(laughter_q == 'shared_humor')
laughter_tension_release = int(laughter_q == 'tension_release')
any_laughter            = int(laughter_q != 'none')

# ── Psychological safety (NEW) ────────────────────────────────────────────
# dissent_response_quality: 1–3 or NA
drq = chunk_summary['dissent_response_quality']
dissent_response_quality     = drq if drq != 'NA' else None
dissent_was_present          = int(drq != 'NA')
dissent_response_exploratory = int(drq == 3)

# ── Risk and ambition (NEW) ────────────────────────────────────────────────
risk_acknowledgment_enthusiasm = int(
    chunk_summary['risk_acknowledgment_with_enthusiasm'] == 'Yes')

# ── Grant and context signals (NEW) ────────────────────────────────────────
funding_awareness  = int(chunk_summary['funding_awareness_signal'] == 'Yes')
prior_relationship = int(chunk_summary['prior_relationship_signal'] == 'Yes')

# ── Meeting process quality (NEW) ──────────────────────────────────────────
structure_map = {'unstructured': 0, 'loosely_structured': 1, 'structured': 2}
meeting_structure_quality = structure_map[chunk_summary['meeting_structure_quality']]
```

### 4.2 Utterance-level aggregated features (per chunk)

```python
# ── Behavioral code counts for all 16 categories ──────────────────────────
for category in included_categories:  # only those passing kappa threshold
    chunk_features[f'num_{category}'] = count_codes(utterances, category)

# ── Inline idea_quality scores (Idea Management, Integration, Knowledge Sharing only) ─
for category in ['Idea_Management', 'Integration_Practices', 'Knowledge_Sharing']:
    scores = [code['idea_quality'] for u in utterances
              for code in u['codes']
              if code['code_name'] == category and 'idea_quality' in code]
    chunk_features[f'mean_idea_quality_{category}'] = mean(scores) if scores else None
    chunk_features[f'pct_high_quality_{category}']  = (
        sum(s >= 2 for s in scores) / len(scores) if scores else None
    )

# ── Building vs. blocking ratio (article Change 1) ────────────────────────
# Building: positive idea management, integration, positive evaluation, knowledge sharing
building_count = (
    count_codes(utterances, 'Idea_Management',
                subcodes=['proposes_new_idea','extends_existing_idea','combines_ideas',
                          'returns_to_earlier_idea']) +
    count_codes(utterances, 'Integration_Practices') +
    count_codes(utterances, 'Evaluation_Practices',
                subcodes=['supports_or_validates']) +
    count_codes(utterances, 'Knowledge_Sharing')
)
# Blocking: explicit challenge/critique/concern subcodes + competitive interruptions
# (replaces the former score=-1 approach; idea_quality only has values 0–2 and
#  is only coded for three categories, so negative scores do not exist in the output)
blocking_count = (
    count_codes(utterances, 'Evaluation_Practices',
                subcodes=['critiques_or_challenges', 'devil_advocate', 'raises_concern']) +
    count_interruption_type(utterances, 'competitive_interruption')
)
chunk_features['building_count']          = building_count
chunk_features['blocking_count']          = blocking_count
chunk_features['building_blocking_ratio'] = building_count / (blocking_count + epsilon)
chunk_features['pct_building']            = (building_count /
                                              (building_count + blocking_count + epsilon))

# ── New individual code category counts ────────────────────────────────────
chunk_features['num_future_oriented']       = count_codes(utterances, 'Future_Oriented_Language')
chunk_features['num_named_next_steps']      = count_codes(utterances, 'Future_Oriented_Language',
                                                           subcodes=['named_next_step'])
chunk_features['num_epistemic_bridging']    = count_codes(utterances, 'Epistemic_Bridging')
chunk_features['num_idea_attribution']      = count_codes(utterances, 'Idea_Ownership_Attribution')
chunk_features['num_complementarity']       = count_codes(utterances, 'Complementarity_Articulation')
chunk_features['num_role_anticipation']     = count_codes(utterances, 'Role_Anticipation')
chunk_features['num_broader_significance']  = count_codes(utterances, 'Broader_Significance')
chunk_features['num_novelty_signals']       = count_codes(utterances, 'Idea_Novelty_Signal')
chunk_features['num_scope_calibration']     = count_codes(utterances, 'Coordination_Decision',
                                                           subcodes=['scope_calibration'])

# ── Setback response features ──────────────────────────────────────────────
chunk_features['num_setback_explores']      = count_codes(utterances, 'Evaluation_Practices',
                                                           subcodes=['setback_response_explores',
                                                                     'setback_response_accepts_builds'])
chunk_features['num_setback_defends']       = count_codes(utterances, 'Evaluation_Practices',
                                                           subcodes=['setback_response_defends',
                                                                     'setback_response_redirects'])
chunk_features['explore_vs_defend_ratio']   = (chunk_features['num_setback_explores'] /
                                                (chunk_features['num_setback_defends'] + epsilon))

# ── Pronoun framing features ────────────────────────────────────────────────
chunk_features['num_joint_framing']         = count_codes(utterances, 'Pronoun_Framing',
                                                           subcodes=['joint_framing'])
chunk_features['num_individual_framing']    = count_codes(utterances, 'Pronoun_Framing',
                                                           subcodes=['individual_framing'])
total_framing = chunk_features['num_joint_framing'] + chunk_features['num_individual_framing']
chunk_features['pct_joint_framing']         = (chunk_features['num_joint_framing'] /
                                                (total_framing + epsilon))

# ── Overall quality aggregates ─────────────────────────────────────────────
chunk_features['total_quality_score'] = sum_all_scores(utterances)
chunk_features['mean_quality_score']  = mean_all_scores(utterances)
chunk_features['pct_high_quality']    = pct_scores_gte(utterances, threshold=1)

# ── Interruption quality ────────────────────────────────────────────────────
collab    = count_interruption_type(utterances, 'collaborative_completion')
elab      = count_interruption_type(utterances, 'elaborative_jump_in')
compet    = count_interruption_type(utterances, 'competitive_interruption')
chunk_features['num_collaborative_completions'] = collab
chunk_features['num_elaborative_jumps']         = elab
chunk_features['num_competitive_interruptions'] = compet
chunk_features['interruption_quality_ratio']    = (collab + elab) / (compet + epsilon)
```

### 4.3 Multimodal signal aggregates (per chunk)

```python
# Camera engagement signals
# NOTE: camera_on_rate is NOT computable from current outputs. The utterance-level schema
# does not include cameras_on_count or cameras_off_count fields. The distraction proxy
# below (pct_turns_distraction, mean_distracted_count) is the valid replacement.

# Proportion of turns with any visible off-screen distraction
chunk_features['pct_turns_distraction'] = pct turns with visible_off_screen_distraction == 'Yes'
# Mean number of visibly distracted participants per turn
chunk_features['mean_distracted_count'] = mean(distracted_participant_count per utterance)

# Behavioral responsiveness (face and voice — Zoom-valid signals)
chunk_features['mean_nod_rate'] = mean(nod_count per utterance)
chunk_features['pct_turns_shared_affect'] = pct turns with shared_affect == 'Yes'
chunk_features['pct_turns_any_smile'] = pct turns with any_smile_other == 'Yes'
chunk_features['pct_turns_audible_backchannel'] = pct turns with audible_backchannel == 'Yes'

# Composite responsiveness index: average of nod_rate, pct_shared_affect, pct_backchannel
# (z-scored before averaging to put on common scale)
chunk_features['responsiveness_index'] = mean_of_zscored(
    mean_nod_rate, pct_turns_shared_affect, pct_turns_audible_backchannel
)

# Chunk-level engagement rating from chunk_summary (holistic 1–4 rating)
chunk_features['collective_engagement_score'] = chunk_summary['collective_engagement_level']

# Vocal affect of speakers
chunk_features['mean_vocal_enthusiasm'] = mean(vocal_enthusiasm per utterance)
chunk_features['pct_high_enthusiasm'] = pct turns with vocal_enthusiasm >= 3
chunk_features['pct_hesitation'] = pct turns with hesitation_flag == 'Yes'
```

> **Note**: The `mean_attending_ratio` and `mean_disengaged_ratio` features from earlier pipeline versions are removed. These were based on gaze-direction coding, which is not valid in Zoom recordings. The new `responsiveness_index`, `pct_turns_distraction`, `mean_nod_rate`, and `pct_turns_audible_backchannel` features are the Zoom-valid replacements and should be used in all models. `camera_on_rate` is also removed: per-utterance camera on/off counts are not coded in the Pass 1 output schema, so this feature cannot be computed.

### 4.4 Session-level aggregation (from chunks)

For each session, aggregate chunk-level features into session-level features:

```python
session_features = {}
for session_id in sessions:
    chunks = get_chunks(session_id)

    # Temporal structure: beginning / middle / end tertiles
    beg = chunks[chunks.chunk_position == 'beginning']
    mid = chunks[chunks.chunk_position == 'middle']
    end = chunks[chunks.chunk_position == 'end']

    # ── Automatic aggregation for all numeric chunk-level features ─────────
    for feat in chunk_level_numeric_features:
        session_features[f'session_mean_{feat}']       = mean(chunks[feat])
        session_features[f'session_beginning_{feat}']  = mean(beg[feat])
        session_features[f'session_middle_{feat}']     = mean(mid[feat])
        session_features[f'session_end_{feat}']        = mean(end[feat])
        session_features[f'session_delta_{feat}']      = (mean(end[feat]) -
                                                           mean(beg[feat]))  # trajectory

    # ── Convergence and trajectory composites ──────────────────────────────
    session_features['convergence_ratio_end']   = mean(end['is_convergent'])
    session_features['divergent_ratio_beg']     = mean(beg['is_divergent'])
    # Did session follow the expected diverge→converge arc?
    session_features['diverge_converge_arc']    = int(
        mean(beg['is_divergent']) > mean(end['is_divergent']) and
        mean(end['is_convergent']) > mean(beg['is_convergent'])
    )

    # ── Commitment signal timing ────────────────────────────────────────────
    commitment_chunks = chunks[chunks['commitment_signal'] == 1]
    session_features['any_commitment_signal']    = int(len(commitment_chunks) > 0)
    session_features['earliest_commitment_chunk'] = (
        commitment_chunks['chunk_index'].min() if len(commitment_chunks) > 0 else -1
    )
    # Normalized timing: 0 = appeared in first chunk, 1 = appeared in last chunk
    n_chunks = len(chunks)
    session_features['commitment_timing_normalized'] = (
        session_features['earliest_commitment_chunk'] / (n_chunks - 1)
        if session_features['any_commitment_signal'] and n_chunks > 1 else None
    )

    # ── Engagement trajectory ───────────────────────────────────────────────
    session_features['engagement_trajectory'] = (
        mean(end['collective_engagement_score'].dropna()) -
        mean(beg['collective_engagement_score'].dropna())
    )

    # ── Cross-disciplinary bridging timing ─────────────────────────────────
    early_chunks = chunks[chunks['chunk_index'] <= 1]
    session_features['early_bridging'] = int(early_chunks['cross_disciplinary_bridging'].any())

    # ── Participation trajectory ────────────────────────────────────────────
    session_features['gini_trajectory'] = (mean(end['gini_coefficient']) -
                                            mean(beg['gini_coefficient']))

    # ── Intellectual quality trajectories (NEW) ─────────────────────────────
    # Problem specificity: how much did the question sharpen over the session?
    valid_spec = chunks[chunks['problem_specificity_level'].notna()]
    if len(valid_spec) > 0:
        beg_spec = chunks[chunks['chunk_position'] == 'beginning']['problem_specificity_level'].dropna()
        end_spec = chunks[chunks['chunk_position'] == 'end']['problem_specificity_level'].dropna()
        session_features['problem_specificity_final']    = mean(end_spec) if len(end_spec) > 0 else None
        session_features['problem_specificity_delta']    = (mean(end_spec) - mean(beg_spec)
                                                             if len(beg_spec) > 0 and len(end_spec) > 0
                                                             else None)
    # Decision crystallization: most important at final chunk
    session_features['decision_crystallization_final']   = (
        end['decision_crystallization_level'].iloc[-1]
        if len(end) > 0 else None
    )
    session_features['decision_crystallization_delta']   = (
        mean(end['decision_crystallization_level']) -
        mean(beg['decision_crystallization_level'])
    )

    # Peak ambition level achieved at any point in the session
    session_features['max_ambition_level']               = chunks['ambition_level_ordinal'].max()
    session_features['any_novel_combination']            = int(chunks['is_novel_combination'].any())

    # ── Complementarity and shared vision (NEW) ─────────────────────────────
    session_features['any_complementarity_recognized']   = int(
        chunks['explicit_complementarity'].any())
    session_features['any_skill_gap_identified']         = int(
        chunks['skill_gap_identified'].any())
    session_features['any_shared_vision']                = int(
        chunks['shared_vision_present'].any())
    session_features['pronoun_shift_occurred']           = int(
        chunks['pronoun_shift_occurred'].any())
    # When (which tertile) did the pronoun shift first appear?
    shift_chunks = chunks[chunks['pronoun_shift_occurred'] == 1]
    session_features['pronoun_shift_timing']             = (
        shift_chunks['chunk_index'].min() / (n_chunks - 1)
        if len(shift_chunks) > 0 and n_chunks > 1 else None
    )
    # Trajectory: pct_joint_framing from beginning to end
    session_features['joint_framing_trajectory']         = (
        mean(end['pct_joint_framing']) - mean(beg['pct_joint_framing'])
    )

    # ── Building vs. blocking (NEW) ─────────────────────────────────────────
    session_features['building_blocking_ratio_session']  = (
        sum(chunks['building_count']) / (sum(chunks['blocking_count']) + epsilon)
    )
    session_features['pct_building_session']             = (
        sum(chunks['building_count']) /
        (sum(chunks['building_count']) + sum(chunks['blocking_count']) + epsilon)
    )

    # ── Psychological safety index (NEW) ─────────────────────────────────────
    # Only computed for chunks where dissent was present
    dissent_chunks = chunks[chunks['dissent_was_present'] == 1]
    session_features['pct_dissent_exploratory']          = (
        mean(dissent_chunks['dissent_response_exploratory'])
        if len(dissent_chunks) > 0 else None
    )
    session_features['mean_dissent_response_quality']    = (
        mean(dissent_chunks['dissent_response_quality'])
        if len(dissent_chunks) > 0 else None
    )
    # Composite psychological safety index
    session_features['psych_safety_index']               = mean_of_zscored(
        session_features['session_mean_interruption_quality_ratio'],
        session_features.get('pct_dissent_exploratory', 0),
        session_features['session_mean_pct_turns_audible_backchannel'],
        1 - session_features['session_mean_pct_hesitation']   # high safety → low hesitation
    )

    # ── Setback response (NEW) ──────────────────────────────────────────────
    session_features['explore_vs_defend_ratio_session']  = (
        sum(chunks['num_setback_explores']) /
        (sum(chunks['num_setback_defends']) + epsilon)
    )

    # ── Affective and interpersonal signals (NEW) ──────────────────────────
    session_features['any_personal_disclosure']          = int(
        chunks['personal_disclosure'].any())
    session_features['pct_chunks_appreciative_laughter'] = mean(
        chunks['laughter_appreciative'])
    session_features['pct_chunks_any_laughter']          = mean(chunks['any_laughter'])
    session_features['any_risk_enthusiasm']              = int(
        chunks['risk_acknowledgment_enthusiasm'].any())

    # ── Energy matching index (vocal enthusiasm synchrony) (NEW) ──────────
    # Computed per session from utterance-level vocal_enthusiasm per speaker
    # Measures whether enthusiasm levels co-move across speakers within turns
    utterances_all = get_all_utterances(session_id)
    session_features['energy_matching_index']            = compute_enthusiasm_synchrony(
        utterances_all
    )  # pearson r between per-turn enthusiasm of different speakers, averaged across pairs

    # ── Parallel monologue index (NEW) ──────────────────────────────────────
    # What fraction of new idea proposals were NOT preceded by the prior speaker's idea?
    session_features['parallel_monologue_index']         = compute_parallel_monologue(
        utterances_all
    )  # pct of proposes_new_idea turns not directly building on prior speaker

    # ── Named next steps (NEW) ─────────────────────────────────────────────
    final_two = chunks.nlargest(2, 'chunk_index')
    session_features['named_next_steps_count']           = sum(
        final_two['num_named_next_steps'])

    # ── Grant context signals (NEW) ─────────────────────────────────────────
    session_features['any_funding_awareness']            = int(
        chunks['funding_awareness'].any())
    session_features['prior_relationship_present']       = int(
        chunks['prior_relationship'].any())
    session_features['any_broader_significance']         = int(
        (chunks['num_broader_significance'] > 0).any())

    # ── Meeting structure quality (NEW) ─────────────────────────────────────
    session_features['mean_meeting_structure_quality']   = mean(
        chunks['meeting_structure_quality'])
    session_features['meeting_structure_final']          = (
        end['meeting_structure_quality'].iloc[-1] if len(end) > 0 else None
    )

    # ── Absence flags (NEW) ─────────────────────────────────────────────────
    # Negative predictors: things that didn't happen
    session_features['no_convergence_flag']              = int(
        session_features['decision_crystallization_final'] is not None and
        session_features['decision_crystallization_final'] <= 2
    )
    session_features['no_commitment_signal']             = int(
        not session_features['any_commitment_signal'])
    session_features['no_complementarity_recognition']   = int(
        not session_features['any_complementarity_recognized'])
    # Any unresolved dissent (dismissed without exploratory follow-up)?
    unresolved_dissent = dissent_chunks[
        dissent_chunks['dissent_response_quality'] == 1
    ] if len(dissent_chunks) > 0 else pd.DataFrame()
    session_features['unresolved_tension_flag']          = int(len(unresolved_dissent) > 0)
```

### 4.5 Join with outcomes and conference

```python
features_df = pd.DataFrame(session_features_dict).T
features_df = features_df.join(outcomes_df, on='session_id')
features_df = features_df.join(conference_df, on='session_id')
```

### 4.6 Multicollinearity check

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Run VIF on all candidate modeling features (excluding outcomes and IDs)
vif_results = compute_vif(features_df[modeling_features])

# Drop features with VIF > 10 iteratively (highest first), document all removals
high_vif_features = vif_results[vif_results.VIF > 10].feature.tolist()
```

### 4.7 Output

```
data/
  chunk_features.parquet       ← one row per chunk, all chunk-level derived features
  session_features.parquet     ← one row per session, aggregated + delta features
  model_ready_features.parquet ← session features after VIF filter, joined with outcomes
  feature_manifest.csv         ← feature name, reliability_kappa, included_in_model (True/False)
```

---

## Stage 5: Descriptive analysis and visualization

**Notebook: `5-descriptive_analysis.ipynb`**

### 5.1 Conference-level summaries

- Distribution of `num_teams` and `num_funded_teams` by conference (violin + strip plots).
- Mean and SD of key behavioral features by conference — this motivates the need for conference controls in all models.
- Heatmap: correlation of session-level behavioral features with each other and with outcomes.

### 5.2 Chunk-position analysis

- Plot mean values of key features (collective_engagement_score, gini_coefficient, pct_convergent, commitment_signal) by chunk_position (beginning/middle/end) across all sessions.
- Compare these profiles between sessions that formed teams vs. did not.

### 5.3 Validation results visualization

- Bar chart: human IRR kappa and human-AI agreement kappa per annotation category.
- Color coding: green (kappa ≥ 0.60), yellow (0.40–0.59), red (< 0.40).
- This figure becomes a key methodological figure in the paper.

### 5.4 Feature distributions

- Histograms for all modeling features (check for outliers, skew).
- Identify sessions to flag as potential outliers (e.g., sessions with extreme team counts as in preliminary pipeline).
- Log-transform or winsorize highly skewed features; document all transformations.

### 5.5 Output

```
figures/
  conference_outcome_distributions.png
  feature_conference_heatmap.png
  chunk_position_profiles.png
  irr_agreement_barplot.png
  feature_distributions/
    <feature_name>_hist.png    ← one per feature
```

---

## Stage 6: Inferential modeling

**Notebook: `6-regression_modeling.ipynb`**

### 6.1 Cross-validation strategy

**Primary**: Leave-one-session-out cross-validation (LOSO-CV) for all models.
- With N ≈ 157, this yields 157 test predictions, one per session.
- AUC is computed from the full vector of 157 out-of-sample predictions vs. true labels.
- This avoids the instability of random train/test splits and prevents cherry-picking.

**Secondary robustness check**: Leave-one-conference-out cross-validation (LOCO-CV).
- Train on all sessions except those in one conference; test on held-out conference.
- This tests whether models generalize across conferences — the most demanding test given the conference heterogeneity observed.

### 6.2 Outcome variables

Four outcomes, analyzed separately:
- `outcome_has_teams` (binary, primary)
- `outcome_has_funded_teams` (binary, primary)
- `outcome_num_teams` (count, secondary)
- `outcome_num_funded_teams` (count, secondary)

### 6.3 Model 1: Regularized logistic regression (primary model)

```python
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegressionCV(
        Cs=np.logspace(-4, 4, 20),
        cv=5,
        penalty='elasticnet',
        solver='saga',
        l1_ratios=[0.1, 0.5, 0.9],
        scoring='roc_auc',
        random_state=42
    ))
])

# LOSO-CV
loo = LeaveOneOut()
predictions = cross_val_predict(pipeline, X, y, cv=loo, method='predict_proba')[:, 1]
auc_loso = roc_auc_score(y, predictions)

# Report: AUC, coefficients from full-data fit, 95% CI via bootstrap
```

### 6.4 Model 2: Mixed-effects logistic regression (conference clustering)

```python
import statsmodels.formula.api as smf

# With conference as random intercept
model = smf.mixedlm(
    formula='outcome_has_funded_teams ~ ' + ' + '.join(modeling_features) + ' + meeting_length + num_members',
    data=features_df,
    groups=features_df['conference_id']
)
result = model.fit()
```

Report: coefficient estimates, SEs, z-statistics, p-values, random effect variance.

### 6.5 Model 3: Single-feature logistic models (univariate screening)

```python
for feature in modeling_features:
    model = LogisticRegression(C=1.0)
    pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
    preds = cross_val_predict(pipeline, X[[feature] + control_features], y, cv=loo, method='predict_proba')[:, 1]
    auc = roc_auc_score(y, preds)
    univariate_results[feature] = {'auc_loso': auc}

# Apply Benjamini-Hochberg correction for multiple testing
```

### 6.6 Model 4: Random Forest (robustness check, not primary)

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=500, max_features='sqrt', random_state=42)
# LOSO-CV
preds = cross_val_predict(rf, X_scaled, y, cv=loo, method='predict_proba')[:, 1]
auc_rf = roc_auc_score(y, preds)

# Feature importance: permutation importance (not Gini — Gini is biased)
from sklearn.inspection import permutation_importance
perm_imp = permutation_importance(rf_fitted, X_test, y_test, n_repeats=50, random_state=42)
```

### 6.7 Beginning-segment models

Rerun Models 1 and 2 using **only beginning-segment features** (`session_beginning_*` columns). This tests whether early meeting behavior alone predicts outcomes — the theoretically important question.

```python
beginning_features = [c for c in modeling_features if c.startswith('session_beginning_')]
# Add session-level controls: meeting_length, num_members, conference (fixed effect)
```

### 7.8 Reporting standards

For every model report:
- **Binary outcomes**: AUC-ROC (LOSO), balanced accuracy, F1.
- **Count outcomes**: RMSE, MAE, R² (LOSO).
- **Never report best-split performance** — only LOSO and LOCO means.
- Report 95% confidence intervals for AUC via DeLong's method or bootstrap (n=1000).
- Report confusion matrix at 0.5 threshold for context.

### 7.9 Sensitivity analyses

1. Exclude chunks/sessions with high low-confidence annotation rates (> 20% flagged fields).
2. Exclude features with moderate reliability only (kappa 0.40–0.59) and rerun.
3. Rerun with conference as fixed effects instead of random effects.
4. Compare models with vs. without multimodal-only features to quantify the added value of video over transcript.

### 7.10 Output

```
results/
  loso_auc_summary.csv                 ← all models × all outcomes
  loco_auc_summary.csv
  univariate_screening_results.csv     ← per-feature AUC + BH-corrected p-values
  mixed_effects_results.csv
  sensitivity_analyses/
    exclude_low_confidence.csv
    exclude_moderate_reliability.csv
    video_vs_transcript_comparison.csv
figures/
  roc_curves_primary_models.png
  feature_importance_lasso.png
  feature_importance_rf_permutation.png
  beginning_vs_full_session_auc_comparison.png
```

---

## Stage 7: Temporal segment analysis

**Notebook: `8-temporal_analysis.ipynb`**

### 7.1 Within-session temporal feature analysis

Test whether the temporal trajectory of behavioral features (beginning → end) predicts outcomes, beyond the level of any single segment.

```python
# Outcome: has_funded_teams
# Predictors: delta features (session_delta_*) = end - beginning
# Controls: meeting_length, num_members, conference fixed effects

for feature in temporal_features:
    delta_col = f'session_delta_{feature}'
    formula = f'outcome_has_funded_teams ~ {delta_col} + meeting_length + num_members + C(conference_id)'
    result = smf.logit(formula, data=features_df).fit()
    temporal_results[feature] = extract_results(result)
```

Apply Benjamini-Hochberg correction across all tested features.

### 7.2 Beginning → end persistence analysis (exploratory)

Replicate and extend the prior pipeline's temporal persistence analysis, now with LOSO-CV estimates and proper sample sizes.

Note: N = 157 sessions provides adequate power for this analysis (unlike the prior N = 16 temporal subsample). Report effect sizes (Cohen's f²) alongside p-values.

### 7.3 Chunk-position interaction models

```python
# Does the predictive value of a feature depend on WHEN it occurs in the meeting?
# Test: feature × chunk_position interaction in a mixed model

formula = f'outcome ~ {feature} * chunk_position + (1|session_id) + (1|conference_id)'
```

### 7.4 Output

```
results/
  temporal_delta_results.csv            ← delta feature logistic models
  temporal_persistence_results.csv      ← beginning → end feature-level regressions
  chunk_position_interaction_results.csv
figures/
  temporal_profiles_by_outcome.png      ← feature trajectories split by outcome class
  delta_feature_forest_plot.png
```

---

## Stage 8: Reproducibility packaging

**Notebook: `9-reproducibility_check.ipynb`** + repository setup

### 8.1 Repository structure

```
/
  README.md                         ← project overview, setup instructions
  requirements.txt                  ← pinned dependency versions
  prompts/
    pass1_chunk_prompt.txt
    pass1_utterance_prompt.txt
    # pass2_quality_prompt.txt  ← eliminated (see design note in Prompt Library)
    prompt_manifest.json
  data/
    chunk_registry_v1.parquet       ← anonymized (no participant names)
    chunk_registry_v2.parquet
    model_ready_features.parquet    ← anonymized
    feature_manifest.csv
  outputs/
    pass1/                          ← raw annotation JSONs (anonymized)
    # pass2/  ← eliminated
  notebooks/
    0-build_registry.ipynb
    1-annotate.ipynb
    2-validate_schema.ipynb
    3-human_validation.ipynb
    4-feature_engineering.ipynb
    5-descriptive_analysis.ipynb
    6-regression_modeling.ipynb
    8-temporal_analysis.ipynb
    9-reproducibility_check.ipynb
  scripts/
    annotate.py
    # annotate_pass2.py  ← eliminated
    utils/
      vtt_parser.py
      gemini_client.py
      schema_validator.py
      feature_utils.py
      stats_utils.py
  results/
  figures/
  human_coding/                     ← rater materials (not uploaded if identifiable)
```

### 8.2 Reproducibility checks in `9-reproducibility_check.ipynb`

1. Re-run annotation on a random sample of 10 chunks (using the same model version, temperature=0, same prompt hash) and verify output is byte-identical to stored outputs.
2. Re-run feature engineering from scratch and verify `model_ready_features.parquet` is identical to the stored version (checksum comparison).
3. Re-run all LOSO-CV models and verify AUC values match stored results to 4 decimal places.
4. Print final prompt manifest with model version, prompt hashes, and run timestamp.

### 8.3 Zenodo deposit

At submission:
- Anonymize all outputs (remove participant names, replace with `Participant_A`, `Participant_B` etc.).
- Upload repository to Zenodo with DOI.
- Include in paper methods: model version string, prompt hashes, Zenodo DOI, and instruction to run `0-build_registry.ipynb` through `9-reproducibility_check.ipynb` in sequence.

---

## How the results connect into the paper narrative

This project provides the empirical grounding for the companion article's central claims. The narrative should mirror the article's structure: from measurement gap → new instrument → validated capability → substantive findings → research agenda.

**Part 1 — The measurement gap (motivating the project)**: Current team science measures perceptions rather than behaviors, and captures snapshots rather than trajectories. The Scialog setting makes this gap concrete: team formation and grant funding are consequential outcomes, yet the behavioral dynamics of the meetings that precede them have never been systematically analyzed. This is where the paper's opening — the Gottman analogy, the survey critique — lands.

**Part 2 — The instrument (the pipeline contribution)**: We introduce a validated, reproducible multimodal AI annotation pipeline for virtual team meetings, demonstrating that LLMs can reliably annotate behavioral, vocal, and affective signals with measured agreement against trained human coders. The human-AI kappa table is the central methodological figure. The multimodal-over-transcript comparison establishes what video adds. This is the "microscope arrives" moment from the article.

**Part 3 — The thin slice finding (primary empirical result)**: Using beginning-segment models and the thin slice threshold analysis, we test whether early meeting behavior predicts team formation outcomes — directly operationalizing the Gottman/Jung premise in a naturalistic team science context. This should be the headline finding with an explicit AUC number the article can cite: "X minutes of observation predicted team formation with Y% accuracy."

**Part 4 — The behavioral mechanisms (secondary empirical result)**: The building/blocking ratio, psychological safety index, and cross-disciplinary bridging features are the specific behavioral signatures associated with outcomes. These answer "what specifically distinguishes meetings that produce teams from those that don't" — moving from prediction to mechanism, which is the article's "from prediction to mechanism" section goal.

**Part 5 — The affective trajectory (tertiary empirical result)**: Whether the temporal trajectory of engagement and convergence — not just its level — predicts outcomes. The delta features (end minus beginning for key behavioral measures) and the convergence_ratio_end variable test whether teams that shift from divergent exploration toward convergent commitment are more likely to form funded teams. This operationalizes the article's temporal dynamics dimension.

**Part 6 — Limitations and scope**: The pipeline is observational. Causal inference from meeting behavior to funding outcomes is not warranted. The virtual Zoom context means gaze-direction engagement is not available; Zoom-valid proxies are partial substitutes. Conference heterogeneity is controlled but not eliminated. The pipeline is presented as a validated measurement tool and a proof of concept for the research agenda the article proposes — not as a definitive predictive system.

---

## References cited in this plan

Bales, R. F. (1950). *Interaction process analysis: A method for the study of small groups*. Addison-Wesley.

Barsade, S. G. (2002). The ripple effect: Emotional contagion and its influence on group behavior. *Administrative Science Quarterly*, 47(4), 644–675.

Edmondson, A. (1999). Psychological safety and learning behavior in work teams. *Administrative Science Quarterly*, 44(2), 350–383.

Gottman, J. M., & Levenson, R. W. (2000). The timing of divorce: Predicting when a couple will divorce over a 14-year period. *Journal of Marriage and Family*, 62(3), 737–745.

Jung, M. F. (2016). Coupling interactions and performance: Predicting team performance from thin slices of conflict. *ACM Transactions on Computer-Human Interaction*, 23(3), 1–32.

Nijstad, B. A., & Stroebe, W. (2006). How the group affects the mind: A cognitive model of idea generation in groups. *Personality and Social Psychology Review*, 10(3), 186–213.

Pentland, A. (2012). The new science of building great teams. *Harvard Business Review*, 90(4), 60–70.

Uzzi, B., Mukherjee, S., Stringer, M., & Jones, B. (2013). Atypical combinations and scientific impact. *Science*, 342(6157), 468–472.

Woolley, A. W., Chabris, C. F., Pentland, A., Hashmi, N., & Malone, T. W. (2010). Evidence for a collective intelligence factor in the performance of human groups. *Science*, 330(6004), 686–688.

Wuchty, S., Jones, B. F., & Uzzi, B. (2007). The increasing dominance of teams in production of knowledge. *Science*, 316(5827), 1036–1039.

---

## Final checklist before submission

- [ ] Human-AI kappa reported for all annotation categories (Table 1)
- [ ] All AUC values are LOSO or LOCO — no best-split values reported
- [ ] All regression models include conference as fixed or random effect
- [ ] Sensitivity analyses completed and reported in supplement
- [ ] Video-vs-transcript comparison analysis included
- [ ] Causal language absent throughout manuscript
- [ ] Prompt text, model version, and prompt hashes in Methods section
- [ ] Zenodo DOI for data/code repository in Data Availability statement
- [ ] Benjamini-Hochberg correction applied to all univariate tests
- [ ] Effect sizes (Cohen's f², odds ratios with 95% CI) reported alongside p-values