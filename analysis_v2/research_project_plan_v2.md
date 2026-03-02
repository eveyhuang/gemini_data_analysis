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

- **Invalid transcript-only visual annotations**: The prior `analyze_text.py` prompt asked the model to annotate gaze, smiles, nods, and gestures from text — generating confabulated visual data. The new pipeline strictly separates what can be inferred from audio/video vs. transcript, and uses a full multimodal model (Gemini 2.5 Pro or equivalent) for all visual signal extraction.
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

```
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
```
        ↓
Stage 9: Reproducibility Packaging
```

Each stage corresponds to one or more notebooks or scripts described below.

---

## Annotation targets

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

### Layer 4: Multimodal-only signals per utterance (audio + video)

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

## Prompt Library

These are the exact, production-ready prompts to be saved as plain `.txt` files in the `prompts/` directory. They are referenced verbatim in the annotation scripts. Any change to a prompt text must be saved as a new file version and re-hashed in `prompt_manifest.json`.

---

### `prompts/pass1_chunk_prompt.txt` — Multimodal behavioral annotation (video + transcript)

```
SYSTEM ROLE:
You are an expert in team science, interaction analysis, multimodal behavioral coding, and deductive qualitative research. You have deep knowledge of team cognition, collective intelligence, interdisciplinary collaboration, and nonverbal communication in group settings.

CONTEXT:
You are analyzing a 5–10 minute video segment from a recorded Zoom meeting. The participants are scientists from diverse disciplinary backgrounds (e.g., biology, physics, chemistry, data science, engineering) who are collaborating to develop novel research ideas in response to a scientific challenge. These meetings are part of a Scialog conference, a structured scientific collaboration program. The goal of the meeting is to generate ideas that could lead to joint research teams and grant proposals.

YOUR TASK:
Analyze the video and transcript below and produce two types of structured annotations:

PART A — CHUNK-LEVEL SUMMARY: One holistic assessment of the entire 5–10 minute segment.
PART B — UTTERANCE-LEVEL CODING: One annotation object per speaker turn.

======================================================================
CRITICAL RULES — READ BEFORE ANNOTATING:
======================================================================
1. In PART B, annotate BEHAVIOR first, then add idea_quality only for the three applicable categories (Idea Management, Integration Practices, Knowledge Sharing). Do not add idea_quality to any other category.
2. Code only what is EXPLICITLY OBSERVABLE in the utterance or video. Do not infer intent, motivation, or background knowledge.
3. For any audio or video field where quality is insufficient (participant off-camera, poor lighting, overlapping audio, blurry frame), append [low_confidence] directly after that field's value. Example: "nod_count": "0 [low_confidence]"
4. If no behavioral code applies to an utterance (e.g., pure filler words like "yep", "mm-hmm", "okay", or utterances shorter than 5 words with no substantive content), assign code_name: "None" and leave other code fields empty.
5. Assign no more than 3 behavioral codes per utterance. Only assign multiple codes if each is clearly and independently present.
6. Do not hallucinate visual signals. If participants are not visible (e.g., camera off, out of frame), set nod_count to 0 and all binary visual fields to "No", and append [low_confidence] to each.
7. Apply Pronoun Framing [12] to EVERY utterance that discusses the research content — it is not optional.
======================================================================

======================================================================
PART A — CHUNK-LEVEL SUMMARY ANNOTATION
======================================================================
Produce a single JSON object "chunk_summary" with the following fields:

--- PARTICIPATION STRUCTURE ---
"speaking_time_seconds": {
  // For every participant who speaks in this chunk, estimate the number of seconds they speak.
  // Use the transcript timestamps to inform this estimate.
  // Format: {"SpeakerName": integer_seconds, ...}
  // Include ALL speakers who appear in the transcript for this chunk.
}
"dominant_speaker_flag": "Yes" or "No"
  // Yes if any single speaker accounts for more than 50% of total speaking time in this chunk.
"dominant_speaker_name": string or "None"
  // Name of the dominant speaker, or "None" if no dominant speaker.

--- IDEA TRAJECTORY ---
"idea_trajectory": one of ["divergent", "convergent", "procedural", "ambiguous"]
  // divergent: the group is primarily generating, exploring, or brainstorming new ideas.
  //            Participants are introducing new angles, asking "what if", expanding the space.
  // convergent: the group is primarily narrowing down, synthesizing, reaching consensus,
  //             making decisions, or committing to a direction.
  // procedural: the discussion is primarily logistical — coordinating who speaks when,
  //             scheduling, housekeeping, or meta-discussion about the meeting itself.
  // ambiguous: the chunk contains a genuine mix that cannot be clearly classified,
  //            or the discussion shifts between modes without a dominant one.
"idea_trajectory_justification": string
  // One sentence explaining why you assigned this trajectory label.
  // Example: "Participants generated four distinct hypotheses without evaluating any of them."

--- COLLECTIVE BEHAVIORAL RESPONSIVENESS ---
"collective_engagement_level": integer 1, 2, 3, or 4
  // A holistic rating of how behaviorally responsive the NON-SPEAKING participants appear
  // throughout this chunk, based ONLY on signals that are genuinely observable in Zoom video.
  //
  // VALID signals to observe (in order of interpretive confidence):
  //   HEAD NODS from non-speakers — clearly observable, strongest engagement signal.
  //   FACIAL EXPRESSIONS — smiles, raised eyebrows, frowns, visible reactions.
  //   AUDIBLE BACKCHANNELS — "mm-hmm", "yeah", laughter, brief affirmations from non-speakers.
  //   CAMERA STATUS — participants who turn cameras off mid-chunk signal reduced engagement.
  //   VISIBLE OFF-SCREEN DISTRACTION — only flag if clearly sustained: participant is
  //     obviously looking far off-screen, visibly using a phone in frame, or talking to
  //     someone off-camera. Do NOT flag brief glances or natural head movements.
  //   POSTURAL SHIFTS — visible lean toward or away from camera (note: lower confidence).
  //
  // Rating rubric:
  // 1 = Most visible non-speakers show NO behavioral responsiveness: no nods, no visible
  //     facial reactions, no audible backchannels throughout. Multiple cameras off, or
  //     visible sustained off-screen distraction from one or more participants.
  // 2 = Mixed: some participants show responsiveness (occasional nod, brief smile),
  //     others do not. Backchannel vocalizations are rare or absent.
  // 3 = Most visible non-speakers show consistent responsiveness: regular nods, visible
  //     expressions, audible backchannels from at least one participant.
  // 4 = Strong collective responsiveness throughout: frequent nods, shared laughter or
  //     smiling, active backchannel vocalizations from multiple participants.
"collective_engagement_justification": string
  // One sentence naming the specific OBSERVABLE signals (not inferred attention) that
  // led to this rating. Example: "Three participants nodded frequently and two produced
  // audible 'mm-hmm' backchannels; no visible off-screen distraction."
  // Do NOT write: "Participants appeared to be paying attention to the speaker."

--- CROSS-DISCIPLINARY BRIDGING ---
"cross_disciplinary_bridging": "Yes" or "No"
  // Yes if ANY participant in this chunk explicitly connects their own disciplinary framing
  // to another participant's. This means explicitly naming or referencing two frameworks,
  // fields, or terminologies and drawing a link between them.
  // Examples that qualify:
  //   "What you're calling a cascade effect is what we in network science call a contagion process."
  //   "In my field we'd approach this as an optimization problem, but I think your framing as
  //    an evolutionary system is actually capturing something we miss."
  // Examples that do NOT qualify:
  //   Simply mentioning your own field without connecting it to another.
  //   Using a technical term without bridging it to another participant's vocabulary.
"cross_disciplinary_bridging_speaker": string or "None"
  // Name of the participant who made the bridging move, or "None".
"cross_disciplinary_bridging_description": string or "None"
  // A brief phrase (≤20 words) describing what two disciplines or frameworks were connected.
  // Example: "Connected network contagion modeling (physics) to immune response cascades (biology)."
  // Write "None" if cross_disciplinary_bridging is "No".

--- EXPLICIT COMMITMENT SIGNALS ---
"explicit_commitment_signal": "Yes" or "No"
  // Yes if ANY participant explicitly expresses interest in forming a team, proposes a
  // specific future collaboration, references a joint next step involving multiple people,
  // or suggests writing a proposal together.
  // This must be an explicit verbal signal — not an implicit one.
  // Examples that qualify:
  //   "I'd really like to continue this conversation — would you be open to co-authoring something?"
  //   "Let's set up a call next week to sketch out a proposal."
  //   "I think the three of us have something here — we should submit together."
  // Examples that do NOT qualify:
  //   "This has been a great discussion."
  //   Expressing enthusiasm about the topic without committing to joint action.
"commitment_signal_speaker": string or "None"
  // Name of the participant who made the commitment signal, or "None".
"commitment_signal_quote": string or "None"
  // Verbatim quote of the commitment signal, 20 words or fewer. Write "None" if no signal.

--- COLLABORATIVE ARTIFACT ENGAGEMENT ---
"screenshare_active": "Yes" or "No"
  // Yes if a screen is being shared by any participant at any point during this chunk.
"artifact_type": one of ["document", "diagram", "data", "slides", "code", "whiteboard", "other", "None"]
  // Type of artifact being shared. Write "None" if no screenshare.
"artifact_interaction": "Yes" or "No" or "NA"
  // Yes if participants actively interact with the shared artifact during this chunk.
  // Interaction means: making live edits, pointing at elements, referencing specific parts
  // of the artifact in speech, annotating, or drawing.
  // No if screen is shared but participants do not interact with it (e.g., static slide as backdrop).
  // NA if no screenshare is active.
"artifact_interaction_description": string or "None"
  // One sentence describing what kind of interaction occurred with the artifact.
  // Example: "Speaker edited the diagram live while explaining a proposed feedback loop."
  // Write "None" if artifact_interaction is "No" or "NA".

--- INTELLECTUAL QUALITY AND SPECIFICITY ---
"problem_specificity_level": integer 1, 2, 3, or 4, or "NA"
  // Rate how specific the research problem the group is working on is by the END of this chunk.
  // 1 = Topic-level only: participants share general interest in an area ("we both work on X")
  //     but no specific question has been identified.
  // 2 = Domain narrowed: a more specific area has been identified, but no concrete question yet
  //     ("something about X in the context of Y").
  // 3 = Specific question: a concrete, answerable research question has been articulated
  //     ("can X mechanism explain Y phenomenon in Z system?").
  // 4 = Question + approach: a specific question with at least a sketch of how to answer it
  //     ("we think X causes Y via Z — we could test that by doing W with dataset Q").
  // Write "NA" if this chunk is entirely procedural (no research content discussed).
"problem_specificity_justification": string or "NA"
  // One sentence referencing specific language from the transcript that supports your rating.

"decision_crystallization_level": integer 1, 2, 3, or 4
  // Rate how crystallized the group's sense of a joint direction is by the END of this chunk.
  // 1 = No shared direction: still at open, individual exploration; no convergence visible.
  // 2 = Shared interest identified: participants agree there is something to explore together,
  //     but no specific joint project has been named.
  // 3 = Specific project named: a specific project idea has been articulated and acknowledged
  //     by at least two participants as worth pursuing.
  // 4 = Project with structure: a specific project plus at least TWO of the following:
  //     a research question, an approach, a timeline, a division of roles, a named next step.
  // This field is MOST important for the final chunk of each session.
"decision_crystallization_justification": string
  // One sentence explaining what specifically in this chunk justified this rating.

"ambition_level": one of ["incremental", "novel_application", "novel_combination",
                           "paradigm_challenging", "not_applicable"]
  // Classify the most ambitious idea proposed in this chunk.
  // incremental: extends existing work in an expected direction; no surprising combination.
  // novel_application: applies established methods or frameworks to a new domain.
  // novel_combination: combines two fields, methods, or frameworks in a way neither has done.
  //   This is the "atypical combination" associated with high-impact science (Uzzi et al., 2013).
  // paradigm_challenging: questions a foundational assumption of one or more of the fields involved.
  // not_applicable: no specific research idea was proposed in this chunk (procedural/social only).

--- COMPLEMENTARITY AND SHARED VISION ---
"explicit_complementarity_recognition": "Yes" or "No"
  // Yes if any participant explicitly articulates that their expertise or approach COMPLEMENTS
  // another participant's — meaning the COMBINATION is more capable than either alone.
  // This is different from noting that fields are related or overlap. The signal is valuation:
  //   "I could never do W without someone who does Y — that's exactly what's been missing."
  //   "Your experimental access is exactly what my models need to be testable."
  // Does NOT qualify: "Oh interesting, we both work on X." / "There's definitely overlap there."
"complementarity_recognition_speaker": string or "None"
"complementarity_recognition_quote": string or "None"
  // Verbatim phrase (≤20 words) capturing the complementarity articulation, or "None".

"skill_gap_identification": "Yes" or "No"
  // Yes if any participant identifies a specific gap in the proposed project AND connects
  // that gap to what another person in the room could provide.
  // "You'd be the one who could actually run the experiment — I don't have the lab for it."
  // "We'd need someone who knows the clinical side. Do you have those connections?"
  // Does NOT qualify: vague acknowledgments that the project is hard or incomplete.
"skill_gap_description": string or "None"
  // One sentence describing the gap and who was identified as filling it, or "None".

"shared_vision_indicator": "Yes" or "No"
  // Yes if the conversation shifted in this chunk from participants describing their OWN
  // separate work toward discussing a SHARED project that belongs jointly to the group.
  // Key linguistic signal: participants use "our," "we," or "together" when referring to
  // the proposed research (not just social politeness).
  // Distinct from commitment signals: shared vision is about language and framing, not
  // an explicit proposal to collaborate.
"shared_vision_quote": string or "None"
  // The phrase or exchange that most clearly marks the shift, or "None".

"pronoun_shift_flag": "Yes" or "No"
  // Yes if this chunk shows a NOTABLE SHIFT from individual framing ("my work," "your work,"
  // "in my field") toward joint framing ("our idea," "we could," "together we") when
  // discussing the proposed research.
  // Code Yes only when the shift OCCURS in this chunk — not if joint language has been
  // present consistently since the beginning.

--- INTERPERSONAL AND RELATIONAL SIGNALS ---
"personal_disclosure": "Yes" or "No"
  // Yes if any participant shares something personally revealing that goes beyond
  // professional role presentation: a research frustration, a career aspiration,
  // a domain passion, a past failure, or an unexpected personal finding.
  // Do NOT flag polite self-introductions or standard academic presentations.
  // Only flag disclosures that are personally candid or emotionally revealing.

"laughter_quality": one of ["tension_release", "shared_humor", "appreciative",
                             "social_lubricant", "none"]
  // If laughter or shared humor occurred, classify its primary function.
  // tension_release: laughter at/after a moment of disagreement, awkwardness, or challenge.
  // shared_humor: laughter in response to a joke or playful remark — social warmth.
  // appreciative: laughter or amusement in direct response to an idea being CLEVER,
  //   SURPRISING, or ELEGANT — signals intellectual rapport and is most predictive of
  //   team formation.
  // social_lubricant: light background laughter with no clear trigger.
  // none: no laughter occurred.

--- PSYCHOLOGICAL SAFETY AND DISSENT ---
"dissent_response_quality": integer 1, 2, 3, or "NA"
  // When any participant expresses disagreement, raises a concern, or proposes a contrarian
  // view in this chunk, how does the group respond?
  // 1 = Dismissive or defensive: dissent is interrupted, dropped without engagement, or the
  //     response signals the dissent was unwelcome.
  // 2 = Neutral: dissent is acknowledged but not deeply engaged with; conversation moves on.
  // 3 = Curious and exploratory: dissent is met with follow-up questions, elaboration requests,
  //     or genuine engagement. Disagreement is treated as useful information.
  // "NA" = No dissent or contrarian view was expressed in this chunk.
  // Based on Edmondson (1999): psychological safety is observable in how disagreement is received.

--- INTELLECTUAL RISK AND AMBITION ---
"risk_acknowledgment_with_enthusiasm": "Yes" or "No"
  // Yes if any participant explicitly acknowledged that the proposed project is risky,
  // uncertain, or hard — AND responded to that acknowledgment with excitement or positive
  // affect rather than hedging, qualification, or retreat.
  // BOTH elements must be present: risk acknowledgment + enthusiasm about the risk.
  // "This is going to be really hard, but that's exactly why it's exciting."
  // "I don't know if it's feasible, but the potential payoff is enormous."
  // Does NOT qualify: acknowledging risk without positive affect, or enthusiasm without
  // acknowledging risk.
"risk_enthusiasm_quote": string or "None"
  // Verbatim phrase (≤20 words) or "None".

--- GRANT AND FUNDING CONTEXT ---
"funding_awareness_signal": "Yes" or "No"
  // Yes if any participant mentions a specific funding mechanism, program priority,
  // grant deadline, review criterion, or funding agency relevant to the proposed work.
  // "Scialog is specifically looking for cross-disciplinary proposals" qualifies.
  // "We should think about funding eventually" does NOT qualify.
"funding_reference_description": string or "None"
  // One sentence describing the specific funding reference, or "None".

"prior_relationship_signal": "Yes" or "No"
  // Yes if any participant mentions prior familiarity with another participant:
  // having read their work, met at a conference before, collaborated previously,
  // or known of each other's research before this meeting.
  // Prior familiarity reduces coordination costs and predicts collaboration follow-through.
"prior_relationship_description": string or "None"
  // One sentence describing the nature of the prior connection, or "None".

--- MEETING PROCESS QUALITY ---
"meeting_structure_quality": one of ["unstructured", "loosely_structured", "structured"]
  // Rate the degree to which this chunk has an implicit or explicit structure.
  // unstructured: conversation is associative; topic jumps without apparent shared plan.
  // loosely_structured: participants have a general shared sense of what they're doing,
  //   but no explicit agenda or reference to phases.
  // structured: participants explicitly reference phases, topics to cover, what has been
  //   accomplished, or what remains; discussion is deliberately organized.

--- NOTABLE OBSERVATION ---
"notable_observation": string or "None"
  // A brief (1–3 sentence) note on anything surprising, unusual, or potentially important
  // that occurred in this chunk that is NOT captured by any field above, but could be
  // relevant to predicting team formation success or grant proposal quality.
  // Examples: an unexpected moment of genuine laughter that broke tension, a participant
  // abruptly changing their position, a striking personal anecdote, an unusually creative
  // leap, or a visible moment of disengagement that seems significant.
  // Set to "None" if nothing stands out.

======================================================================
PART B — UTTERANCE-LEVEL BEHAVIORAL CODING
======================================================================
Produce a JSON array "utterance_annotations" with one object per speaker turn.
Each object must have the following fields:

--- IDENTIFICATION ---
"speaker": string — Full name of the speaker.
"timestamp": string — Start and end time in MM:SS-MM:SS format.
"speaking_duration_seconds": integer — Estimated seconds this turn lasts.

--- BEHAVIORAL CODES (LAYER 2) ---
"codes": array of code objects, or [{"code_name": "None"}] if no code applies.
  // Assign 1–3 codes. Each code object has:
  //   "code_name": one of the 16 categories listed below
  //   "subcode": string or "None" — a more specific label within the category (see below)
  //   "evidence": string — verbatim quote from the utterance that directly supports the code.
  //   "explanation": string — 1–2 sentences describing what in the utterance justifies
  //                  this code.
  //   "idea_quality": integer 0, 1, or 2 — ONLY for these three categories:
  //                   Idea Management, Integration Practices, Knowledge Sharing.
  //                   Omit this field entirely for all other code categories.
  //
  //   idea_quality rubric (for the three applicable categories only):
  //   0 = Undeveloped, vague, tangential, or already well-known to the group. The utterance
  //       does not meaningfully advance the team's intellectual work.
  //   1 = Clear, relevant, and specific. Another participant could meaningfully respond to it.
  //   2 = Notably novel, richly developed, fills a specific gap, or opens a new direction
  //       the group had not yet considered. Clearly advances the team's work.
  //
  //   Examples of when to include idea_quality:
  //   → code_name is "Idea Management" → always include idea_quality
  //   → code_name is "Integration Practices" → always include idea_quality
  //   → code_name is "Knowledge Sharing" → always include idea_quality
  //   → code_name is "Relational Climate" → DO NOT include idea_quality
  //   → code_name is "Pronoun Framing" → DO NOT include idea_quality

BEHAVIORAL CODE CATEGORIES AND SUBCODES:

[1] Idea Management
  // The speaker introduces, develops, extends, or responds to an idea.
  Subcodes: "proposes_new_idea" | "extends_existing_idea" | "combines_ideas" |
            "returns_to_earlier_idea" | "redirects_idea"

[2] Information Seeking
  // The speaker asks a question or requests information, clarification, or elaboration
  // from another participant.
  Subcodes: "asks_factual_question" | "asks_clarifying_question" |
            "asks_for_elaboration" | "asks_for_opinion" | "asks_rhetorical_question"

[3] Knowledge Sharing
  // The speaker contributes domain knowledge, data, findings, methods, or expertise
  // relevant to the discussion topic.
  Subcodes: "shares_domain_knowledge" | "shares_data_or_findings" |
            "shares_method_or_approach" | "shares_personal_experience"

[4] Evaluation Practices
  // The speaker evaluates, critiques, challenges, validates, or compares an idea or claim.
  Subcodes: "supports_or_validates" | "critiques_or_challenges" |
            "compares_options" | "raises_concern" | "devil_advocate" |
            "setback_response_explores" | "setback_response_defends" |
            "setback_response_redirects" | "setback_response_accepts_builds"
  //
  // The setback_response subcodes apply when the speaker responds to criticism of their
  // OWN idea or to an obstacle raised about the team's current direction:
  //   setback_response_explores: speaker asks follow-up questions about the criticism,
  //     seeking to understand it rather than counter it.
  //   setback_response_defends: speaker argues against the criticism without engaging
  //     its content; position is held rather than examined.
  //   setback_response_redirects: speaker changes subject or moves on without addressing
  //     the concern.
  //   setback_response_accepts_builds: speaker acknowledges the concern and incorporates
  //     it into a revised or strengthened proposal.

[5] Relational Climate
  // The speaker's utterance primarily serves a relational or social function rather than
  // a cognitive one: building rapport, expressing appreciation, managing tension,
  // encouraging participation, or expressing humor.
  Subcodes: "expresses_appreciation" | "encourages_participation" |
            "manages_tension" | "uses_humor" | "expresses_enthusiasm"

[6] Participation Dynamics
  // The utterance explicitly invites, redirects, or manages who speaks or contributes.
  // This includes facilitating, yielding the floor, or gatekeeping.
  Subcodes: "invites_contribution" | "yields_floor" | "redirects_speaker" |
            "summarizes_for_group" | "gatekeeps"

[7] Coordination and Decision Practices
  // The speaker proposes or manages a process, agenda, decision rule, or next step
  // for the group as a whole.
  Subcodes: "proposes_process" | "calls_for_decision" | "records_or_documents" |
            "proposes_next_step" | "checks_consensus" | "scope_calibration"
  //
  // scope_calibration: speaker explicitly discusses whether the project scope is
  //   appropriate for a grant proposal — too big, too small, or about right.
  //   "We probably can't answer all of that in one grant cycle." / "This is actually
  //   a perfect scale for an NIH R01."

[8] Integration Practices
  // The speaker synthesizes, connects, or reconciles contributions from multiple people
  // or multiple ideas into a coherent whole.
  Subcodes: "synthesizes_contributions" | "identifies_common_ground" |
            "resolves_contradiction" | "frames_shared_problem"

[9] Idea Ownership and Attribution
  // The speaker explicitly claims ownership of an idea or attributes an idea to a
  // specific person. Tracks intellectual investment and credit dynamics.
  Subcodes: "claims_own_idea" | "attributes_to_other" | "challenges_attribution"

[10] Future-Oriented Language
  // The utterance explicitly references future joint work, shared plans, next steps, or
  // proposed continuation beyond this meeting. Must be explicit, not implied.
  Subcodes: "vague_future_reference" | "specific_future_plan" | "named_next_step"
  // vague_future_reference: "We should think more about this" / "It would be interesting..."
  // specific_future_plan: "Let's schedule a follow-up call" / "We could submit in March."
  // named_next_step: a concrete action with a responsible person and approximate timeline.
  //   "I'll send you my dataset by end of week." / "Can you draft the outline by Friday?"

[11] Epistemic Bridging
  // The speaker explicitly translates a concept, term, or framework across disciplinary
  // boundaries. Must explicitly name or invoke two different frameworks or fields.
  Subcodes: "translates_terminology" | "connects_methods" | "reframes_cross_disciplinarily"

[12] Pronoun Framing  *** APPLY TO ALL SUBSTANTIVE UTTERANCES ABOUT THE RESEARCH ***
  // How does the speaker frame the proposed research — as belonging to individuals separately
  // or as a joint shared endeavor? Apply this code to every utterance that discusses the
  // research content (not to social pleasantries or procedural talk).
  Subcodes: "individual_framing" | "joint_framing" | "ambiguous"
  // individual_framing: speaker refers to their own work or the other's work as separate
  //   ("my research," "your approach," "in my field we do X"). The research is described
  //   as belonging to individuals.
  // joint_framing: speaker uses "we," "our," "together," or equivalent when describing
  //   the proposed work — the research is described as a shared project.
  // ambiguous: the utterance discusses the research but cannot be clearly classified as
  //   either individual or joint framing.
  // Track this across the session — the SHIFT from individual to joint framing is
  // theoretically the key moment of team formation.

[13] Complementarity Articulation
  // The speaker explicitly names how their expertise or approach COMPLEMENTS another
  // participant's — articulating that the COMBINATION is more capable than either alone.
  // This is different from noting that fields are related or overlap.
  Subcodes: "expertise_complementarity" | "resource_complementarity" | "method_complementarity"
  // expertise_complementarity: "Your theoretical framing is exactly what my empirical work
  //   has been missing."
  // resource_complementarity: "I have the cohort data; you have the computational tools."
  // method_complementarity: "Your experimental approach combined with my modeling could
  //   test things neither of us can test alone."

[14] Role Anticipation
  // The speaker begins to map out — implicitly or explicitly — who would do what in the
  // proposed collaboration. Forward-modeling the collaboration as a real future project.
  Subcodes: "explicit_role_assignment" | "implicit_role_suggestion"
  // explicit_role_assignment: "You'd handle the theoretical side; I'd run the experiments."
  // implicit_role_suggestion: "I could see myself doing the field work while someone with
  //   your modeling background handles the analysis."

[15] Broader Significance
  // The speaker articulates why the proposed work matters BEYOND the immediate research
  // question — to the field, to science generally, to society, or to a funding priority.
  // Grant reviewers weight broader impact heavily; teams that discuss it in initial meetings
  // are more likely to build compelling broader impact sections into proposals.
  Subcodes: "field_significance" | "societal_significance" | "funding_priority_alignment"
  // field_significance: "This would settle a debate that's been open for 20 years."
  // societal_significance: "If we can solve this, it has direct implications for X."
  // funding_priority_alignment: "This fits exactly what [agency] is prioritizing right now."

[16] Idea Novelty Signal
  // Any participant explicitly marks an idea as surprising, unexpected, or unlike prior
  // approaches. This is distinct from enthusiasm (which could be social) — it specifically
  // marks PERCEIVED GENUINE NOVELTY of the intellectual content.
  Subcodes: "novelty_recognized_self" | "novelty_recognized_other"
  // novelty_recognized_self: speaker marks their own idea as novel or unexpected.
  // novelty_recognized_other: speaker reacts to another's idea as novel.
  // Examples: "I've never thought about it that way." / "Has anyone actually done that
  //   combination before?" / "That's a really different angle on the problem."

--- INTERRUPTION TYPE ---
"interruption_type": one of ["not_interruption", "collaborative_completion",
                              "elaborative_jump_in", "competitive_interruption"]
  // not_interruption: this turn followed a complete turn by the previous speaker.
  // collaborative_completion: the speaker begins talking before the prior speaker finishes
  //   and completes or extends their sentence in agreement or support.
  // elaborative_jump_in: the speaker begins before the prior speaker finishes and
  //   adds new content in support — not finishing their sentence but jumping in helpfully.
  // competitive_interruption: the speaker cuts off the prior speaker to redirect,
  //   contradict, or take the floor away.

--- MULTIMODAL SIGNALS (VIDEO + AUDIO — DO NOT ESTIMATE FROM TRANSCRIPT ALONE) ---
// The following fields require direct video and audio observation.
// If participants are off-camera or video quality is insufficient, apply [low_confidence].
//
// Only annotate the specific behaviors listed below, which remain genuinely observable.

// --- Off-screen distraction (conservative — only flag the obvious) ---
"visible_off_screen_distraction": "Yes" or "No"
  // Yes ONLY if at least one non-speaking participant is clearly and SUSTAINEDLY showing
  // off-screen distraction. Do NOT flag brief eye movements or natural head shifts.
  // When in doubt, write "No".
"distracted_participant_count": integer — Set to 0 if visible_off_screen_distraction is "No".

// --- Facial and vocal responsiveness from non-speakers ---
"nod_count": integer
  // Total number of visible head nods from non-speaking participants during this turn.
  // Count individual nod gestures, not nodders (one person nodding 3 times = 3).
  // Set to 0 if no nods are visible. Append [low_confidence] if video quality is poor.
"shared_affect": "Yes" or "No"
  // Yes if 2 or more participants simultaneously display a positive emotional response
  // (smiling, laughing, leaning in with visible positive affect) during this turn.
"any_smile_other": "Yes" or "No"
  // Yes if at least one non-speaking participant appears to be smiling during this turn.
"audible_backchannel": "Yes" or "No"
  // Yes if any non-speaking participant produces an audible backchannel vocalization
  // during this turn: "mm-hmm", "yeah", "right", laughter, or other brief verbal response
  // that does not constitute a full turn.

// --- Speaker vocal affect ---
"vocal_enthusiasm": integer 1, 2, 3, or 4
  // Rating of the SPEAKER's vocal energy, pitch variation, and expressiveness.
  // Base this on audio only — do not conflate with content quality.
  // 1 = Flat, monotone, very low energy — difficult to distinguish emphasized words.
  // 2 = Moderate, conversational energy — some variation in pace and pitch.
  // 3 = Noticeably energetic and engaged — clear emphasis, varied pace, expressive.
  // 4 = High energy, passionate, emphatic — speaker sounds genuinely excited or invested.
"hesitation_flag": "Yes" or "No"
  // Yes if there is a notable pause or audible hesitation (longer than ~2 seconds,
  // or repeated false starts) BEFORE the speaker's key claim or main point in this turn.
"pace": one of ["fast", "normal", "slow"]
  // fast: noticeably faster than typical conversational pace.
  // normal: conversational pace, neither rushed nor labored.
  // slow: noticeably slower than typical, potentially for emphasis or uncertainty.

======================================================================
OUTPUT FORMAT:
======================================================================
Return a single, valid JSON object with exactly THREE top-level keys:
  "chunk_summary": { the Part A object }
  "utterance_annotations": [ { utterance object 1 }, { utterance object 2 }, ... ]
  "session_state": {
    // Compact handoff for the next chunk. Always produce this, even for the first chunk.
    "pronoun_shift_occurred_cumulative": bool,
      // True if a pronoun shift (individual → joint framing) has occurred in ANY chunk
      // up to and including this one. Once true, remains true for the rest of the session.
      // Carry forward true from prior session_state if it was already true.
    "shared_vision_established_cumulative": bool,
      // True if shared_vision_indicator was "Yes" in ANY prior or current chunk.
      // Carry forward true from prior session_state if it was already true.
    "idea_trajectory_sequence": [string, ...]
      // The full sequence of idea_trajectory labels from all chunks processed so far.
      // Append this chunk's idea_trajectory value to whatever was passed in as prior context.
      // If no prior context, this is a one-element array: [this chunk's idea_trajectory].
    "ideas_currently_on_table": [string, ...]
      // Up to 5 distinct research ideas that are active at the end of this chunk.
      // Each idea described in ≤15 words. Carry forward ideas from prior context,
      // add new ones raised in this chunk, remove any explicitly rejected or dropped.
    "commitment_signals_cumulative": [{"speaker": string, "quote": string}, ...]
      // All commitment signals from all prior chunks PLUS any new ones from this chunk.
      // Carry forward the list from prior session_state; append a new entry only if
      // explicit_commitment_signal is "Yes" in this chunk.
    "speakers_identified": [string, ...]
      // Complete list of all speaker names seen across all chunks so far including this one.
      // Use consistent full names. Carry forward from prior session_state and add new names.
    "chunk_summaries": [string, ...]
      // A list of per-chunk summaries, one entry per chunk processed so far INCLUDING this one.
      // Carry forward all entries from the prior session_state's chunk_summaries list unchanged,
      // then APPEND a new entry for THIS chunk at the end.
      // Each entry is 2–3 sentences describing ONLY that chunk's content — not the full session.
      // Cover: (1) what ideas or topics were raised or developed, (2) the relational/affective
      // tone and any notable dynamics (who drove the discussion, any tension or strong rapport),
      // and (3) where the group stood at the end of that chunk (convergence, open questions,
      // commitments made). Write in past tense. Do not list fields — write flowing prose.
      // Example entry for chunk 1: "Emily proposed using city sensor networks for CO2
      // measurement and Alissa raised the possibility of extending this to indoor air capture;
      // Emily drove most of the ideation while Alissa asked clarifying questions. The tone was
      // collaborative with visible head nods and active backchannels. The group ended the chunk
      // in early divergent exploration with no specific project named."
  }

Do not include any text outside the JSON object.
Do not include markdown code fences.
Ensure all string values use double quotes.
Every speaker turn in the transcript must have a corresponding utterance annotation object.
```

---


### Prompt usage rules and versioning

| Prompt file | Used in | Input modalities | Temperature |
|---|---|---|---|
| `pass1_chunk_prompt.txt` | `analyze_video.py` | Video + transcript | 0 |
| `pass1_transcript_only_prompt.txt` | `analyze_video.py` (fallback) | Transcript only | 0 |
| ~~`pass2_quality_prompt.txt`~~ | *eliminated — see design note* | — | — |

Any edit to a prompt must:
1. Be saved under a new filename (e.g., `pass1_chunk_prompt_v2.txt`).
2. Update `prompt_manifest.json` with the new SHA-256 hash and model version.
3. Trigger a full re-annotation of all chunks using that prompt (not a partial update).
4. Be documented in the Git commit message with the reason for the change.

---

### Session context preamble (cross-chunk continuity)

Chunks within a session are annotated sequentially. To preserve context across chunks, `analyze_video.py` prepends a **SESSION CONTEXT** block to the base prompt before each chunk, using structured state carried forward from all prior chunks in the session.

**How it works:**
- Each chunk's annotation response includes a `session_state` object (third top-level key in the JSON output — see OUTPUT FORMAT above).
- Before annotating chunk N, `build_chunk_prompt()` injects a preamble above the base prompt containing the state produced by chunk N-1.
- For already-annotated chunks (e.g., on a re-run), state is reconstructed by loading the saved `.json` file, so context accumulates correctly even when restarting a partial run.

**What the preamble contains (shown to the model before the base prompt):**

```
======================================================================
SESSION CONTEXT — FROM PRIOR CHUNKS (do not re-annotate; use for continuity only)
======================================================================
This is chunk N of the session.

WHAT HAPPENED IN EACH PRIOR CHUNK:
  Chunk 1: <2–3 sentence summary of chunk 1 only>
  Chunk 2: <2–3 sentence summary of chunk 2 only>
  ...

pronoun_shift_already_occurred: true/false
shared_vision_already_established: true/false
decision_crystallization_at_end_of_last_chunk: <1–4>
problem_specificity_at_end_of_last_chunk: <1–4>
idea_trajectory_sequence_so_far: ["divergent", "ambiguous", ...]
ideas_currently_on_table: ["idea 1 in ≤15 words", ...]
commitment_signals_so_far: [{"speaker": "...", "quote": "..."}]
speakers_identified_so_far: ["Name1", "Name2", ...]
======================================================================
```

**Design rationale for `chunk_summaries` over a rolling summary:**
An earlier design used a single rolling `session_summary` that was rewritten each chunk to cover the entire session so far. This is lossy: by chunk 6, a 2–3 sentence compression of chunks 1–5 cannot preserve the specific texture of chunk 1. The per-chunk list is append-only — chunk 1's entry is carried forward verbatim to chunk 6, guaranteeing no information loss. At 2–3 sentences per chunk and ~9 chunks per session, the full list is still compact (~20–27 sentences total).


---

# DETAILED STEPS AND STAGES

## Stage 0: Preprocessing and chunk registry

**Notebook: `0-build_registry.ipynb`**

### 0.1 Load raw data sources

```
inputs/
  sessions/          ← one folder per session_id
    <session_id>/
      chunks/        ← mp4 files named <session_id>_chunk_<index>.mp4
  rosters/           ← CSV per conference: participant name, session_id, conference_id
  outcomes/          ← master outcome sheet: session_id, num_teams, num_funded_teams
```

### 0.2 Build canonical chunk registry

Create a DataFrame where each row is one chunk. Columns:

| Column | Type | Description |
|---|---|---|
| `chunk_id` | str | Unique ID: `<session_id>__chunk<index>` |
| `session_id` | str | Parent session identifier |
| `conference_id` | str | Parent conference identifier |
| `chunk_index` | int | Position of chunk within session (0-indexed) |
| `chunk_position` | str | `beginning` / `middle` / `end` (based on tertile of chunk_index within session) |
| `total_chunks_in_session` | int | Total number of chunks for this session |
| `video_path` | str | Absolute path to mp4 file |
| `transcript_path` | str | Absolute path to vtt file |
| `has_video` | bool | Whether mp4 exists for this chunk |
| `num_participants` | int | Number of participants in session (from roster) |
| `conference_id` | str | Conference membership for clustering control |
| `outcome_num_teams` | int | Session-level outcome |
| `outcome_num_funded_teams` | int | Session-level outcome |
| `outcome_has_teams` | int | Binary: 1 if num_teams > 0 |
| `outcome_has_funded_teams` | int | Binary: 1 if num_funded_teams > 0 |
| `annotation_status` | str | `pending` / `annotated` / `failed` / `flagged_for_review` |
| `human_validation_set` | bool | Whether this chunk is in the human validation sample |

### 0.3 Stratified sampling for human validation set

Sample ~15–20% of chunks for human expert validation. Stratify by:
- `conference_id` (at least 2 chunks per conference)
- `chunk_position` (beginning/middle/end represented proportionally)
- `outcome_has_funded_teams` (balance positive and negative outcome sessions)

Set `human_validation_set = True` for sampled chunks. Save this assignment to the registry before any annotation runs so sampling is not influenced by annotation results.

### 0.4 Version control and prompt hashing

- Save registry as `data/chunk_registry_v1.parquet` (versioned filename).
- Store all prompt strings in `prompts/` directory as plain `.txt` files.
- Compute SHA-256 hash of each prompt file and store in `prompts/prompt_manifest.json` alongside model version string (e.g., `gemini-2.5-pro-preview-05-06`).
- Every annotation output file stores: `prompt_hash`, `model_version`, `temperature`, `timestamp_utc`.

### 0.5 Output

```
data/
  chunk_registry_v1.parquet
prompts/
  pass1_chunk_prompt.txt
  pass1_utterance_prompt.txt
  # pass2_quality_prompt.txt  ← eliminated (see design note in Prompt Library)
  prompt_manifest.json        ← {prompt_name: sha256_hash, model_version: ...}
```

---

## Stage 1: Multimodal annotation (behavioral coding + inline quality)

**Script: `annotate.py`** (called from notebook `1-annotate.ipynb`)

### 1.1 Input per chunk

- Full video file (mp4) uploaded to Gemini Files API.
- Full chunk transcript (parsed from .vtt) formatted as: `Speaker (MM:SS-MM:SS): utterance text`.
- Pass 1 prompt (loaded from `prompts/pass1_chunk_prompt.txt`).

### 1.2 Prompt reference

Use the exact text from `prompts/pass1_chunk_prompt.txt` as defined in the **Prompt Library** section above. Load it at runtime with `{transcript}` substituted as the full formatted chunk transcript string. If the video file is unavailable for a chunk, fall back to `prompts/pass1_transcript_only_prompt.txt` and set `multimodal_source = "transcript_only"` in the registry for that chunk. Both prompts produce the same JSON schema — null values in the transcript-only version preserve schema compatibility for downstream processing.

### 1.3 Execution logic (`annotate.py`)

```python
for chunk_id in registry[registry.annotation_status == 'pending'].chunk_id:
    chunk = registry.loc[chunk_id]

    # Upload video to Gemini Files API (reuse if already uploaded this session)
    video_file = upload_or_retrieve(chunk.video_path)

    # Parse transcript
    transcript_str = parse_vtt(chunk.transcript_path)

    # Build prompt
    prompt = load_prompt('prompts/pass1_chunk_prompt.txt').format(
        transcript=transcript_str
    )

    # Call Gemini (temperature=0 for reproducibility)
    response = gemini_client.generate_content(
        model=MODEL_VERSION,
        contents=[video_file, prompt],
        generation_config={'temperature': 0, 'response_mime_type': 'application/json'}
    )

    # Save raw response
    output = {
        'chunk_id': chunk_id,
        'model_version': MODEL_VERSION,
        'prompt_hash': PROMPT_HASH,
        'temperature': 0,
        'timestamp_utc': datetime.utcnow().isoformat(),
        'pass': 1,
        'raw_response': response.text
    }
    save_json(output, f'outputs/pass1/{chunk_id}.json')

    # Update registry status
    registry.loc[chunk_id, 'annotation_status'] = 'annotated_pass1'
```

- On API failure or JSON parse error: set `annotation_status = 'failed'`, log error, continue to next chunk.
- Retry failed chunks up to 2 times with exponential backoff before marking as `failed`.
- Run in batches to respect API rate limits; log progress to `logs/pass1_run.log`.

### 1.4 Output structure

```
outputs/
  pass1/
    <chunk_id>.json        ← one file per chunk
```

Each file contains the raw model JSON plus metadata fields (`chunk_id`, `model_version`, `prompt_hash`, `temperature`, `timestamp_utc`).

---

## Stage 2: Schema validation and quality flagging

**Notebook: `2-validate_schema.ipynb`**

### 2.1 Define Pydantic schemas

Define strict schemas for:
- `ChunkSummary` (chunk-level annotation object)
- `UtteranceAnnotation` (utterance-level object from Pass 1)

All fields are typed. Enum constraints are enforced (e.g., `idea_trajectory` must be one of `divergent`, `convergent`, `procedural`, `ambiguous`). Integer fields have valid ranges. String fields have max length.

### 2.2 Validate all outputs

```python
validation_results = []
for chunk_id in registry.chunk_id:
    raw = load_json(f'outputs/pass1/{chunk_id}.json')
    try:
        validated = AnnotatedChunk.model_validate(raw)
        status = 'valid'
        issues = []
    except ValidationError as e:
        status = 'schema_error'
        issues = e.errors()

    # Count low_confidence flags
    lc_count = count_low_confidence_flags(raw)
    lc_rate = lc_count / total_fields(raw)

    validation_results.append({
        'chunk_id': chunk_id,
        'validation_status': status,
        'issues': issues,
        'low_confidence_rate': lc_rate,
        'flagged_for_review': lc_rate > 0.30 or status == 'schema_error'
    })

registry = registry.merge(pd.DataFrame(validation_results), on='chunk_id')
```

### 2.3 Triage flagged chunks

Chunks with `flagged_for_review = True` are either:
- Re-queued for a second annotation attempt (if schema error or high low-confidence rate due to API issue), or
- Moved to `manual_review/` for human inspection if they fail twice.

Report summary statistics: total chunks, % valid on first attempt, % flagged, % failed after retry.

### 2.4 Output

```
data/
  chunk_registry_v2.parquet     ← updated registry with validation columns
  validation_report.csv         ← per-chunk validation summary
  manual_review/                ← chunks requiring human inspection
```

---

## Stage 3: Human expert validation and agreement computation

**Notebook: `3-human_validation.ipynb`**

This is the methodological centerpiece. It must be completed before any features are used in predictive models.

### 3.1 Prepare human coding materials

For each chunk in the `human_validation_set`:
1. Export a human-readable coding sheet: transcript formatted as speaker turns, one row per utterance, with empty columns for each code category.
2. Include the codebook (`code_book_v4` + 3 new categories) as a reference PDF.
3. Export the chunk video clip for rater viewing.
4. Do NOT expose AI annotations to raters at this stage.

Deliver materials via a secure shared folder (e.g., encrypted Google Drive). Each rater receives a unique copy so annotations remain independent.

### 3.2 Rater training protocol

- Minimum 2 raters with background in team science, organizational behavior, or conversation analysis.
- Calibration phase: 3 practice chunks coded together with discussion.
- Reliability check phase: 5 chunks coded independently; compute preliminary kappa. If kappa < 0.60 on any category, revise codebook anchors and repeat. These calibration chunks are excluded from the final validation sample.
- Main validation phase: remaining ~200–300 chunks coded independently (no discussion until after coding is complete).

### 3.3 Compute inter-rater reliability (human vs. human)

```python
from sklearn.metrics import cohen_kappa_score
from pingouin import intraclass_corr

for category in code_categories:
    # Categorical codes: Cohen's kappa
    kappa = cohen_kappa_score(rater1_codes[category], rater2_codes[category])

    # Ordinal/continuous (quality scores): ICC(2,1) two-way mixed
    icc = intraclass_corr(data=scores_df, targets='utterance_id',
                          raters='rater_id', ratings=category)

    human_irr_results[category] = {'kappa': kappa, 'icc': icc}
```

### 3.4 Compute human-AI agreement

Using majority-vote human code as reference (or single rater if only 2, resolved by discussion on disagreements):

```python
for category in code_categories:
    ai_codes = [pass1_data[chunk_id][utterance_id][category] for ...]
    human_codes = [human_coding[chunk_id][utterance_id][category] for ...]

    kappa_ai = cohen_kappa_score(human_codes, ai_codes)
    agreement_results[category] = {'human_ai_kappa': kappa_ai}
```

### 3.5 Feature inclusion decision rule

- If human-AI kappa ≥ 0.60 for a category: **include** all features derived from that category in predictive models.
- If human-AI kappa is 0.40–0.59: **include with caveat** — feature is included but flagged as "moderate reliability" in the paper; sensitivity analyses exclude it.
- If human-AI kappa < 0.40: **exclude** from predictive models. Revise prompt and re-annotate if the feature is theoretically important. Document exclusion explicitly in paper.

### 3.6 Output

```
data/
  human_coding/
    rater1_codes.csv
    rater2_codes.csv
    resolved_codes.csv          ← majority vote / reconciled
  validation/
    human_irr_results.csv       ← kappa, ICC per category, human vs. human
    human_ai_agreement.csv      ← kappa, ICC per category, human vs. AI
    feature_inclusion_decision.csv  ← which features pass the reliability threshold
```

Report: a table of all annotation categories with human IRR and human-AI agreement — this becomes Table 1 in the paper.

---

## Stage 4: Feature engineering

**Notebook: `4-feature_engineering.ipynb`**

All feature engineering is deterministic and version-controlled. Only features that passed the reliability threshold in Stage 4 are included.

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
# Blocking: negative-scored utterances + competitive interruptions
blocking_count = (
    count_utterances_with_score(utterances, score=-1) +
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
# camera_on_rate: proportion of non-speaking participants with cameras on across utterances in chunk
chunk_features['camera_on_rate'] = mean(
    cameras_on_count / (cameras_on_count + cameras_off_count + epsilon) for each utterance
)
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

> **Note**: The `mean_attending_ratio` and `mean_disengaged_ratio` features from earlier pipeline versions are removed. These were based on gaze-direction coding, which is not valid in Zoom recordings. The new `responsiveness_index`, `camera_on_rate`, `mean_nod_rate`, and `pct_turns_audible_backchannel` features are the Zoom-valid replacements and should be used in all models.

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