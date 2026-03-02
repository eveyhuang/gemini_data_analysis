import ffmpeg
import re
import time, json, os
from google import genai
from dotenv import load_dotenv
from google.genai.errors import ServerError
import subprocess
import unicodedata
import argparse


def init(prompt_type='scialog'):
    load_dotenv()
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    client = genai.Client(api_key=GOOGLE_API_KEY)

    ## pass1_chunk_prompt (multimodal behavioral annotation — video + transcript)
    ## Source: research_project_plan_v2.md → prompts/pass1_chunk_prompt.txt
    prompt = """
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
  // led to this rating. Do NOT write: "Participants appeared to be paying attention to the speaker."

--- CROSS-DISCIPLINARY BRIDGING ---
"cross_disciplinary_bridging": "Yes" or "No"
  // Yes if ANY participant in this chunk explicitly connects their own disciplinary framing
  // to another participant's. This means explicitly naming or referencing two frameworks,
  // fields, or terminologies and drawing a link between them.
"cross_disciplinary_bridging_speaker": string or "None"
"cross_disciplinary_bridging_description": string or "None"
  // A brief phrase (<=20 words) describing what two disciplines or frameworks were connected.
  // Write "None" if cross_disciplinary_bridging is "No".

--- EXPLICIT COMMITMENT SIGNALS ---
"explicit_commitment_signal": "Yes" or "No"
  // Yes if ANY participant explicitly expresses interest in forming a team, proposes a
  // specific future collaboration, references a joint next step involving multiple people,
  // or suggests writing a proposal together. Must be an explicit verbal signal.
"commitment_signal_speaker": string or "None"
"commitment_signal_quote": string or "None"
  // Verbatim quote of the commitment signal, 20 words or fewer. Write "None" if no signal.

--- COLLABORATIVE ARTIFACT ENGAGEMENT ---
"screenshare_active": "Yes" or "No"
"artifact_type": one of ["document", "diagram", "data", "slides", "code", "whiteboard", "other", "None"]
"artifact_interaction": "Yes" or "No" or "NA"
  // Yes if participants actively interact with the shared artifact (live edits, pointing,
  // referencing specific elements). No if screen is shared but not interacted with. NA if no screenshare.
"artifact_interaction_description": string or "None"

--- INTELLECTUAL QUALITY AND SPECIFICITY ---
"problem_specificity_level": integer 1, 2, 3, or 4, or "NA"
  // 1 = Topic-level only ("we both work on X")
  // 2 = Domain narrowed but no specific question
  // 3 = Specific research question articulated
  // 4 = Specific question with hypothesis and approach
  // Write "NA" if this chunk is entirely procedural.
"problem_specificity_justification": string or "NA"

"decision_crystallization_level": integer 1, 2, 3, or 4
  // 1 = No shared direction; still at open exploration
  // 2 = Shared interest identified but no specific project
  // 3 = A specific project idea named and agreed upon by at least two participants
  // 4 = A specific project with at least two of: research question, approach, timeline,
  //     division of roles, or named next step
"decision_crystallization_justification": string

"ambition_level": one of ["incremental", "novel_application", "novel_combination",
                           "paradigm_challenging", "not_applicable"]
  // incremental: extends existing work in an expected direction; no surprising combination.
  // novel_application: applies established methods to a new domain.
  // novel_combination: combines two fields/methods/frameworks in a way neither has done.
  // paradigm_challenging: questions a foundational assumption of one or more fields.
  // not_applicable: no specific research idea was proposed (procedural/social only).

--- COMPLEMENTARITY AND SHARED VISION ---
"explicit_complementarity_recognition": "Yes" or "No"
  // Yes if any participant explicitly articulates that their expertise COMPLEMENTS
  // another's — the COMBINATION is more capable than either alone.
"complementarity_recognition_speaker": string or "None"
"complementarity_recognition_quote": string or "None"
  // Verbatim phrase (<=20 words) or "None".

"skill_gap_identification": "Yes" or "No"
  // Yes if any participant identifies a specific gap AND connects it to what another
  // person in the room could provide.
"skill_gap_description": string or "None"

"shared_vision_indicator": "Yes" or "No"
  // Yes if the conversation shifted from participants describing their OWN separate work
  // toward discussing a SHARED project. Key signal: "our," "we," "together" when referring
  // to the proposed research (not just social politeness).
"shared_vision_quote": string or "None"

"pronoun_shift_flag": "Yes" or "No"
  // Yes if this chunk shows a NOTABLE SHIFT from individual ("my work," "your work")
  // toward joint framing ("our idea," "we could") within this chunk. Code Yes only when
  // the shift OCCURS here — not if joint language has been present since the beginning.

--- INTERPERSONAL AND RELATIONAL SIGNALS ---
"personal_disclosure": "Yes" or "No"
  // Yes if any participant shares something personally revealing beyond professional role
  // presentation: a research frustration, career aspiration, domain passion, past failure.
  // Do NOT flag polite self-introductions.

"laughter_quality": one of ["tension_release", "shared_humor", "appreciative",
                             "social_lubricant", "none"]
  // tension_release: laughter at/after disagreement or awkwardness.
  // shared_humor: laughter in response to a joke — social warmth.
  // appreciative: laughter in direct response to an idea being CLEVER or SURPRISING —
  //   most predictive of intellectual rapport and team formation.
  // social_lubricant: light background laughter with no clear trigger.
  // none: no laughter occurred.

--- PSYCHOLOGICAL SAFETY AND DISSENT ---
"dissent_response_quality": integer 1, 2, 3, or "NA"
  // 1 = Dismissive or defensive: dissent is interrupted, dropped, or unwelcome.
  // 2 = Neutral: dissent is acknowledged but not deeply engaged.
  // 3 = Curious and exploratory: dissent is met with follow-up questions or genuine engagement.
  // "NA" = No dissent or contrarian view was expressed in this chunk.

--- INTELLECTUAL RISK AND AMBITION ---
"risk_acknowledgment_with_enthusiasm": "Yes" or "No"
  // Yes if any participant explicitly acknowledged the project is risky/uncertain/hard
  // AND responded with excitement rather than hedging. BOTH elements must be present.
"risk_enthusiasm_quote": string or "None"

--- GRANT AND FUNDING CONTEXT ---
"funding_awareness_signal": "Yes" or "No"
  // Yes if any participant mentions a specific funding mechanism, program priority,
  // grant deadline, review criterion, or funding agency. Vague mentions do not qualify.
"funding_reference_description": string or "None"

"prior_relationship_signal": "Yes" or "No"
  // Yes if any participant mentions prior familiarity with another participant:
  // having read their work, met before, collaborated previously.
"prior_relationship_description": string or "None"

--- MEETING PROCESS QUALITY ---
"meeting_structure_quality": one of ["unstructured", "loosely_structured", "structured"]
  // unstructured: conversation is associative; topic jumps without apparent shared plan.
  // loosely_structured: general shared sense of what they're doing, but no explicit agenda.
  // structured: participants explicitly reference phases, topics to cover, or what remains.

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
  //   "code_name": one of the 16 categories below
  //   "subcode": string or "None"
  //   "evidence": verbatim quote that supports the code.
  //   "explanation": string — 1–2 sentences justifying this code.
  //   "idea_quality": integer 0, 1, or 2 — ONLY for: Idea Management, Integration Practices, Knowledge Sharing.
  //                   Omit this field entirely for all other categories.
  //
  //   idea_quality rubric:
  //   0 = Undeveloped, vague, tangential, or already well-known. Does not advance the team's work.
  //   1 = Clear, relevant, specific. Another participant could meaningfully respond.
  //   2 = Notably novel, richly developed, fills a gap, or opens a new direction.

BEHAVIORAL CODE CATEGORIES AND SUBCODES:

[1] Idea Management
  Subcodes: "proposes_new_idea" | "extends_existing_idea" | "combines_ideas" |
            "returns_to_earlier_idea" | "redirects_idea"

[2] Information Seeking
  Subcodes: "asks_factual_question" | "asks_clarifying_question" |
            "asks_for_elaboration" | "asks_for_opinion" | "asks_rhetorical_question"

[3] Knowledge Sharing
  Subcodes: "shares_domain_knowledge" | "shares_data_or_findings" |
            "shares_method_or_approach" | "shares_personal_experience"

[4] Evaluation Practices
  Subcodes: "supports_or_validates" | "critiques_or_challenges" |
            "compares_options" | "raises_concern" | "devil_advocate" |
            "setback_response_explores" | "setback_response_defends" |
            "setback_response_redirects" | "setback_response_accepts_builds"
  // setback_response subcodes apply when the speaker responds to criticism of their own idea
  // or to an obstacle raised about the team's current direction.

[5] Relational Climate
  Subcodes: "expresses_appreciation" | "encourages_participation" |
            "manages_tension" | "uses_humor" | "expresses_enthusiasm"

[6] Participation Dynamics
  Subcodes: "invites_contribution" | "yields_floor" | "redirects_speaker" |
            "summarizes_for_group" | "gatekeeps"

[7] Coordination and Decision Practices
  Subcodes: "proposes_process" | "calls_for_decision" | "records_or_documents" |
            "proposes_next_step" | "checks_consensus" | "scope_calibration"
  // scope_calibration: speaker explicitly discusses whether the project scope is appropriate
  // for a grant proposal — too big, too small, or about right.

[8] Integration Practices
  Subcodes: "synthesizes_contributions" | "identifies_common_ground" |
            "resolves_contradiction" | "frames_shared_problem"

[9] Idea Ownership and Attribution
  Subcodes: "claims_own_idea" | "attributes_to_other" | "challenges_attribution"

[10] Future-Oriented Language
  Subcodes: "vague_future_reference" | "specific_future_plan" | "named_next_step"
  // vague_future_reference: "We should think more about this."
  // specific_future_plan: "Let's schedule a follow-up call."
  // named_next_step: concrete action with responsible person and approximate timeline.

[11] Epistemic Bridging
  Subcodes: "translates_terminology" | "connects_methods" | "reframes_cross_disciplinarily"

[12] Pronoun Framing  *** APPLY TO ALL SUBSTANTIVE UTTERANCES ABOUT THE RESEARCH ***
  Subcodes: "individual_framing" | "joint_framing" | "ambiguous"
  // individual_framing: "my research," "your approach," "in my field we do X."
  // joint_framing: "we," "our," "together" when describing proposed work.
  // ambiguous: discusses research but cannot be clearly classified as either.

[13] Complementarity Articulation
  Subcodes: "expertise_complementarity" | "resource_complementarity" | "method_complementarity"

[14] Role Anticipation
  Subcodes: "explicit_role_assignment" | "implicit_role_suggestion"

[15] Broader Significance
  Subcodes: "field_significance" | "societal_significance" | "funding_priority_alignment"

[16] Idea Novelty Signal
  Subcodes: "novelty_recognized_self" | "novelty_recognized_other"
  // Explicitly marks an idea as surprising or unlike prior approaches — distinct from enthusiasm.

--- INTERRUPTION TYPE ---
"interruption_type": one of ["not_interruption", "collaborative_completion",
                              "elaborative_jump_in", "competitive_interruption"]
  // not_interruption: this turn followed a complete turn by the previous speaker.
  // collaborative_completion: speaker finishes prior speaker's sentence in agreement/support.
  // elaborative_jump_in: speaker adds new content before prior speaker finishes, in support.
  // competitive_interruption: speaker cuts off prior speaker to redirect or contradict.

--- MULTIMODAL SIGNALS (VIDEO + AUDIO — DO NOT ESTIMATE FROM TRANSCRIPT ALONE) ---
// If participants are off-camera or video quality is insufficient, apply [low_confidence].

"visible_off_screen_distraction": "Yes" or "No"
  // Yes ONLY if at least one non-speaking participant is clearly and SUSTAINEDLY showing
  // off-screen distraction. Do NOT flag brief eye movements or natural head shifts.
  // When in doubt, write "No".
"distracted_participant_count": integer — Set to 0 if visible_off_screen_distraction is "No".

"nod_count": integer — Total visible head nods from non-speakers. Count individual gestures.
"shared_affect": "Yes" or "No" — 2+ participants simultaneously display positive affect.
"any_smile_other": "Yes" or "No" — At least one non-speaking participant is smiling.
"audible_backchannel": "Yes" or "No" — Any non-speaker produces audible backchannel ("mm-hmm", "yeah", laughter).

"vocal_enthusiasm": integer 1, 2, 3, or 4
  // 1 = Flat, monotone, very low energy.
  // 2 = Moderate, conversational energy.
  // 3 = Noticeably energetic and engaged.
  // 4 = High energy, passionate, emphatic.
"hesitation_flag": "Yes" or "No" — Notable pause (>2s) or repeated false starts before key claim.
"pace": one of ["fast", "normal", "slow"]

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
    """

    code_book = """
        (1) present new idea: introduces a novel concept, approach, or hypothesis not previously mentioned. Example: "What if we used reinforcement learning instead?"
        (2) expand on existing idea: builds on a previously mentioned idea, adding details, variations, or improvements. Example: "Yes, and if we train it with synthetic data, we might improve accuracy."
        (3) provide supporting evidence: provides data, references, or logical reasoning to strengthen an idea. Example: "A recent Nature paper shows that this method outperforms others."
        (4) explain or define term or concept: explains a concept, method, or terminology for clarity. Example: "When I say 'feature selection,' I mean choosing the most important variables."
        (5) ask clarifying question: requests explanation or elaboration on a prior statement. Example:"Can you explain what you mean by 'latent variable modeling'?"
        (6) propose decision: suggests a concrete choice for the group. "I think we should prioritize dataset A."
        (7) confirm decision: explicitly agrees with a proposed decision, finalizing it. Example: "Yes, let's go with that approach."
        (9) express alternative decision: rejects a prior decision and suggests another. Example: "Instead of dataset A, we should use dataset B because it has more variability."
        (10) express agreement: explicitely agrees with proposed idea or decision. Example: "I agree with your approach."
        (11) assign task: assigns responsibility for a task to a team member. Example: "Alex, can you handle data processing?"
        (12) offer constructive criticism: critiques with the intent to improve. Example: "This model has too many parameters, maybe we can prune them."
        (13) reject idea: dismisses or rejects an idea but does not offer a new one or ways to improve. "I don't think that will work"
        (14) resolve conflict: mediates between opposing ideas to reach concensus. "we can test both and compare the results."
        (15) express frustration: verbalizes irritation, impatience, or dissatisfaction. "We have been going in circles on this issue."
        (16) acknowledge contribution: verbally recognizes another person's input, but not agreeing or expanding. "That is a great point."
        (17) encourage participation: invites someone else to contribute. "Alex, what do you think?"
        (18) express enthusiasm: expresses excitement, optimism, or encouragement. "This is an exciting direction!"
        (19) express humor: makes a joke or laughs. example: "well, at least this model isn't as bad as our last one!"
    """



    return client, prompt, code_book

# Save the path dictionary to a JSON file
def save_path_dict(path_dict, file_name, destdir):
    with open(f"{destdir}/{file_name}", 'w') as json_file:
        json.dump(path_dict, json_file, indent=4)

def sanitize_name(name, replace_char='_'):
    """
    Sanitize a name by replacing spaces and hyphens with underscores.
    
    Args:
        name: The name to sanitize
        replace_char: Character to use as replacement for spaces and hyphens
        
    Returns:
        A sanitized version of the name
    """
    # Normalize Unicode characters (e.g., convert 'é' to 'e')
    name = unicodedata.normalize('NFKD', name)
    
    sanitized = name.replace(' ', replace_char).replace('-', replace_char).replace('._', replace_char)
    sanitized = re.sub(f'{replace_char}+', replace_char, sanitized)
    sanitized = sanitized.strip(replace_char)
    return sanitized

# get all the video files (scialog directory is categorized by folders of each conference)
def get_video_in_folders(directory):
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv']
    video_files = []

    # Get all items in the given directory
    items = os.listdir(directory)
    
    # First check for folders
    folder_names = [f for f in items if os.path.isdir(os.path.join(directory, f))]
    
    # If folders exist, process videos in folders
    if folder_names:
        for folder in folder_names:
            folder_path = os.path.join(directory, folder)
            # Get all files in the current folder
            files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
            for file in files:
                file_name, file_extension = os.path.splitext(file)
                if file_extension.lower() in video_extensions:
                    if file_extension.lower() == '.mkv':
                        # convert to mp4 first if it is a mkv file
                        mp4_path = os.path.join(folder_path, file_name + '.mp4')
                        if not os.path.exists(mp4_path):
                            mp4_path = convert_mkv_to_mp4(os.path.join(folder_path, file))
                        video_files.append((mp4_path, folder_path, f"{folder}/{file_name}", file))
                    else:
                        video_files.append((os.path.join(folder_path, file), folder_path, f"{folder}/{file_name}", file))
    
    # If no folders or additional files exist in root directory, check for direct video files
    files_in_root = [f for f in items if os.path.isfile(os.path.join(directory, f))]
    for file in files_in_root:
        file_name, file_extension = os.path.splitext(file)
        if file_extension.lower() in video_extensions:
            if file_extension.lower() == '.mkv':
                # convert to mp4 first if it is a mkv file
                mp4_path = os.path.join(directory, file_name + '.mp4')
                if not os.path.exists(mp4_path):
                    mp4_path = convert_mkv_to_mp4(os.path.join(directory, file))
                video_files.append((mp4_path, directory, f"{file_name}", file))
            else:
                video_files.append((os.path.join(directory, file), directory, f"{file_name}", file))

    return video_files

# get videos files in a directory (used for split directory with splitted videos)
def get_videos(directory):
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv']
    video_files = []

    # Get all files in the given directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    for file in files:
        file_name, file_extension = os.path.splitext(file)
        if file_extension.lower() in video_extensions:
            if file_extension.lower() == '.mkv':
                # Check if MP4 version exists
                mp4_path = os.path.join(directory, file_name + '.mp4')
                if not os.path.exists(mp4_path):
                    # Convert MKV to MP4
                    converted_path = convert_mkv_to_mp4(os.path.join(directory, file))
                    if converted_path:
                        video_files.append(os.path.basename(converted_path))
                else:
                    # Use existing MP4 version
                    video_files.append(file_name + '.mp4')
            else:  # Skip MKV files but include other video formats
                video_files.append(file)
        
    return video_files

# convert mkv to mp4 using ffmpeg
def convert_mkv_to_mp4(input_path):
    """
    Converts an MKV video file to MP4 using ffmpeg.
    Re-encodes audio to AAC to ensure compatibility.
    """
    import subprocess
    import os

    if not input_path.lower().endswith(".mkv"):
        raise ValueError("Input file must be an MKV file.")
    
    output_path = os.path.splitext(input_path)[0] + ".mp4"

    command = [
        "ffmpeg",
        "-i", input_path,
        "-c:v", "copy",        # Copy video stream
        "-c:a", "aac",         # Re-encode audio to AAC
        "-b:a", "192k",        # Set audio bitrate (optional)
        output_path
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Conversion successful: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Conversion failed: {e}")
        return None

# Create or update the path dictionary with video file paths and their split chunks.
# All recordings in a session folder are merged into one ordered list per session,
# sorted by recording timestamp so context flows correctly across the full meeting.
def create_or_update_path_dict(directory, cur_dir):
    folder_name = os.path.basename(directory)
    path_dict_file = os.path.join(cur_dir, f"{folder_name}_path_dict.json")

    # Build a flat status map from any existing path_dict: chunk_path → (gemini_name, analyzed).
    # Keyed by full file path so it survives key-structure changes between runs.
    status_map = {}
    if os.path.exists(path_dict_file):
        with open(path_dict_file, 'r') as f:
            existing = json.load(f)
        for chunks in existing.values():
            for entry in chunks:
                if len(entry) >= 4:
                    status_map[entry[1]] = (entry[2], entry[3])

    # Collect all video files, skipping MKV files that already have an MP4 version.
    video_files = get_video_in_folders(directory)
    processed_files = set()

    # Group recordings by session folder (the immediate parent folder of each video).
    # session_key  → {'folder_path': str, 'recordings': [(video_path, full_filename)]}
    sessions = {}
    for video_path, folder_path, _vfn, full_filename in video_files:
        if video_path in processed_files:
            continue
        _, file_extension = os.path.splitext(full_filename)
        # Skip MKV if an MP4 version already exists (get_video_in_folders may return both)
        if file_extension.lower() == '.mkv':
            mp4_path = os.path.join(os.path.dirname(video_path), os.path.splitext(full_filename)[0] + '.mp4')
            if os.path.exists(mp4_path):
                continue
        session_key = os.path.basename(folder_path)
        if session_key not in sessions:
            sessions[session_key] = {'folder_path': folder_path, 'recordings': []}
        sessions[session_key]['recordings'].append((video_path, full_filename))
        processed_files.add(video_path)

    path_dict = {}
    for session_key, session_data in sessions.items():
        folder_path = session_data['folder_path']
        # Sort recordings by the timestamp embedded in the filename (HH_MM_SS / HH-MM-SS).
        recordings = sorted(session_data['recordings'],
                            key=lambda r: extract_recording_timestamp(r[1]))

        all_chunks = []
        for video_path, full_filename in recordings:
            file_name = os.path.splitext(full_filename)[0]
            split_dir = os.path.join(os.path.dirname(video_path), f"split-{file_name}")

            if os.path.exists(split_dir):
                # Long recording — use its ordered chunks.
                chunk_files = get_videos(split_dir)
                chunk_files = [f for f in chunk_files
                               if not (f.endswith('.mkv') and
                                       os.path.exists(os.path.join(split_dir, f.replace('.mkv', '.mp4'))))]
                chunk_files = list(dict.fromkeys(chunk_files))
                chunk_files.sort(key=lambda x: extract_chunk_number(x)
                                 if extract_chunk_number(x) is not None else float('inf'))
                for chunk_file in chunk_files:
                    chunk_path = os.path.join(split_dir, chunk_file)
                    gemini_name, analyzed = status_map.get(chunk_path, (' ', False))
                    all_chunks.append([chunk_file, chunk_path, gemini_name, analyzed])
            else:
                # Short recording — treat the file itself as a single chunk.
                gemini_name, analyzed = status_map.get(video_path, (' ', False))
                all_chunks.append([full_filename, video_path, gemini_name, analyzed])

        # Deduplicate while preserving order.
        seen, unique = set(), []
        for entry in all_chunks:
            if entry[1] not in seen:
                unique.append(entry)
                seen.add(entry[1])

        path_dict[session_key] = unique

    return path_dict

# Split a video into chunks of specified length
def split_video(video_full_path, duration, chunk_length=10*60):
    
    # Calculate the number of chunks
    num_chunks = int(duration // chunk_length) + 1
    
    # Get the file name and directory
    file_name, _ = os.path.splitext(os.path.basename(video_full_path))
    directory = os.path.dirname(video_full_path)
    
    # Create a directory to store the split videos
    split_dir = os.path.join(directory, f"split-{file_name}")
    os.makedirs(split_dir, exist_ok=True)
    
    # Split the video into chunks
    chunk_paths = []
    for i in range(num_chunks):
        start_time = i * chunk_length
        output_file_name = os.path.join(split_dir, f"{file_name}_chunk{i+1}.mp4")
        # Add -n flag to skip existing files
        ffmpeg.input(video_full_path, ss=start_time, t=chunk_length).output(output_file_name, n=None).run()
        chunk_paths.append(output_file_name)
        print(f"Created chunk: {output_file_name}")
    
    return chunk_paths

# Process all videos in a directory, splitting them if necessary
def process_videos_in_directory(directory):
    
    # path to video, path_to_folder, video file name for path_dict, original filename
    video_files = get_video_in_folders(directory)

    split_videos_dict = {}
    
    for video_file in video_files:
        video_full_path = video_file[0]
        file_name, file_extension = os.path.splitext(video_file[3])
        split_dir = os.path.join(os.path.dirname(video_full_path), f"split-{file_name}")
        if video_full_path:
            # Check if the file is MKV and needs conversion
            if file_extension.lower() == '.mkv':
                mp4_path = video_full_path.replace('.mkv', '.mp4')
                if not os.path.exists(mp4_path):
                    print(f"Converting MKV to MP4: {video_full_path}")
                    video_full_path = convert_mkv_to_mp4(video_full_path)
                else:
                    video_full_path = mp4_path
            
            if not os.path.exists(split_dir):
                try:
                    probe = ffmpeg.probe(video_full_path)
                    duration = float(probe['format']['duration'])
                    if duration > 10 * 60:
                        print(f"Splitting video: {video_file}")
                        split_videos_dict[video_file] = split_video(video_full_path, duration)
                    else:
                        print(f"Video {video_file} is shorter than 10 minutes, no need to split.")
                        split_videos_dict[video_file] = [video_full_path]
                except Exception as e:
                    print(f"Having issues with video: {video_full_path}")
                    print(f"Here is the exception: {e}")
            else:
                print(f"Found a folder with splitted videos for {video_file} already.")
    
    return split_videos_dict

# Get the list of files on gemini
def safe_list_files(client):
    try:
        # Get the list of files
        files = client.files.list()
        
        # Filter out any files with problematic names
        safe_files = []
        for file in files:
            try:
                # Use our sanitization function
                safe_name = sanitize_filename(file.name)
                safe_files.append(safe_name)
            except Exception as e:
                # Skip files with problematic names
                print(f"Skipping file {file} with problematic name: {e}")
                continue
                
        return safe_files
    except Exception as e:
        print(f"When getting all files on gemini, received unexpected error: {e}")
        return []  # Return empty list instead of None

# Sanitize a filename to ensure it contains only ASCII characters
def sanitize_filename(name):
    """
    Sanitize a filename to ensure it contains only ASCII characters.
    Replace or remove problematic characters.
    """
    if name is None:
        return None
        
    try:
        # Try to encode the name to ASCII, replacing non-ASCII chars
        safe_name = name.encode('ascii', 'replace').decode('ascii')
        return safe_name
    except Exception as e:
        print(f"Error sanitizing filename {name}: {e}")
        
    
# Return the video file from Gemini, uploading it if necessary
def get_gemini_video(client, file_name, file_path, gemini_name):
    """Return the video file from Gemini, uploading it if necessary"""
    video_file = None
    
    # Sanitize the gemini_name if it exists
    safe_gemini_name = sanitize_filename(gemini_name)
    
    # If gemini_name is not empty or just whitespace, try to get the file
    if safe_gemini_name and safe_gemini_name.strip():
        try:
            gemini_file = client.files.get(name=safe_gemini_name)
            if gemini_file:
                print(f"{file_name} already uploaded to Gemini, returning that...")
                return gemini_file, safe_gemini_name
        except Exception as e:
            print(f"Could not get file with name {safe_gemini_name}: {e}")
    
    # If we couldn't get the file, upload it
    print(f"Uploading {file_name} to Gemini")
    safe_file_path = sanitize_filename(file_path)
    try:
        video_file = client.files.upload(file=safe_file_path)
        print(f"Completed upload: {video_file.uri}")  
        while video_file.state.name == "PROCESSING":
            print('.', end='')
            time.sleep(10)
            video_file = client.files.get(name=video_file.name)
            safe_gemini_name = sanitize_filename(video_file.name)
        if video_file.state.name == "FAILED":
            print(f"File processing failed for {file_name}")
    except Exception as e:
        print(f"When processing {file_name} at {safe_file_path} encountered the following error: {e}")
   
        
    return video_file, safe_gemini_name


# Build a chunk prompt by prepending session context from prior chunks
def build_chunk_prompt(base_prompt, session_state=None, chunk_index=0):
    """
    Prepend a SESSION CONTEXT block to the base prompt when prior state exists.
    chunk_index is 0-based (first chunk = 0).
    """
    if session_state is None:
        context_block = (
            f"\n[SESSION CONTEXT: This is chunk {chunk_index + 1} of the session. "
            "No prior context — annotate without assumptions about what preceded this segment.]\n\n"
        )
    else:
        prior_traj = json.dumps(session_state.get('idea_trajectory_sequence', []))
        ideas = json.dumps(session_state.get('ideas_currently_on_table', []))
        signals = json.dumps(session_state.get('commitment_signals_cumulative', []))
        speakers = json.dumps(session_state.get('speakers_identified', []))
        pronoun_shifted = session_state.get('pronoun_shift_occurred_cumulative', False)
        shared_vision = session_state.get('shared_vision_established_cumulative', False)
        dcl = session_state.get('decision_crystallization_level', 'unknown')
        psl = session_state.get('problem_specificity_level', 'unknown')
        chunk_summaries = session_state.get('chunk_summaries', [])
        summaries_block = '\n'.join(
            f"  Chunk {i + 1}: {s}" for i, s in enumerate(chunk_summaries)
        ) or '  (none)'

        context_block = f"""
======================================================================
SESSION CONTEXT — FROM PRIOR CHUNKS (do not re-annotate; use for continuity only)
======================================================================
This is chunk {chunk_index + 1} of the session.

WHAT HAPPENED IN EACH PRIOR CHUNK:
{summaries_block}

pronoun_shift_already_occurred: {str(pronoun_shifted).lower()}
  // If true, do NOT flag pronoun_shift_flag=Yes in this chunk unless a NEW shift occurs here.

shared_vision_already_established: {str(shared_vision).lower()}
  // If true, the group has already moved to joint framing in a prior chunk.

decision_crystallization_at_end_of_last_chunk: {dcl}
  // Use this as the baseline when rating decision_crystallization_level for this chunk.

problem_specificity_at_end_of_last_chunk: {psl}
  // Use this as the baseline when rating problem_specificity_level for this chunk.

idea_trajectory_sequence_so_far: {prior_traj}
  // The sequence of trajectory labels from all prior chunks.

ideas_currently_on_table: {ideas}
  // Use these to correctly code extends_existing_idea and returns_to_earlier_idea in Part B.
  // An utterance that builds on one of these ideas is extends_existing_idea, not proposes_new_idea.

commitment_signals_so_far: {signals}
  // Commitment signals already recorded — do not re-flag these.

speakers_identified_so_far: {speakers}
  // Use consistent names throughout — match against this list when identifying speakers.
======================================================================

"""
    return context_block + base_prompt


# Extract session state from a parsed chunk response for handoff to the next chunk
def extract_session_state(response_json, chunk_index, prior_state=None):
    """
    Pull session_state from the model's response JSON.
    Falls back to reconstructing minimal state from chunk_summary if session_state is absent.
    """
    chunk_summary = response_json.get('chunk_summary', {})
    model_state = response_json.get('session_state', {})

    # The model should have produced session_state; use it directly if present.
    if model_state:
        state = dict(model_state)
    else:
        # Fallback: reconstruct from chunk_summary fields
        prior_traj = (prior_state or {}).get('idea_trajectory_sequence', [])
        this_traj = chunk_summary.get('idea_trajectory')
        state = {
            'pronoun_shift_occurred_cumulative': (
                (prior_state or {}).get('pronoun_shift_occurred_cumulative', False)
                or chunk_summary.get('pronoun_shift_flag') == 'Yes'
            ),
            'shared_vision_established_cumulative': (
                (prior_state or {}).get('shared_vision_established_cumulative', False)
                or chunk_summary.get('shared_vision_indicator') == 'Yes'
            ),
            'idea_trajectory_sequence': prior_traj + ([this_traj] if this_traj else []),
            'ideas_currently_on_table': [],
            'commitment_signals_cumulative': list((prior_state or {}).get('commitment_signals_cumulative', [])),
            'speakers_identified': list((prior_state or {}).get('speakers_identified', [])),
        }
        if chunk_summary.get('explicit_commitment_signal') == 'Yes':
            state['commitment_signals_cumulative'].append({
                'speaker': chunk_summary.get('commitment_signal_speaker', ''),
                'quote': chunk_summary.get('commitment_signal_quote', ''),
            })

    # Always carry forward the last crystallization/specificity levels for the preamble
    state['decision_crystallization_level'] = chunk_summary.get('decision_crystallization_level')
    state['problem_specificity_level'] = chunk_summary.get('problem_specificity_level')
    state['chunk_index'] = chunk_index

    # Ensure chunk_summaries is a proper accumulated list regardless of path taken above.
    # The model should have appended its own entry; if it produced a flat string or is missing,
    # reconstruct by carrying forward prior entries and appending whatever the model gave.
    prior_summaries = list((prior_state or {}).get('chunk_summaries', []))
    model_summaries = state.get('chunk_summaries')
    if isinstance(model_summaries, list) and len(model_summaries) == len(prior_summaries) + 1:
        # Model correctly appended one new entry — use as-is
        pass
    else:
        # Model returned a flat string, wrong length, or missing field — salvage
        if isinstance(model_summaries, list) and len(model_summaries) > len(prior_summaries) + 1:
            # Model returned too many entries; keep only the last N entries matching our count
            state['chunk_summaries'] = model_summaries[-(len(prior_summaries) + 1):]
        elif isinstance(model_summaries, str) and model_summaries:
            state['chunk_summaries'] = prior_summaries + [model_summaries]
        else:
            state['chunk_summaries'] = prior_summaries + ['(no summary produced for this chunk)']

    return state


# Analyze a video using the Gemini API.
# Tries gemini-3.1-pro-preview first; on 503 UNAVAILABLE falls back to gemini-3.0-pro-preview.
# The video_file is already uploaded so no re-upload occurs on model switch.
def gemini_analyze_video(client, prompt, video_file, filename, max_tries=3, delay=1):
    models = ['gemini-3.1-pro-preview', 'gemini-2.5-pro']
    for model in models:
        print(f"Making LLM inference request for {filename} using {model}...")
        for attempt in range(max_tries):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=[prompt, video_file],
                    config={'temperature': 0},
                )
                print(f"Got a response from {model}!")
                return response
            except Exception as e:
                error_str = str(e)
                if '503' in error_str or 'UNAVAILABLE' in error_str.upper():
                    print(f"503 UNAVAILABLE from {model} for {filename}, switching to fallback model...")
                    break  # skip remaining retries for this model; try next model
                print(f"Error with {model} for {filename}: {e}")
                if attempt < max_tries - 1:
                    time.sleep(delay)
                else:
                    print(f"Exhausted {max_tries} attempts with {model} for {filename}. Error: {e}")
    print(f"All models failed for {filename}.")
    return None

# Analyze all videos in the path dictionary
def analyze_video(client, path_dict, prompt, dir):
    cur_dir = os.getcwd()
    n_path_dict = path_dict.copy()
    folder_name = os.path.basename(dir)

    # Create the base outputs directory
    base_output_dir = os.path.join(cur_dir, "outputs", f"{folder_name}")
    os.makedirs(base_output_dir, exist_ok=True)

    for session_key in n_path_dict.keys():
        # Sort in place on the dict's own list so mutations (gemini_name, analyzed flag)
        # still propagate. Chunks without a chunk number sort last.
        n_path_dict[session_key].sort(
            key=lambda x: extract_chunk_number(x[0]) if extract_chunk_number(x[0]) is not None else float('inf')
        )
        list_chunks = n_path_dict[session_key]
        safe_session_key = sanitize_name(session_key)
        output_dir = os.path.join(base_output_dir, f"output_{safe_session_key}")
        os.makedirs(output_dir, exist_ok=True)

        # Session state resets for each new session
        session_state = None

        for m in range(len(list_chunks)):
            # [chunk file name, full path to this video, gemini upload file name, analysis status]
            chunk_file_name = list_chunks[m][0]
            fileName, _ = os.path.splitext(chunk_file_name)
            file_path = list_chunks[m][1]
            gemini_name = list_chunks[m][2]
            analyzed = list_chunks[m][3]

            if not analyzed:
                print(f"Analyzing {chunk_file_name}")
                chunk_prompt = build_chunk_prompt(prompt, session_state, chunk_index=m)
                video_file, gemini_name = get_gemini_video(client, chunk_file_name, file_path, gemini_name)
                list_chunks[m][2] = gemini_name
                if video_file:
                    response = gemini_analyze_video(client, chunk_prompt, video_file, chunk_file_name)
                    if response and response.text:
                        print(f"Trying to save output for {chunk_file_name} to json file")
                        try:
                            save_to_json(response.text, fileName, output_dir)
                            list_chunks[m][3] = True
                            # Advance session state from the saved file — avoids re-parsing
                            # raw response.text which may contain markdown fences or preamble.
                            try:
                                saved_path = os.path.join(
                                    sanitize_name(output_dir),
                                    f"{sanitize_name(fileName)}.json"
                                )
                                with open(saved_path) as f:
                                    saved_json = json.load(f)
                                session_state = extract_session_state(saved_json, m, session_state)
                            except Exception as e:
                                print(f"Could not extract session state from {chunk_file_name}: {e}")
                        except ValueError:
                            list_chunks[m][3] = False
                    else:
                        print(f"No response for {chunk_file_name}. Response: {response}")
                        list_chunks[m][3] = False
                save_path_dict(n_path_dict, f"{folder_name}_path_dict.json", cur_dir)
            else:
                print(f"{chunk_file_name} already analyzed, advancing session state...")
                # Load the saved JSON to extract session state so context continues correctly
                saved_path = os.path.join(output_dir, f"{sanitize_name(fileName)}.json")
                if os.path.exists(saved_path):
                    try:
                        with open(saved_path) as f:
                            saved_json = json.load(f)
                        session_state = extract_session_state(saved_json, m, session_state)
                    except Exception as e:
                        print(f"Could not load session state from {saved_path}: {e}")

    print("Analysis all finished. Returning updated path_dict.")

    return n_path_dict

def parse_json_garbage(s):
    s = s[next(idx for idx, c in enumerate(s) if c in "{["):]
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        return json.loads(s[:e.pos])
    
# Save data to a JSON file in the specified directory
def save_to_json(text, file_name, output_dir):
    """
    Save the analysis text to a JSON file, handling various response formats
    
    Args:
        text (str): The text to save (can be JSON or plain text)
        file_name (str): The name of the file being analyzed
        output_dir (str): The directory to save the output to
    """
    # Ensure the output directory exists
    output_dir = sanitize_name(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    file_name = sanitize_name(file_name)
    # Construct the output file path using os.path.join
    output_file = os.path.join(output_dir, f"{file_name}.json")

    # Try to parse as JSON first
    if text:
        try:
            parsed_json = json.loads(text)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(parsed_json, f, ensure_ascii=False, indent=4)
            print(f"Successfully saved JSON to {output_file}")
            return
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
            print("Attempting to fix the JSON format...")
            # Attempt to fix the JSON format by removing any trailing characters
            
            try: 
                parsed_json = parse_json_garbage(text)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(parsed_json, f, ensure_ascii=False, indent=4)
                print(f"Successfully saved fixed JSON to {output_file}")
            except json.JSONDecodeError as e:
                output_file = os.path.join(output_dir, f"ATTN_{file_name}.json")
                print(f"Failed to fix JSON format: {e}. Saving the original response text to the file for manual inspection.")
                with open(output_file, 'w') as json_file:
                    json_file.write(text)
    else:
        raise ValueError(f"No text to save for {file_name}")
        # print(f"No text to save for {file_name}")

# save path dict file
def save_path_dict(path_dict, file_name, destdir):
    with open(f"{destdir}/{file_name}", 'w') as json_file:
        json.dump(path_dict, json_file, indent=4)

# Annotate utterances with Gemini
def annotate_utterances(client, merged_list, codebook):
    """
    Iterates over each utterance, using the full list as context.
    Returns structured annotations in JSON format.
    """

    annotations = []  # Store annotations for all utterances

    for i, utterance in enumerate(merged_list):
        # Prepare the prompt
        comm_prompt = f"""
        You are an expert in interaction analysis, team science, and qualitative coding. Your task is to analyze an 
        utterance from a scientific collaboration meeting and annotate it using the codes in the provided codebook.

        **Annotation Guidelines:**
        - If the utterance has multiple sentences, assign one most relevant code for each sentence. If no code applies, write 'None'.
        - For each code you choose, provide:
          1. **Code Name** 
          2. **Explanation**: Justify why this code applies in 1 sentence, using relevant prior context.
        - Ensure that **each annotation follows the structured JSON format**.

        ** Codebook for Annotation:**
        {codebook}

        ** Full Conversation Context:**
        {json.dumps(merged_list[:i+1], indent=2)}  # Include past utterances up to the current one

        ** Utterance to Annotate:**
        "{utterance}"

        **Expected Output:**
        Output a structured JSON file where each coded category is a key and the explanation is the value. If there are multiple explanations for this code, summarize them into one coherent sentence.
        Do not include codes that are not relevant to this utterance in your output.
        """

        # Call Gemini API (adjust depending on your implementation)
        response = client.models.generate_content(
            model='gemini-3.1-pro-preview',
            contents=[comm_prompt],
            config={
                'response_mime_type':'application/json',
                'temperature':0,
                'max_output_tokens':8192,      
            },)
        annotation = json.loads(response.text)  # Parse response as JSON

        annotations.append({
            "utterance": utterance,
            "annotations": annotation  # Store structured annotation
        })

    return annotations  # Return all annotations

# Extract the chunk number from a string
def extract_chunk_number(string):
    match = re.search(r'chunk(\d+)', string)
    if match:
        return int(match.group(1))
    else:
        return None

# Extract a sortable datetime string from a recording filename.
# Handles both "YYYY-MM-DD HH-MM-SS" and "YYYY_MM_DD_HH_MM_SS" conventions.
def extract_recording_timestamp(filename):
    stem = os.path.splitext(filename)[0]
    match = re.search(r'(\d{4})[-_](\d{2})[-_](\d{2})[-_ ](\d{2})[-_](\d{2})[-_](\d{2})', stem)
    if match:
        return ''.join(match.groups())  # YYYYMMDDHHMMSS — naturally sortable
    return ''.join(re.findall(r'\d+', stem))  # fallback: all digits in order

# Add two time strings in HH:MM format
def add_times(time1, time2):
    # Split the time strings into hours and minutes
    hours1, minutes1 = map(int, time1.split(':'))
    hours2, minutes2 = map(int, time2.split(':'))
    
    # Add the hours and minutes separately
    total_hours = hours1 + hours2
    total_minutes = minutes1 + minutes2
    
    # If total minutes are 60 or more, convert to hours
    if total_minutes >= 60:
        total_hours += total_minutes // 60
        total_minutes = total_minutes % 60
    
    # Format the result as "HH:MM"
    return f"{total_hours:02}:{total_minutes:02}"

class InvalidJsonContentError(Exception):
    """Exception raised when JSON file is invalid or empty."""
    pass

# Load JSON files from a directory and sort them by chunk number
def load_json_files(directory):
    # Get all JSON files except those starting with 'all' or 'verbal'
    json_files = [f for f in os.listdir(directory) if f.endswith('.json') and not (f.startswith('all') or f.startswith('verbal'))]
    
    if not json_files:
        raise InvalidJsonContentError(f"No JSON files found in directory: {directory}")
    
    # Sort files by chunk number, handling files without chunk numbers
    def sort_key(filename):
        chunk_num = extract_chunk_number(filename)
        return chunk_num if chunk_num is not None else float('inf')
    
    json_files.sort(key=sort_key)
    
    result = []
    invalid_files = []
    
    for json_file in json_files:
        file_path = os.path.join(directory, json_file)
        try:
            with open(file_path) as f:
                data = json.load(f)
                # Check if JSON is empty or has no content
                if data == {} or data == [] or not data:
                    invalid_files.append(f"{json_file} (empty content)")
                    continue
                
                chunk_number = extract_chunk_number(json_file)
                
                # Only process files with valid chunk numbers
                if chunk_number is not None:
                    # Ensure the result list is long enough
                    while len(result) < chunk_number:
                        result.append(None)
                    result[chunk_number-1] = data
                
        except json.JSONDecodeError:
            invalid_files.append(f"{json_file} (invalid JSON)")
        except Exception as e:
            invalid_files.append(f"{json_file} (error: {str(e)})")
    
    if invalid_files:
        raise InvalidJsonContentError(
            f"Found invalid or empty JSON files in {directory}:\n" + 
            "\n".join(f"- {f}" for f in invalid_files)
        )
    
    if not result:
        raise InvalidJsonContentError(f"No valid JSON files with chunk numbers found in {directory}")
    
    return result

# Merge output JSON files into a single list of annotations
def merge_output_json(directory):
    data_list = load_json_files(directory)
    num_chunks = len(data_list)
    full_output = []
    utterance_list=[]

    for m in range(num_chunks):
        ck = data_list[m]
        if ck:
            for key in ck.keys():
                d_list = ck[key]
                
                for i in d_list:
                    data = i.copy()
                    sp_time = data['timestamp'].split('-')
                    if len(sp_time)==2:
                        data['start_time'] = add_times(sp_time[0], str(10*m)+':00')
                        data['end_time'] = add_times(sp_time[1], str(10*m)+':00')
                    else:
                        data['start_time'] = add_times(sp_time[0], str(10*m)+':00')
                        
                    full_output.append(data)
                    utterance_list.append(f"{data['speaker']}: {data['transcript']} ")
        else:
            print(f"Having issues with data in directory: {directory}")
                
    return (full_output, utterance_list)

# Annotate and merge the output JSON files
def annotate_and_merge(client, path_dict, directory, codebook):
    for key in path_dict.keys():
        split_vid_name = key.split("/")
        if len(split_vid_name)>1:
            vid_name = split_vid_name[1]
        else:
            vid_name = split_vid_name[0]

        output_subfolder = os.path.join(directory, f"output-{key}")
        verbal_file = os.path.join(output_subfolder, f"verbal_{vid_name}.json")
        all_file = os.path.join(output_subfolder,f"all_{vid_name}.json" )
        
        if os.path.exists(output_subfolder):
            try:
                merged_output = merge_output_json(output_subfolder)
            except InvalidJsonContentError as e:
                print(f"Skipping {output_subfolder} due to invalid JSON content: {str(e)}")
                continue
                
            if not os.path.exists(all_file):
                verbal_annotations = []
                if not os.path.exists(verbal_file):
                    print(f"Annotating verbal behvaiors for {key}")
                    annotations = annotate_utterances(client, merged_output[1],codebook)
                    verbal_annotations = annotations
                    output_file = verbal_file
                    with open(output_file, "w") as f:
                        json.dump(annotations, f, indent=4)
                else:
                    print(f"{verbal_file} already exists, loading that to merge...")
                    with open(verbal_file, 'r') as f:
                        verbal_annotations = json.load(f)
                    
                if len(verbal_annotations)== len(merged_output[0]):
                    all_anno = merged_output[0].copy()
                    for i in range(len(verbal_annotations)):
                        anno = verbal_annotations[i]['annotations']
                        all_anno[i]['annotations'] = anno
                    print(f"Merged verbal annotations with existing video annotations.")
                    with open(all_file, "w") as f:
                        json.dump(all_anno, f, indent=4)
            else:
                print(f"{all_file} already exists, skipping...")
        else:
            print(f"{output_subfolder} does not exist, skipping...")


def main(vid_dir, annotate_video, prompt_type='scialog'):
    folder_name = os.path.basename(vid_dir)
    client, prompt, codebook = init(prompt_type)
    cur_dir = os.getcwd()
    process_videos_in_directory(vid_dir)
    path_dict = create_or_update_path_dict(vid_dir, cur_dir)
    save_path_dict(path_dict, f"{folder_name}_path_dict.json", cur_dir)
    if annotate_video == 'yes':
        path_dict = analyze_video(client, path_dict, prompt, vid_dir)
        save_path_dict(path_dict, f"{folder_name}_path_dict.json", cur_dir)

    return path_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze videos using Gemini AI')
    parser.add_argument('--dir', required=True,
                       help='Full path to the directory where videos are stored')
    parser.add_argument('--annotate-video', choices=['yes', 'no'], required=True,
                       help='Whether to annotate videos (yes/no)')
    parser.add_argument('--prompt-type', choices=['scialog', 'covid'], default='scialog',
                       help='Prompt type to use (default: scialog)')

    args = parser.parse_args()

    main(args.dir, args.annotate_video, args.prompt_type)