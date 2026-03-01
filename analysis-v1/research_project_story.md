## What this project is trying to answer
This project asks three core questions:
1. Can we use multimodal AI to annotate behaviors and interactions from recorded meetings at scale accurately and reliably?
2. Which AI annotated behavioral patterns are linked to success outcomes (team formation and funding)?
3. How predictive are these patterns when we test statistical and machine learning models?


## Targeted outlet and audience:
1. Nature machine intelligence
2. PNAS


## How video analysis and behavior annotation are done (from scripts)
This project uses a two-stage annotation pipeline before notebook analysis:
1. `analyze_video.py` (video-level parsing): each full meeting video is sent to Gemini with a structured prompt to produce `meeting_annotations` containing speaker, timestamp, transcript, speaking duration, interruption, screenshare, screenshare content, and other observations.
2. `annotate_team_behavior.py` (utterance-level team coding): each utterance is then coded with a deductive team-behavior codebook (`code_book_v4`) and scored for quality (`-1` to `2`) with JSON outputs per utterance.

### Prompt used to annotate videos (`analyze_video.py`, `scialog_prompt`)
```text
Objective:
You are an expert in interaction analysis and team research. You are provided with recording of a zoom meeting between a group of scientists collaborating on novel ideas to address scientific challenges.
Your objective is to annotate behavior and verbal cues to help us understand this team's behavior and processes.
Each time someone speaks in this video, provide the following structured annotation:
speaker: Full names of speaker.
timestamp: startiing and ending time of this person's speech in MM:SS-MM:SS format, before the speaker changes
transcript: Verbatim speech transcript. Remove filler words unless meaningful and one return one long string without line breaks or paragraph breaks.
speaking duration: the total number of seconds the speaker talks in this segment
interuption: Was this an interruption? (Yes/No) – if the speaker started talking before the previous speaker finished.
screenshare: Did anyone share their screen during this segment? (Yes/No)
screenshare_content: If there was screenshare, summarize the content shared on the screen and changes made to the content within the segment in no more than 3 sentences. Otherwise, write "None".
other: If there are any other unusual or surprising observations about the interaction in this segment, please include them here using a short sentence. Otherwise, write "None".

Notes:
For each label, if you feel uncertain due to poor visibility, add [low confidence] next to the annotation.
Ensure timestamps, speaker IDs, and behavior annotations are consistent throughout the video.

Input:
A video recording of a zoom meeting among a team of scientists from diverse backgrounds engaged in a collaborative task.

Output Format:
Return a JSON object with the key 'meeting_annotations' and list of annotations as value.
```

### Prompt used for utterance-level team behavior coding (`annotate_team_behavior.py`, deductive `comm_prompt`)
```text
You are an expert in deductive qualitative coding, team science, and qualitative coding. Your task is to analyze an utterance from a scientific collaboration meeting and annotate it using the codes in the provided codebook, then score each code based on quality criteria.
Apply code based on what is explicitly observed from the utterance, not inferred intent or motivation. Do not force a code that is not explicitly observed in the utterance.
Use the full conversation context to understand what has been previously discussed when deciding the most accurate code that applies to the utterance.

Annotation Guidelines:
- If no code applies, write 'None' as the code name.
- Only choose multiple codes (no more than 3) if they are all explicitly observed in the utterance.
- If the utterance only has a few words such as "yep", "umm", "I see", always choose the code "None".
- For each code you choose, provide:
  code_name, explanation, score (-1/0/1/2), score_justification.

Codebook for Annotation:
- Injected dynamically from `code_book_v4` (8 high-level team process categories).

Scoring Criteria (-1 to 2):
-1 = Blocks/dismisses ideas or hurts the process
0 = Vague/minimal contribution
1 = Clear contribution with some detail
2 = Novel/elaborated/reasoned contribution that moves the team forward

The prompt then provides category-specific anchors/examples for:
Idea Management, Information Seeking, Knowledge Sharing, Evaluation Practices,
Relational Climate, Participation Dynamics, Coordination and Decision Practices,
Integration Practices.

Prior Conversation Context:
- Previous and following 3 utterances are injected.

Utterance to Annotate:
- Current utterance is injected.

Expected Output:
- Output ONLY one JSON object.
- Keys are coded categories.
- Values contain explanation, score, and score_justification.
- If no code applies, output:
  {"None": {"explanation": "No relevant code applies to this utterance", "score": 0, "score_justification": "No code to score"}}.
```

### Additional transcript-only pipeline (`analyze_text.py`)
`analyze_text.py` supports transcript-first analysis (without video files):
1. Loads `.vtt` transcript files recursively and parses each line into structured entries: `speaker`, `timestamp`, `transcript`.
2. Formats parsed transcript into `Speaker (MM:SS-MM:SS): text` strings.
3. Sends transcript content + prompt to Gemini (`gemini-2.0-flash`, `temperature=0`).
4. Saves JSON outputs to `outputs/TEXT_<input_folder>/output_<file_name>/...`.
5. Uses the same `meeting_annotations` output pattern for downstream merging.

### Prompt used for transcript analysis (`analyze_text.py`, `scialog_prompt`)
```text
Objective:
You are an expert in interaction analysis and team research. You are provided with a transcript of a meeting between a group of scientists collaborating on novel ideas to address scientific challenges.
Your objective is to annotate behavior and verbal cues to help us understand this team's behavior and processes.
Each time someone speaks in this transcript, provide the following structured annotation:
speaker: Full names of speaker.
timestamp: startiing and ending time of this person's speech in MM:SS-MM:SS format, before the speaker changes
transcript: Verbatim speech transcript. Remove filler words unless meaningful.
speaking duration: the total number of seconds the speaker talks in this segment
nods_others: Count of head nods from other participants during this speaker's turn.
smile_self: Percentage of time this speaker was smiling during their turn.
smile_other: Percentage of time at least one other person was smiling.
distracted_others: Number of people looking away or using their phone during this speaker's turn.
hand_gesture: what type of hand gesture did the speaker use? (Raising Hand, Pointing, Open Palms, Thumbs up, Crossed Arms, None)
interuption: Was this an interruption? (Yes/No) – if the speaker started talking before the previous one finished.
overlap: Was this turn overlapped? (Yes/No) – if another person spoke at the same time
screenshare: Did anyone share their screen during this segment? (Yes/No)
screenshare_content: If there was screenshare, summarize the content shared on the screen and changes made to the content within the segment in no more than 3 sentences. Otherwise, write "None".

Notes:
If uncertain about a label due to poor visibility, return [low confidence] next to the annotation.
Ensure timestamps, speaker IDs, and behavior annotations are consistent throughout the transcript.
For transcripts, remove filler words unless meaningful and return one long string without line breaks or paragraph breaks.

Input:
A transcript of a meeting among a team of scientists from diverse backgrounds engaged in a collaborative task.

Output Format:
Return a JSON object with the key 'meeting_annotations' and list of annotations as value.
```



The analysis is built across four notebooks:
- `1-merge_data.ipynb`
- `2-vis_data.ipynb`
- `2-vis-analyze_data.ipynb`
- `3-reg_analysis.ipynb`

## Step 1: Build the analysis dataset (`1-merge_data.ipynb`)
The first notebook does the data engineering work.

Main work completed:
1. Loaded participant lists and outcome sheets (teams and funded teams).
2. Matched names between attendee records, team outcomes, and annotation outputs (including fuzzy fixes for spelling/name variants).
3. Merged session-level annotation outputs into unified session JSON files.
4. Joined behavioral features with outcome variables.
5. Created both count outcomes (`num_teams`, `num_funded_teams`) and binary outcomes (`has_teams`, `has_funded_teams`).
6. Created temporal versions of the dataset (`beginning`, `middle`, `end` segments).

Resulting data products:
- Full merged dataset reached **157 sessions** across conferences.
- Main feature table includes about **55 columns** (and **59** with temporal metadata).
- A model-ready subset used later has **156 sessions** (after removing one outlier session with extreme team count).

## Step 2: Describe patterns visually (`2-vis_data.ipynb`, `2-vis-analyze_data.ipynb`)
These two notebooks are largely parallel and focus on descriptive analysis and visual checks.

What was analyzed:
1. Conference-level distributions of team outcomes, funded outcomes, meeting length, screenshare, and team size.
2. Feature-level summaries grouped into categories (annotation counts, people-count features, score/ratio features, and other process indicators).
3. Scatter/correlation views for relationships among outcomes and key behavioral variables.
4. Deep dives into individual sessions and side-by-side session comparisons.

Key descriptive results:
- Conference outcome levels vary a lot.
- Mean `num_teams` ranges roughly from **1.08 to 1.94** by conference.
- Mean `num_funded_teams` ranges roughly from **0.21 to 0.93**.
- Meeting duration and screenshare behavior differ strongly across conferences.
- High-volume behavior categories include knowledge sharing and relational climate features.

Interpretation from this stage:
- The data contains real variation across contexts, which supports modeling.
- But it also suggests conference effects and heterogeneity, so simple one-pattern explanations may be too weak.

## Step 3: Regression and inference (`3-reg_analysis.ipynb` + model sections in `1-merge_data.ipynb`)
This stage tests whether behavior features explain or predict outcomes.

### 3A) Initial checks
- Distribution and relationship plots were generated for many features.
- Multicollinearity was assessed with VIF.
- In one early setup, many variables showed high VIF; later standardized setup kept all 25 modeling features under the chosen threshold.

### 3B) Linear and count-style modeling
- Linear regression on count outcomes showed weak explanatory power.
- Typical test-set performance in these runs:
  - `num_teams`: R² around **0.09 to 0.22** depending on model setup.
  - `num_funded_teams`: R² around **0.06 to 0.17**.
- This indicates limited ability to explain exact counts from current features.

### 3C) Logistic regression (binary outcomes)
Single-feature logistic models with controls (3-reg notebook):
- **44 models** total (22 per outcome).
- Only **2** models reached p < 0.05.
- Mean test AUC:
  - `has_teams`: **0.476**
  - `has_funded_teams`: **0.590**
- Overall mean test AUC: **0.528** (weak signal).
- Top individual AUCs were around **0.62**, not strong.

Multivariate residualized logistic models in `1-merge_data.ipynb`:
- `has_teams`: AUC about **0.568**.
- `has_funded_teams`: AUC about **0.710** (best among this modeling block).
- Best regularization search reported **C = 1000** with CV score about **0.605**.

## Step 4: Extended ML comparisons (`3-reg_analysis.ipynb`, later sections)
The notebook then compares Lasso logistic, Random Forest, and linear-probability style models across 5 random train/test splits.

### For `has_funded_teams`
- Lasso: mean AUC about **0.540**, best split AUC about **0.706**.
- Random Forest: mean AUC about **0.512**, best split AUC about **0.611**.
- Linear model: mean AUC about **0.593**, best split AUC about **0.726**.

### For `has_teams`
- Lasso: mean AUC about **0.574**, best split AUC about **0.760**.
- Random Forest: mean AUC about **0.594**, best split AUC in outputs around **0.726–0.771** depending on run block.

Recurring feature themes across better splits:
- Positive association in several runs: `positive_intensity`, `meeting_length`, coordination-related features.
- Negative association in several runs: `num_knowledge_sharing` (in some model parameterizations), `facilitator_dominance_ratio`, and some screenshare-heavy patterns.

## Step 5: Additional ML pipeline from `ml_analysis.py` + `ml_viz`
This script is a separate, more standardized model-comparison pipeline.

What `ml_analysis.py` does:
1. Runs three models for a binary outcome: Lasso Logistic Regression, Random Forest, and Linear Regression (LPM).
2. Uses **20 random 80/20 stratified train-test splits**.
3. Evaluates with Accuracy, F1, AUC-ROC (and R² for linear model).
4. Saves one plot per model with:
   - best-split performance bars
   - ROC curve
   - confusion matrix
   - top 10 features
5. Writes plots to `ml_viz/<outcome>/...`.

### Results shown in `ml_viz` (best split in each saved plot)
#### Main set (`ml_viz/has_funded_teams` and `ml_viz/has_teams`)
- **`has_funded_teams`**
  - Lasso: Best split 3, AUC around **0.62**
  - Random Forest: Best split 1, AUC around **0.72**
  - Linear (LPM): Best split 8, AUC around **0.74**
- **`has_teams`**
  - Lasso: Best split 11, AUC around **0.71**
  - Random Forest: Best split 11, AUC around **0.85**
  - Linear (LPM): Best split 12, AUC around **0.75**

#### Beginning-segment set (`ml_viz/*/all_data_df_beginning`)
- **`has_funded_teams`** (all three best at split 7)
  - Lasso: AUC **0.718**
  - Random Forest: AUC **0.837**
  - Linear (LPM): AUC **0.778**
- **`has_teams`**
  - Lasso: Best split 18, AUC **0.789**
  - Random Forest: Best split 7, AUC **0.771**
  - Linear (LPM): Best split 18, AUC **0.857**

### Feature patterns highlighted in `ml_viz`
Top-feature panels repeatedly surface:
- coordination/decision process variables
- knowledge-sharing and quality-ratio variables
- participation and relational-climate variables
- facilitator behavior variables (dominance/high-quality ratio/average score)
- meeting length and screenshare-related process variables

### Interpretation of this ML block
1. In this pipeline, **Random Forest and Linear models often outperform Lasso on AUC**.
2. The **beginning-segment dataset** shows especially strong discrimination for funded-team prediction (RF AUC ~0.84).
3. For `has_teams`, high AUC in some best splits (up to ~0.86) suggests separable signal, but class imbalance means these should be interpreted with caution.
4. Combined with notebook results, this supports the same overall conclusion: there is meaningful predictive signal, but strength is model- and setup-dependent.

## Step 6: Temporal segment regression (`regression/temporal_analysis.py`)
This script tests whether early-meeting behavior predicts later-meeting behavior at the feature level.

What the script does:
1. Loads per-session JSON features from `data/*/featurized_with_when/` for `beginning`, `middle`, and `end`.
2. Keeps only sessions with all three segments.
3. Creates segment-prefixed features (`beginning_*`, `middle_*`, `end_*`).
4. Runs per-feature linear regressions for temporal links (including beginning→end).
5. Uses standardized predictors and controls by default.

For the **beginning → end** results file (`regression/temporal_beginning_to_end_results.xlsx`):
- Sheets included: `All_Results`, `Significant_Results`, `High_R_Squared`.
- Total analyzed features: **33** (`All_Results`).
- Significant at p < 0.05: **8** features.
- Significant at p < 0.01: **5** features.
- Sample size per model: **N = 16**.
- Controls used in each model: `beginning_num_facilitator`, `beginning_total_utterances`, `num_members`, `meeting_length` (4 controls total).
- Mean R² across features: **0.512** (median **0.532**, max **0.884**).

Strongest beginning→end effects in this file:
- `facilitator_dominance_ratio`: R² **0.884**, p **0.007**
- `mean_score_information_seeking`: R² **0.835**, p **0.00001**
- `num_knowledge_sharing`: R² **0.829**, p **0.024**
- `total_score`: R² **0.775**, p **0.022**
- `num_evaluation_practices`: R² **0.739**, p **0.002**
- `num_relational_climate`: R² **0.690**, p **0.009**
- `percent_time_screenshare`: R² **0.562**, p **0.010**
- `mean_score_coordination_decision`: R² **0.481**, p **0.025**

Interpretation:
1. Several process-quality features show temporal persistence from the beginning to the end of meetings.
2. Facilitator dynamics and information-seeking quality appear especially stable/predictive in this temporal setup.
3. This supports the broader project story that early interaction patterns may shape downstream team process quality.
4. Important caveat: these models are fit/evaluated on the same small sample (N=16), so R² values should be treated as exploratory, not out-of-sample predictive performance.

## How the results connect into one overall story
This project shows a clear progression:
1. **Data pipeline success**: raw annotations were transformed into a consistent, session-level analytic dataset.
2. **Strong descriptive variation**: conferences and sessions differ meaningfully in behavior and outcomes.
3. **Modest predictive signal**: models can do better than random in some settings, but performance is unstable and generally not strong.
4. **Outcome-specific predictability**: funded-team prediction sometimes looks better than team-formation prediction, especially in selected multivariate runs.
5. **No single dominant driver**: effects are distributed across multiple behavior/process features, with direction and strength changing by model setup.

## Final takeaway
The project provides a credible behavioral measurement pipeline and evidence of **real but limited** predictive signal. The main value right now is not a high-accuracy prediction tool, but a structured way to identify behavioral patterns that are *potentially* linked to collaboration outcomes and worth deeper follow-up.
