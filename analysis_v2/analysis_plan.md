## Overview

This document is an **execution-status analysis plan** for the multimodal AI annotation of team meetings project, based on completed notebooks in `analysis_v2/notebooks/`.

Primary research questions:

1. **Session-level**: Can behavioral features annotated by a multimodal LLM pipeline predict whether a Scialog session produces teams and/or funded teams?
2. **Person-level**: Which behavioral patterns — and which individuals — are more likely to join teams or funded teams?
3. **Temporal window**: How much of a session is needed to predict outcomes — does early behavior carry disproportionate signal?
4. **Validation**: Does the AI annotation pipeline agree with trained human coders at acceptable reliability levels?

## Data Used (Actual)

- **Chunk registry**: `1,310` chunks from `196` recordings across `8` Scialog conferences.
- **Conferences**: `2020NES` (142 chunks), `2021ABI` (203), `2021CMC` (167), `2021MND` (170), `2021MZT` (168), `2021NES` (163), `2021SLU` (167), `2022MND` (130).
- **Outcome data**: Session-level JSON files in `analysis_v1/data/<conferenceID>/<conferenceID>_session_outcomes.json`.
- **Outcome missingness**: `50` chunks have no outcome data; `9` session groups had no outcome match.
- **Annotation JSONs loaded**: `1,286` (24 chunks skipped: empty, malformed, or missing JSON).
- **Session-level model data**: `156` sessions with usable outcomes.
- **Person-level model data**: `504` global-person rows (stratified CV); `639` person-conference rows (LOCO); `579` non-facilitator rows.
- **Interrater validation**: `212` matched utterances across `20` code labels.
- **Human expert validation**: `212` utterances coded by Max; `226` coded by Gemini; `14` unmatched Gemini rows.

## Compact Results Table

| Analysis | Data / unit | Outcome | Main result | Saved artifact |
|---|---:|---|---|---|
| Stage 0 registry build | 1,310 chunks, 196 recordings, 8 conferences | Session outcomes joined from v1 JSONs | 50 chunks lack outcome data; 9 session groups had no outcome match | `analysis_v2/data/chunk_registry_v1.csv` |
| Stage 4 feature engineering | 1,310 registry rows; 1,286 JSONs loaded | Chunk/session/model-ready feature tables | Chunk features `1,310 × 101`; session features `162 × 493`; model-ready features `162 × 360`; feature manifest `483 × 3` | `analysis_v2/results/tables/4-feature_engineering/` |
| Session logistic LOSO | 156 sessions | `outcome_has_teams` | AUC `0.4573`; 95% CI `0.3478–0.5751`; balanced accuracy `0.4586`; F1 `0.7186`; confusion `[[8,25],[40,83]]` | `analysis_v2/results/tables/6-regression_modeling/loso_auc_summary.csv` |
| Session logistic LOSO | 156 sessions | `outcome_has_funded_teams` | AUC `0.6222`; 95% CI `0.5288–0.7162`; balanced accuracy `0.5976`; F1 `0.5507`; confusion `[[56,32],[30,38]]` | `analysis_v2/results/tables/6-regression_modeling/loso_auc_summary.csv` |
| Session logistic LOCO | 156 sessions | `outcome_has_teams` | AUC `0.4090`; 95% CI `0.2968–0.5260`; balanced accuracy `0.4313`; F1 `0.6987`; confusion `[[7,26],[43,80]]` | `analysis_v2/results/tables/6-regression_modeling/loco_auc_summary.csv` |
| Session logistic LOCO | 156 sessions | `outcome_has_funded_teams` | AUC `0.5655`; 95% CI `0.4639–0.6618`; balanced accuracy `0.5551`; F1 `0.4925`; confusion `[[55,33],[35,33]]` | `analysis_v2/results/tables/6-regression_modeling/loco_auc_summary.csv` |
| Session random forest LOSO | 156 sessions | `outcome_has_teams` | AUC `0.6194`; 95% CI `0.5072–0.7260`; balanced accuracy `0.5525`; F1 `0.8864`; confusion `[[4,29],[2,121]]` | `analysis_v2/results/tables/6-regression_modeling/random_forest_results.csv` |
| Session random forest LOSO | 156 sessions | `outcome_has_funded_teams` | AUC `0.5234`; 95% CI `0.4281–0.6166`; balanced accuracy `0.4903`; F1 `0.3243`; confusion `[[63,25],[50,18]]` | `analysis_v2/results/tables/6-regression_modeling/random_forest_results.csv` |
| Beginning-segment logistic LOSO | 156 sessions | `outcome_has_teams` | AUC `0.4398`; 95% CI `0.3218–0.5620`; balanced accuracy `0.4804`; F1 `0.8231` | `analysis_v2/results/tables/6-regression_modeling/beginning_segment_results.csv` |
| Beginning-segment logistic LOSO | 156 sessions | `outcome_has_funded_teams` | AUC `0.4948`; 95% CI `0.4088–0.5834`; balanced accuracy `0.4418`; F1 `0.3511` | `analysis_v2/results/tables/6-regression_modeling/beginning_segment_results.csv` |
| Count outcome elasticnet LOSO | 156 sessions | `outcome_num_teams` | RMSE `1.2882`; MAE `0.9798`; R² `−0.2270` | `analysis_v2/results/tables/6-regression_modeling/count_outcome_summary.csv` |
| Count outcome elasticnet LOSO | 156 sessions | `outcome_num_funded_teams` | RMSE `0.8168`; MAE `0.6203`; R² `−0.2874` | `analysis_v2/results/tables/6-regression_modeling/count_outcome_summary.csv` |
| Person model, global person CV | 504 global-person rows | joined team | AUC `0.8594`; AUPRC `0.8096`; prevalence `0.4722` | `analysis_v2/results/tables/7-person_level_modeling/person_model_comparison_summary.csv` |
| Person model, global person CV | 504 global-person rows | joined funded team | AUC `0.7916`; AUPRC `0.4776`; prevalence `0.2440` | `analysis_v2/results/tables/7-person_level_modeling/person_model_comparison_summary.csv` |
| Person model, LOCO | 639 person-conference rows | joined team | AUC `0.8357`; AUPRC `0.7581`; prevalence `0.4601` | `analysis_v2/results/tables/7-person_level_modeling/person_model_loco_summary.csv` |
| Person model, LOCO | 639 person-conference rows | joined funded team | AUC `0.7553`; AUPRC `0.3737`; prevalence `0.2050` | `analysis_v2/results/tables/7-person_level_modeling/person_model_loco_summary.csv` |
| Person model visualization summary | 639 person-conference rows | joined team | AUC `0.8580`; AUPRC `0.8074`; accuracy `0.7809`; F1 `0.7535`; 61 features | `analysis_v2/results/tables/7-person_level_modeling/person_model_visualization_summary.csv` |
| Person model visualization summary | 639 person-conference rows | joined funded team | AUC `0.7947`; AUPRC `0.4487`; accuracy `0.7246`; F1 `0.5111`; 17 features | `analysis_v2/results/tables/7-person_level_modeling/person_model_visualization_summary.csv` |
| Non-facilitator sensitivity | 579 non-facilitator rows | joined team | AUC `0.8325`; AUPRC `0.8121`; accuracy `0.7513`; F1 `0.7419` | `analysis_v2/results/tables/7-person_level_modeling/person_model_non_facilitator_sensitivity.csv` |
| Non-facilitator sensitivity | 579 non-facilitator rows | joined funded team | AUC `0.7675`; AUPRC `0.4475`; accuracy `0.7168`; F1 `0.5176` | `analysis_v2/results/tables/7-person_level_modeling/person_model_non_facilitator_sensitivity.csv` |
| High-level vs detailed person model | 639 rows | joined team | High-level: AUC `0.8749`, AUPRC `0.8183`; detailed: AUC `0.8580`, AUPRC `0.8074` | `analysis_v2/results/tables/7-person_level_modeling/person_model_highlevel_vs_detailed.csv` |
| Heckman two-stage model | 639 all rows; 294 selected | joined team then funded team | Selection AUC `0.8936`; selected-only funded AUC `0.6609`; IMR p `0.7619` | `analysis_v2/results/tables/7-person_level_modeling/person_model_heckman_two_stage_summary.csv` |
| Heckman two-stage model (non-fac) | 579 non-fac rows; 294 selected | joined team then funded team | Selection AUC `0.8687`; selected-only funded AUC `0.6585`; IMR p `0.8212` | `analysis_v2/results/tables/7-person_level_modeling/person_model_heckman_two_stage_summary.csv` |
| Temporal person-window model | 579 aligned rows, same subcode set | joined team | First 1 min AUC `0.731`; first 5 min `0.779`; full session `0.833`; last 5 min `0.810`; last 1 min `0.762` | `analysis_v2/results/tables/8-temporal_predictive_power/temporal_window_model_summary.csv` |
| Human/Gemini utterance-code agreement | 212 matched utterances; 20 code labels | utterance code presence | PABAK: Idea Novelty `0.991`; Prior Relationship `0.991`; Complementarity `0.887`; Broader Significance `0.868`; Epistemic Bridging `0.736`; Evaluation Practices `0.660` | `analysis_v2/notebooks/interrater_pabak.csv` |

## Notebooks and Analyses

### `0-build_registry.ipynb`

#### Notebook Scope and Global Settings

Purpose: build the master chunk registry from all top-level `*_path_dict.json` files and join session-level outcome metadata from `analysis_v1/data`.

Ground-truth audited notebook: `analysis_v2/notebooks/0-build_registry.ipynb`.

Output directory: `analysis_v2/data/`.

#### Setup and Imports

Step-by-step:

1. Defines standardized output paths under `analysis_v2/data`, `analysis_v2/results/tables/baseline`, and `analysis_v2/results/figures/baseline`.
2. Imports `pandas`, `json`, `pathlib`, and related utilities.

#### Build Chunk Registry

Step-by-step:

1. Discovers 8 path-dict files at repository root: `2020NES_path_dict.json`, `2021ABI_path_dict.json`, `2021CMC_path_dict.json`, `2021MND_path_dict.json`, `2021MZT_path_dict.json`, `2021NES_path_dict.json`, `2021SLU_path_dict.json`, `2022MND_path_dict.json`.
2. Parses per-conference session-outcome JSON files from `analysis_v1/data/<conferenceID>/<conferenceID>_session_outcomes.json`. Each file maps `session_group` → `{teams: {team_id: {members, funded_status}}}`. Derives `num_teams`, `num_funded_teams`, `has_teams`, `has_funded_teams`.
3. Expands each path-dict chunk into one registry row with: `chunk_id`, `session_key`, `chunk_file_name`, `chunk_path`, `chunk_index`, `n_chunks_in_session`, `chunk_position` (beginning/middle/end/whole), `analyzed` flag, `conference_id`, and outcome fields.
4. Derives `chunk_position` via a position-label helper: index 0 → `beginning`, index n-1 → `end`, single chunk → `whole`, otherwise → `middle`.
5. Validates uniqueness of `chunk_id`.
6. Initializes `human_validation_set`, `utterance_validation_set`, and `oversampled_for` columns (all False/None — filled in Stage 3a).
7. Prints distribution by conference and chunk-position.
8. Saves `analysis_v2/data/chunk_registry_v1.parquet` and `analysis_v2/data/chunk_registry_v1.csv`.

Executed output:

- Registry size: `1,310` chunks from `196` recordings across `8` conferences.
- Outcome matching gap: `9` session groups had no outcome entry.
- Missing outcome rows: `50` chunks have no outcome data.
- Distribution by conference: `2020NES=142`, `2021ABI=203`, `2021CMC=167`, `2021MND=170`, `2021MZT=168`, `2021NES=163`, `2021SLU=167`, `2022MND=130`.

Tables:

- `analysis_v2/data/chunk_registry_v1.parquet`
- `analysis_v2/data/chunk_registry_v1.csv`

---

### `3a-sample_validation_set.ipynb`

#### Notebook Scope and Global Settings

Purpose: select the human-validation sample and update the registry with validation flags.

Ground-truth audited notebook: `analysis_v2/notebooks/3a-sample_validation_set.ipynb`.

#### Validation Tier Definitions

Step-by-step:

1. Loads `chunk_registry_v1`.
2. Defines validation tiers for chunk-level fields:
   - **Tier 1** (blockers for model use): `idea_trajectory`, `collective_engagement_level`, `explicit_commitment_signal`, `decision_crystallization_level`, `pronoun_shift_flag`, `cross_disciplinary_bridging`, `shared_vision_indicator`.
   - **Tier 2** (include with caveat if kappa 0.40–0.59): `problem_specificity_level`, `ambition_level`, `laughter_quality`, `dissent_response_quality`, `risk_acknowledgment_with_enthusiasm`, `personal_disclosure`, `meeting_structure_quality`.
   - **Tier 3** (descriptive; AI-only acceptable): `screenshare_active`, `artifact_interaction`, `funding_awareness_signal`, `prior_relationship_signal`, `explicit_complementarity_recognition`, `skill_gap_identification`.
3. Defines priority utterance fields: `Idea_Management`, `Integration_Practices`, `Pronoun_Framing`, `interruption_type`.

#### Stratified Sampling

Step-by-step:

1. Derives a combined stratification key from `conference_id`, `chunk_position`, and `outcome_has_funded_teams`.
2. Samples 20% of chunks using `StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)`.
3. Optionally oversamples rare AI-positive chunks for `explicit_commitment_signal`, `cross_disciplinary_bridging`, and `risk_acknowledgment_with_enthusiasm` (target: min 15 positive examples each).
4. Selects an utterance-level subsample (50 chunks) from within the chunk validation set using stratified sampling at 25% within each conference × chunk-position cell.
5. Saves `analysis_v2/data/chunk_registry_v2.parquet` and `analysis_v2/data/chunk_registry_v2.csv`.

Audit note: the notebook code is present, but the audited notebook had no saved execution outputs for the sample counts. Treat sample sizes as not independently verified from notebook output.

Tables:

- `analysis_v2/data/chunk_registry_v2.parquet`
- `analysis_v2/data/chunk_registry_v2.csv`

---

### `3b-export_coding_materials.ipynb`

#### Notebook Scope and Global Settings

Purpose: export human-coding materials from the validation registry.

Ground-truth audited notebook: `analysis_v2/notebooks/3b-export_coding_materials.ipynb`.

#### Export Steps

Step-by-step:

1. Loads the validation registry (`chunk_registry_v2`).
2. Defines three instruments:
   - **Instrument A** (intellectual trajectory): `idea_trajectory`, `problem_specificity_level`, `decision_crystallization_level`, `ambition_level`, `cross_disciplinary_bridging`, `explicit_commitment_signal`.
   - **Instrument B** (social dynamics): `pronoun_shift_flag`, `shared_vision_indicator`, `laughter_quality`, `personal_disclosure`, `dissent_response_quality`, `risk_acknowledgment_with_enthusiasm`, `meeting_structure_quality`.
   - **Instrument C** (behavioral responsiveness): `collective_engagement_level` and its sub-signals (nods, facial expressions, backchannels, cameras-off).
3. Exports per-chunk coding-sheet CSVs to `data/human_coding/materials/A/`, `B/`, and `C/`.
4. Copies chunk video files (`.mp4`) to `data/human_coding/videos/`.
5. Saves a validation-materials manifest.

Audit note: code is present, but no saved execution outputs were embedded in the notebook. Treat as implemented but not independently verified from notebook output.

---

### `3c-compute_agreement.ipynb`

#### Notebook Scope and Global Settings

Purpose: compute human-human reliability, human-AI agreement, disagreement summaries, and feature-inclusion decisions.

Ground-truth audited notebook: `analysis_v2/notebooks/3c-compute_agreement.ipynb`.

#### Agreement Computation

Step-by-step:

1. Defines instrument fields and reliability field groups.
2. Loads registry and rater files.
3. Flags disagreements and writes a disagreement summary CSV.
4. Computes human-human reliability: Cohen's kappa for binary/categorical fields, ICC(2,1) for ordinal fields.
5. Computes human-AI agreement against resolved human codes.
6. Applies inclusion thresholds: include at `κ ≥ 0.60`, caveat at `0.40–0.59`, exclude below `0.40`.

Audit note: code is present, but no saved execution outputs were embedded in the notebook. See `interrater_reliability.ipynb` for executed Human/Gemini PABAK results.

---

### `4-feature_engineering.ipynb`

#### Notebook Scope and Global Settings

Purpose: transform raw AI JSON annotations into chunk-level, session-level, and model-ready features.

Ground-truth audited notebook: `analysis_v2/notebooks/4-feature_engineering.ipynb`.

Output paths: `analysis_v2/results/tables/4-feature_engineering/`.

#### `## Setup and Imports`

Step-by-step:

1. Discovers repository root by finding `analysis_v2/` and `analysis_v1/` directories.
2. Imports `pandas`, `numpy`, `json`, `pathlib`, statistical utilities.
3. Defines output paths for chunk features, session features, model-ready features, and feature manifest.
4. Creates output directories.

Executed output:

- Registry loaded: `1,310` chunks from `196` recordings across `8` conferences.

#### `## Load Annotation JSONs`

Step-by-step:

1. Loads `chunk_registry_v1.csv`.
2. Resolves output JSON paths from registry `chunk_path` entries.
3. Loads each AI output JSON; skips files that are empty, malformed, or missing.

Executed output:

- JSONs loaded: `1,286`.
- Skipped chunks: `24` (empty, malformed, or missing JSON).

#### `## Chunk-Level Feature Extraction`

Step-by-step:

1. Maps noisy or variant code names into canonical code categories from `chunk_summary` fields.
2. Computes chunk-level features including:
   - **Participation**: speaking-time Gini coefficient, dominant-speaker flag.
   - **Trajectory**: `idea_trajectory` (divergent/convergent/procedural/ambiguous).
   - **Engagement**: `collective_engagement_level` (1–4 rating).
   - **Bridging**: `cross_disciplinary_bridging` binary.
   - **Commitment**: `explicit_commitment_signal` binary.
   - **Artifact**: `screenshare_active`, `artifact_interaction`.
   - **Specificity/crystallization**: `problem_specificity_level` (1–4), `decision_crystallization_level` (1–4).
   - **Ambition**: `ambition_level` (incremental/novel_application/novel_combination/paradigm_challenging).
   - **Complementarity**: `explicit_complementarity_recognition`, `skill_gap_identification`.
   - **Shared vision**: `shared_vision_indicator`, `pronoun_shift_flag`.
   - **Social**: `personal_disclosure`, `laughter_quality`, `dissent_response_quality`, `risk_acknowledgment_with_enthusiasm`.
   - **Funding/relationships**: `funding_awareness_signal`, `prior_relationship_signal`.
   - **Structure**: `meeting_structure_quality`.

#### `## Utterance-Level Feature Aggregation`

Step-by-step:

1. Aggregates utterance-level code counts per chunk: counts of each `code_name` category.
2. Aggregates subcode counts.
3. Aggregates `idea_quality` scores (0/1/2) for Idea Management, Integration Practices, Knowledge Sharing.
4. Aggregates multimodal signals: `vocal_enthusiasm`, `hesitation_flag`, `pace`, `nod_count`, `shared_affect`, `any_smile_other`, `audible_backchannel`, `interruption_type`.
5. Builds responsiveness-index features from combined multimodal signals.

#### `## Session-Level Feature Aggregation`

Step-by-step:

1. Aggregates chunk features to session level: means, sums, beginning/middle/end segment values, and delta features (end minus beginning).
2. Joins session-level outcome variables and conference metadata.
3. Runs multicollinearity checks (variance inflation factors, pairwise correlations).
4. Builds a feature manifest listing feature name, source, and family.
5. Drops near-zero-variance and high-VIF columns to build the model-ready table.

Executed output:

- Chunk features: `1,310 × 101`.
- Session features: `162 × 493`.
- Model-ready features: `162 × 360`.
- Feature manifest: `483 × 3`.

Tables:

- `analysis_v2/results/tables/4-feature_engineering/chunk_features.csv`
- `analysis_v2/results/tables/4-feature_engineering/session_features.csv`
- `analysis_v2/results/tables/4-feature_engineering/model_ready_features.csv`
- `analysis_v2/results/tables/4-feature_engineering/feature_manifest.csv`

---

### `5-descriptive_analysis.ipynb`

#### Notebook Scope and Global Settings

Purpose: descriptive figures and feature-distribution diagnostics.

Ground-truth audited notebook: `analysis_v2/notebooks/5-descriptive_analysis.ipynb`.

#### Descriptive Steps

Step-by-step:

1. Loads Stage 4 outputs (`session_features.csv`, `chunk_features.csv`).
2. Creates conference-level outcome summaries (n sessions, team rates, funded-team rates).
3. Plots chunk-position profiles for key features across beginning/middle/end.
4. Plots validation/reliability results when `human_irr_results.csv` and `human_ai_agreement.csv` exist.
5. Writes feature-distribution plots and outlier diagnostics.

Audit note: code structure is present, but no saved execution outputs were embedded in the notebook. Existing generic figures in `analysis_v2/results/figures/baseline/figure_001.png` through `figure_006.png` are not self-descriptive enough to serve as final paper assets without renaming.

Figures:

- `analysis_v2/results/figures/baseline/figure_001.png` through `figure_006.png` (existing, not self-descriptively named)
- Curated copies: `analysis_v2/figures/reporting/session-level/`, `analysis_v2/figures/reporting/validation/`

---

### `6-regression_modeling.ipynb`

#### Notebook Scope and Global Settings

Notebook title: session-level predictive modeling for team and funded-team binary and count outcomes.

Ground-truth audited notebook: `analysis_v2/notebooks/6-regression_modeling.ipynb`.

Output paths: `analysis_v2/results/tables/6-regression_modeling/`, `analysis_v2/results/figures/`.

Global settings:

- Model table: `analysis_v2/results/tables/4-feature_engineering/model_ready_features.csv`.
- Candidate modeling features: `354`.
- Beginning-only features: `64`.
- Session rows after dropping missing outcomes: `156`.
- Primary CV scheme: leave-one-session-out (LOSO).
- Robustness CV scheme: leave-one-conference-out (LOCO).
- Logistic regression: fixed regularization (replaced original nested `LogisticRegressionCV` inside LOSO for runtime).
- Random forest: `100` trees, LOSO.
- Count outcomes: `ElasticNetCV`, LOSO.
- Random state: `42`.

#### `## Setup and Data Loading`

Step-by-step:

1. Loads model-ready features from Stage 4.
2. Selects `354` candidate features and `64` beginning-only features.
3. Drops rows with missing outcome values; retains `156` rows.
4. Defines binary outcome variables: `outcome_has_teams`, `outcome_has_funded_teams`.
5. Defines count outcome variables: `outcome_num_teams`, `outcome_num_funded_teams`.

#### `## Primary LOSO Logistic Regression`

Step-by-step:

1. For each session, holds it out and trains a fixed-regularization logistic regression on all remaining sessions.
2. Collects out-of-fold predicted probabilities.
3. Computes AUC, 95% bootstrap CI, balanced accuracy, F1, and confusion matrix.
4. Repeats for both binary outcomes.

Executed output:

- `outcome_has_teams`: AUC `0.4573`; 95% CI `0.3478–0.5751`; balanced accuracy `0.4586`; F1 `0.7186`; confusion `[[8,25],[40,83]]`.
- `outcome_has_funded_teams`: AUC `0.6222`; 95% CI `0.5288–0.7162`; balanced accuracy `0.5976`; F1 `0.5507`; confusion `[[56,32],[30,38]]`.

Tables:

- `analysis_v2/results/tables/6-regression_modeling/loso_auc_summary.csv`

Figures:

- `analysis_v2/figures/reporting/session-level/roc_curves_primary_models.png`

#### `## LOCO Robustness Check`

Step-by-step:

1. Groups sessions by conference (8 groups).
2. Holds out each conference and trains on all other conferences.
3. Reports same metrics as LOSO for each binary outcome.

Executed output:

- `outcome_has_teams`: AUC `0.4090`; 95% CI `0.2968–0.5260`; balanced accuracy `0.4313`; F1 `0.6987`; confusion `[[7,26],[43,80]]`.
- `outcome_has_funded_teams`: AUC `0.5655`; 95% CI `0.4639–0.6618`; balanced accuracy `0.5551`; F1 `0.4925`; confusion `[[55,33],[35,33]]`.

Tables:

- `analysis_v2/results/tables/6-regression_modeling/loco_auc_summary.csv`

#### `## Random Forest Robustness`

Step-by-step:

1. Fits a random forest (`n_estimators=100`, `random_state=42`) in LOSO scheme.
2. Reports AUC, balanced accuracy, F1, and confusion matrix.
3. Extracts feature importances.

Executed output:

- `outcome_has_teams`: AUC `0.6194`; 95% CI `0.5072–0.7260`; balanced accuracy `0.5525`; F1 `0.8864`; confusion `[[4,29],[2,121]]`.
- `outcome_has_funded_teams`: AUC `0.5234`; 95% CI `0.4281–0.6166`; balanced accuracy `0.4903`; F1 `0.3243`; confusion `[[63,25],[50,18]]`.

Tables:

- `analysis_v2/results/tables/6-regression_modeling/random_forest_results.csv`

Figures:

- `analysis_v2/figures/reporting/session-level/feature_importance_rf_permutation.png`
- `analysis_v2/figures/reporting/session-level/feature_importance_lasso.png`

#### `## Beginning-Segment Model`

Step-by-step:

1. Restricts features to `64` beginning-only features (beginning-chunk aggregates only).
2. Runs same LOSO logistic regression scheme.
3. Compares beginning-segment AUC against full-session AUC.

Executed output:

- `outcome_has_teams`: AUC `0.4398`; 95% CI `0.3218–0.5620`; balanced accuracy `0.4804`; F1 `0.8231`.
- `outcome_has_funded_teams`: AUC `0.4948`; 95% CI `0.4088–0.5834`; balanced accuracy `0.4418`; F1 `0.3511`.

Tables:

- `analysis_v2/results/tables/6-regression_modeling/beginning_segment_results.csv`

Figures:

- `analysis_v2/figures/reporting/session-level/beginning_vs_full_session_auc_comparison.png`

#### `## Count Outcome ElasticNet`

Step-by-step:

1. Fits `ElasticNetCV` in LOSO scheme for `outcome_num_teams` and `outcome_num_funded_teams`.
2. Reports RMSE, MAE, and R².

Executed output:

- `outcome_num_teams`: RMSE `1.2882`; MAE `0.9798`; R² `−0.2270`.
- `outcome_num_funded_teams`: RMSE `0.8168`; MAE `0.6203`; R² `−0.2874`.

Tables:

- `analysis_v2/results/tables/6-regression_modeling/count_outcome_summary.csv`

Runtime note: nested `LogisticRegressionCV` and `ElasticNetCV` inside LOSO loops were replaced with fixed regularized models to enable end-to-end execution in a practical runtime; the outer LOSO/LOCO design is preserved.

---

### `7-person_level_modeling.ipynb`

#### Notebook Scope and Global Settings

Purpose: person-level feature construction and prediction of who joins a team or funded team.

Ground-truth audited notebook: `analysis_v2/notebooks/7-person_level_modeling.ipynb`.

Output paths: `analysis_v2/results/tables/7-person_level_modeling/`, `analysis_v2/figures/person-aggregation-features/`.

#### `## Setup and Participant Identity`

Step-by-step:

1. Loads global participant identity mappings (linking names across sessions and conferences).
2. Loads participant alias mapping from `analysis_v2/notebooks/participant_alias_mapping.csv`.
3. Discovers repository root using path walking.

#### `## Person-Level Feature Construction`

Step-by-step:

1. Iterates over all annotated sessions and aggregates per-person, per-conference features.
2. Person-conference features include: sessions attended, chunks seen, utterance count, speaking seconds, dominant-speaker chunk count, hesitation count, backchannel count.
3. Aggregates behavioral code counts and subcode counts per person per conference.
4. Flags facilitator status.
5. Joins team and funded-team outcome: `outcome_joined_team` (binary), `outcome_joined_funded_team` (binary).

Saved class balance:

- `outcome_joined_team`: `639` total rows, `294` positives, `345` negatives, positive rate `0.4601`.
- `outcome_joined_funded_team`: `639` total rows, `131` positives, `508` negatives, positive rate `0.2050`.

#### `## Global-Person Stratified CV`

Step-by-step:

1. Collapses person-conference rows to unique global-person rows (`504` rows).
2. Runs stratified 5-fold cross-validation with logistic regression.
3. Reports AUC and AUPRC.

Executed output:

- Joined team: AUC `0.8594`, AUPRC `0.8096`.
- Joined funded team: AUC `0.7916`, AUPRC `0.4776`.

Tables:

- `analysis_v2/results/tables/7-person_level_modeling/person_model_comparison_summary.csv`

#### `## Within-Conference LOCO`

Step-by-step:

1. Uses person-conference rows (`639` rows).
2. Holds out each conference and trains on remaining conferences.
3. Reports AUC and AUPRC.

Executed output:

- Joined team: AUC `0.8357`, AUPRC `0.7581`.
- Joined funded team: AUC `0.7553`, AUPRC `0.3737`.

Tables:

- `analysis_v2/results/tables/7-person_level_modeling/person_model_loco_summary.csv`

#### `## Visualization Summary Models`

Step-by-step:

1. Fits logistic regression models for visualization: one for joined team (61 features), one for funded team (17 features after feature selection).
2. Reports AUC, AUPRC, accuracy, and F1.
3. Exports coefficient tables with p-values.

Executed output:

- Joined team: AUC `0.8580`, AUPRC `0.8074`, accuracy `0.7809`, F1 `0.7535`, `61` features.
- Joined funded team: AUC `0.7947`, AUPRC `0.4487`, accuracy `0.7246`, F1 `0.5111`, `17` features.

Key coefficients — joined team detailed model:

- **Positive significant**: `n_sessions_attended` coef `0.6798`, p `0.000268`; `checks_consensus` coef `0.7256`, p `0.0316`; `extends_existing_idea` coef `0.6805`, p `0.0341`.
- **Negative significant**: `proposes_process` coef `−1.8538`, p `0.0063`; `expresses_appreciation` coef `−0.8378`, p `0.0094`; `summarizes_for_group` coef `−0.7716`, p `0.0199`.

Key coefficients — joined funded-team detailed model:

- **Positive significant**: `n_sessions_attended` coef `0.5136`, p `0.000611`; `extends_existing_idea` coef `0.3910`, p `0.0330`.
- **Negative significant**: `proposes_process` coef `−1.3135`, p `0.0104`.

Key coefficients — non-facilitator high-level code model for joined team:

- **Positive significant**: `Idea Management` coef `0.9871`, p `0.000405`; `Epistemic Bridging` coef `0.3829`, p `0.0318`.
- **Negative significant**: `Coordination & Decision` coef `−0.5758`, p `0.0126`; `Relational Climate` coef `−0.4712`, p `0.0137`; `Participation Dynamics` coef `−1.0168`, p `0.0344`.

Key coefficients — high-level code model for joined funded team:

- `Knowledge Sharing` coef `0.5372`, p `0.0478` (only p < .05 code category in saved table).

Tables:

- `analysis_v2/results/tables/7-person_level_modeling/person_model_visualization_summary.csv`
- `analysis_v2/results/tables/7-person_level_modeling/person_model_coefficients_joined_team.csv`
- `analysis_v2/results/tables/7-person_level_modeling/person_model_coefficients_joined_funded_team.csv`

#### `## Non-Facilitator Sensitivity`

Step-by-step:

1. Drops facilitator-flagged rows; retains `579` non-facilitator rows.
2. Reruns visualization summary models.

Executed output:

- Joined team: AUC `0.8325`, AUPRC `0.8121`, accuracy `0.7513`, F1 `0.7419`.
- Joined funded team: AUC `0.7675`, AUPRC `0.4475`, accuracy `0.7168`, F1 `0.5176`.

Tables:

- `analysis_v2/results/tables/7-person_level_modeling/person_model_non_facilitator_sensitivity.csv`

#### `## High-Level vs Detailed Code Comparison`

Step-by-step:

1. Runs models using high-level code category counts vs. detailed subcode counts for joined team.
2. Compares AUC and AUPRC.

Executed output:

- High-level categories: AUC `0.8749`, AUPRC `0.8183`, accuracy `0.7903`, F1 `0.7666`.
- Detailed subcodes: AUC `0.8580`, AUPRC `0.8074`, accuracy `0.7809`, F1 `0.7535`.

Tables:

- `analysis_v2/results/tables/7-person_level_modeling/person_model_highlevel_vs_detailed.csv`

#### `## Heckman-Style Two-Stage Model`

Step-by-step:

1. **Stage 1 (selection)**: predict who joins a team using all rows; save predicted probabilities.
2. **Stage 2 (outcome)**: among team-joiners only (`294` rows), predict who joins a funded team.
3. Computes Inverse Mills Ratio (IMR) from Stage 1 and adds it as a covariate in Stage 2.
4. Runs both full-sample and non-facilitator versions.

Executed output:

- All rows: selection AUC `0.8936`; selected-only funded-team AUC `0.6609`; IMR p `0.7619` (not significant — no evidence of selection bias).
- Non-facilitators: selection AUC `0.8687`; selected-only funded-team AUC `0.6585`; IMR p `0.8212`.

Tables:

- `analysis_v2/results/tables/7-person_level_modeling/person_model_heckman_two_stage_summary.csv`

#### `## Presentation Figure Exports`

Step-by-step:

1. Generates coefficient charts for non-facilitator joined-team and joined-funded-team models.
2. Generates role/speaker-count model coefficient charts.
3. Generates feature correlation heatmap.

Speaker-role/count model coefficients:

- Count model: speaking minutes positive (coef `1.0221`, p < 0.001); dominant-speaker count negative (coef `−0.6170`, p < 0.001); sessions attended positive (coef `0.3823`, p `0.0011`); cross-disciplinary bridging positive (coef `0.3823`, p `0.0161`).
- Binary model: speaking minutes positive (coef `0.9566`, p < 0.001); ever dominant speaker negative (coef `−0.4377`, p < 0.001); sessions attended positive (coef `0.3837`, p `0.0011`); ever cross-disciplinary bridging positive (coef `0.3005`, p `0.0067`).

Figures:

- `analysis_v2/figures/person-aggregation-features/metrics_bar_joined_team.png`
- `analysis_v2/figures/person-aggregation-features/metrics_bar_joined_funded_team.png`
- `analysis_v2/figures/person-aggregation-features/person_model_results_outcome_joined_team.png`
- `analysis_v2/figures/person-aggregation-features/person_model_results_outcome_joined_funded_team.png`
- `analysis_v2/figures/person-aggregation-features/coef_chart_codename_nonfac.png`
- `analysis_v2/figures/person-aggregation-features/coef_chart_without_facilitators.png`
- `analysis_v2/figures/person-aggregation-features/feature_correlation_heatmap.png`
- `analysis_v2/figures/reporting/person-level/` (curated copies)

Cleanup note: this notebook mixes feature production, modeling, diagnostics, and presentation figure cells. For publication workflow it should be split into: feature construction, model estimation, and figure generation.

---

### `8-temporal_predictive_power.ipynb`

#### Notebook Scope and Global Settings

Purpose: estimate person-level predictive power from early and late temporal windows within a session.

Ground-truth audited notebook: `analysis_v2/notebooks/8-temporal_predictive_power.ipynb`.

Output paths: `analysis_v2/results/tables/8-temporal_predictive_power/`, `analysis_v2/figures/person-aggregation-features/`.

#### `## Setup and Helper Functions`

Step-by-step:

1. Discovers repository root using path walking.
2. Loads full-session person/subcode features from Stage 7.
3. Defines window extraction helper functions.

#### `## Temporal Window Extraction`

Step-by-step:

1. Extracts subcode records for four windows: first 1 minute, first 5 minutes, last 5 minutes, last 1 minute.
2. Aligns each window to the full-session feature set (same subcode columns where present, zero-filled otherwise).

Executed output:

- Total records collected: `10,680`.
- Window record counts: last 5 min `4,298`; first 5 min `3,910`; last 1 min `1,389`; first 1 min `1,083`.
- Person-session and subcode coverage by window:
  - First 1 min: `239` person-sessions, `46` subcodes.
  - First 5 min: `478` person-sessions, `57` subcodes.
  - Full session: `639` rows, `69` subcodes.
  - Last 5 min: `424` person-sessions, `60` subcodes.
  - Last 1 min: `305` person-sessions, `52` subcodes.

#### `## Window Models`

Step-by-step:

1. Fits comparable logistic regression models for each window and for full session.
2. Uses the same aligned subcode feature set across windows.
3. Collects AUC, AUPRC, accuracy, and F1 for each window.
4. Attempts statsmodels coefficient inference for window charts; falls back to sklearn coefficients when `Singular matrix` errors occur (sparse window data).

Executed output:

- First 1 min: AUC `0.731`, AUPRC `0.672`, accuracy `68.2%`, F1 `0.669`.
- First 5 min: AUC `0.779`, AUPRC `0.742`, accuracy `71.7%`, F1 `0.705`.
- Full session: AUC `0.833`, AUPRC `0.813`, accuracy `75.5%`, F1 `0.746`.
- Last 5 min: AUC `0.810`, AUPRC `0.803`, accuracy `73.9%`, F1 `0.729`.
- Last 1 min: AUC `0.762`, AUPRC `0.729`, accuracy `71.2%`, F1 `0.701`.

Tables:

- `analysis_v2/results/tables/8-temporal_predictive_power/temporal_window_model_summary.csv`

Figures:

- `analysis_v2/figures/person-aggregation-features/temporal_window_comparison.png`
- `analysis_v2/figures/person-aggregation-features/temporal_auc_lineplot.png`
- `analysis_v2/figures/reporting/temporal-windows/` (curated copies)

---

### `interrater_reliability.ipynb`

#### Notebook Scope and Global Settings

Purpose: compare Max-coded utterance labels against Gemini-coded labels to estimate human-AI agreement.

Ground-truth audited notebook: `analysis_v2/notebooks/interrater_reliability.ipynb`.

#### Matching and Agreement

Step-by-step:

1. Loads Max's annotations (`212` utterances).
2. Loads Gemini annotations (`226` utterances).
3. Merges on chunk key, timestamp, and speaker.
4. Builds binary code-presence matrices across code labels.
5. Computes PABAK (prevalence-adjusted bias-adjusted kappa) and percent agreement per code.
6. Saves results.

Executed output:

- Matched utterances: `212`.
- Unmatched Max rows: `0`.
- Unmatched Gemini rows: `14`.
- Codes evaluated: `20`; Pronoun Framing excluded from summary.
- PABAK results by code:

| Code | PABAK |
|---|---|
| Idea Novelty Signal | 0.991 |
| Prior Relationship Signal | 0.991 |
| Complementarity Articulation | 0.887 |
| Broader Significance | 0.868 |
| Role Anticipation | 0.840 |
| Future-Oriented Language | 0.774 |
| Epistemic Bridging | 0.736 |
| Participation Dynamics | 0.679 |
| Evaluation Practices | 0.660 |
| Relational Climate | 0.660 |

Tables:

- `analysis_v2/notebooks/interrater_pabak.csv`

Figures:

- `analysis_v2/notebooks/interrater_pabak.png`
- `analysis_v2/figures/reporting/validation/` (curated copy)

---

### `sample_v2.ipynb` (support notebook)

#### Notebook Scope and Global Settings

Purpose: create a video-level stratified human-coding sample with balanced code coverage. This is a support/sampling notebook, not a primary results notebook.

Ground-truth audited notebook: `analysis_v2/notebooks/sample_v2.ipynb`.

#### Sampling Steps

Step-by-step:

1. Builds a video registry from existing output JSONs.
2. Scans `1,325` total JSON files; excludes `149` for bad recording name; skips `35` empty/invalid JSONs.
3. Identifies `228` unique videos; targets 20% sample (`46` videos).
4. Samples `45` videos stratified by conference.
5. Greedily selects chunks and utterances to improve code coverage across all code labels.
6. Plots before/after code-distribution comparison.

Executed output:

- Sampled videos: `45`.
- Unique chunks sampled: `110`.
- Total utterances sampled: `226`.
- By-conference sampled utterances: `2020NES=16`, `2021ABI=18`, `2021CMC=32`, `2021MND=32`, `2021MZT=34`, `2021NES=36`, `2021SLU=26`, `2022MND=32`.

Tables:

- `analysis_v2/notebooks/sampled_v2.xlsx`

Figures:

- `analysis_v2/figures/reporting/sampling/` (curated copy)

---

## Visualization and Output Organization

Current organization is review-ready:

### Curated Reporting Folders

- `analysis_v2/figures/reporting/session-level/` — ROC curves, feature importance plots, beginning vs full session AUC comparison.
- `analysis_v2/figures/reporting/person-level/` — person model metric bars, coefficient charts, correlation heatmap.
- `analysis_v2/figures/reporting/temporal-windows/` — temporal window AUC comparison and line plots.
- `analysis_v2/figures/reporting/validation/` — PABAK charts.
- `analysis_v2/figures/reporting/sampling/` — code coverage before/after sampling.
- `analysis_v2/figures/reporting/README.md` — describes folder categories.

### Results Tables by Stage

- `analysis_v2/results/tables/4-feature_engineering/` — chunk features, session features, model-ready features, feature manifest.
- `analysis_v2/results/tables/6-regression_modeling/` — LOSO/LOCO AUC summaries, random forest results, beginning-segment results, count outcome summary.
- `analysis_v2/results/tables/7-person_level_modeling/` — person model comparison summary, LOCO summary, visualization summary, non-facilitator sensitivity, high-level vs detailed, Heckman two-stage summary, coefficient tables.
- `analysis_v2/results/tables/8-temporal_predictive_power/` — temporal window model summary.

### Support Files

- `analysis_v2/notebooks/participant_alias_mapping.csv` — participant name mapping across sessions.
- `analysis_v2/notebooks/unmatched_*participants.csv` — unmatched name review files.
- `analysis_v2/notebooks/finalized_matching_csvs/` — finalized participant matching.
- `analysis_v2/notebooks/gemini_behavior_codebook.pptx` — codebook reference.

---

## Notebook Cleanup Status

- **Fully executable end-to-end**: `4-feature_engineering.ipynb`, `6-regression_modeling.ipynb`, `7-person_level_modeling.ipynb`, `8-temporal_predictive_power.ipynb`.
- **Stepwise markdown headers added**: `7-person_level_modeling.ipynb`, `8-temporal_predictive_power.ipynb`.
- **Repository-root discovery**: Stages 7 and 8 use path walking rather than hard-coded personal paths.
- **Remaining caveats**:
  - Stage 7 emits statsmodels convergence warnings in a few coefficient models; notebook completes and writes all outputs.
  - Stage 5 descriptive figures still include older generic baseline files; curated reporting copies should be used for writing.
  - Stages 3a/3b/3c have no embedded executed outputs — treat as implemented but not independently verified.
