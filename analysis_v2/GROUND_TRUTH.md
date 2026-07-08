# Ground truth — team formation / funding prediction

Last updated: 2026-07-08

Data-tracking document. Four models: {session-level, person-level} x
{high-level codes (16 categories), low-level codes (~70 subcodes)}. Each
section below is descriptive stats, visualizations, and model detail only —
no interpretation. For methodology/planning see `analysis_plan.md` and
`research_project_plan_v2.md`.

## Status

| | High-level codes | Low-level codes |
|---|---|---|
| **Session-level** | done (below) | done (below) |
| **Person-level** | done (below) | done (below) |

All 4 models built as of 2026-07-08.

---

## Model 1 — Session-level, high-level codes

**Descriptive stats**
- n = 162 sessions, 8 conferences (rebuilt via the session-level name-matching
  fix — see change log)
- `has_teams` prevalence: 79.0% (128/162)
- `has_funded_teams` prevalence: 44.4% (72/162)
- Feature set: 493 raw session-level features -> 360 after VIF filtering
  (`model_ready_features.csv`)

**Visualizations**
- `figures/conference_outcome_distributions.png` — outcome rates by conference
- `figures/feature_conference_heatmap.png` — feature x conference heatmap
- `figures/chunk_position_profiles.png` — beginning/middle/end feature trajectories
- `figures/beginning_vs_full_session_auc_comparison.png`
- `figures/feature_importance_lasso.png`, `figures/feature_importance_rf_permutation.png`
- `figures/roc_curves_primary_models.png`

**Model details**

| Model | CV | Target | AUC | 95% CI |
|---|---|---|---|---|
| Logistic elasticnet | LOCO | has_teams | 0.525 | [0.421, 0.626] |
| Logistic elasticnet | LOCO | has_funded_teams | 0.554 | [0.472, 0.650] |
| Logistic elasticnet | LOSO | has_teams | 0.559 | — |
| Logistic elasticnet | LOSO | has_funded_teams | 0.613 | — |
| Random forest | LOSO | has_teams | 0.672 | [0.575, 0.779] |
| Random forest | LOSO | has_funded_teams | 0.543 | [0.454, 0.637] |

Top 15 features by univariate screening (logistic + controls, LOSO AUC;
`has_funded_teams` only — the notebook never ran this screen for
`has_teams`):

| Feature | AUC | p | q (BH-corrected) |
|---|---|---|---|
| Evaluation Practices (middle) | 0.645 | 0.066 | 1.0 |
| Dissent was present (mean) | 0.639 | 0.077 | 1.0 |
| Evaluation Practices (mean) | 0.637 | 0.081 | 1.0 |
| Dissent response exploratory (mean) | 0.635 | 0.085 | 1.0 |
| Knowledge Sharing (mean) | 0.607 | 0.172 | 1.0 |
| Knowledge Sharing (middle) | 0.606 | 0.177 | 1.0 |
| Dissent was present (middle) | 0.596 | 0.224 | 1.0 |
| Dissent response exploratory (middle) | 0.588 | 0.265 | 1.0 |
| Relational Climate (delta) | 0.587 | 0.270 | 1.0 |
| Coordination & Decision (mean) | 0.585 | 0.280 | 1.0 |
| Participation Dynamics (mean) | 0.579 | 0.312 | 1.0 |
| Relational Climate (end) | 0.577 | 0.328 | 1.0 |
| Meeting structure quality (mean) | 0.574 | 0.348 | 1.0 |
| Idea Management (mean) | 0.571 | 0.364 | 1.0 |
| Idea Management (middle) | 0.571 | 0.364 | 1.0 |

All 75 screened features: q=1.0 (BH correction). Mixed-effects model:
VIF=inf on dissent features, coefficients in the hundreds, SEs in the
hundred-thousands — not usable as reported (quasi-complete separation).
Full table: `results/tables/6-regression_modeling/univariate_screening_results.csv`.

---

## Model 2 — Person-level, high-level codes (codename model)

**Descriptive stats**
- n = 639 person-within-conference rows
- `outcome_joined_team` prevalence: 46.0%
- `outcome_joined_funded_team` prevalence: 20.5%
- Feature set: 16 codename categories + `n_sessions_attended` +
  `speaking_minutes_total` + `is_facilitator` = 19 features

**Visualizations**
- `figures/person-aggregation-features/coef_chart_codename_model.png`
- `figures/person-aggregation-features/coef_chart_codename_nonfac.png` (non-facilitator sensitivity)
- `figures/person-aggregation-features/coef_chart_codename_outcome2.png` (funded, conditional on joining)

**Model details**

Model: statsmodels logistic regression (coefficients) + sklearn logistic
pipeline with LOCO-CV (AUC). `outcome_joined_team`: AUC 0.875, AUPRC 0.818,
Accuracy 0.790, F1 0.767 (`person_model_highlevel_vs_detailed.csv`).

Full coefficient table, all 19 features, `outcome_joined_team`:

| Feature | coef | p | sig |
|---|---|---|---|
| n_sessions_attended | +0.520 | 0.0002 | *** |
| Idea Management | +1.096 | 0.0004 | *** |
| Coordination & Decision Practices | -0.984 | 0.013 | * |
| Relational Climate | -0.759 | 0.014 | * |
| Epistemic Bridging | +0.395 | 0.032 | * |
| Participation Dynamics | -1.640 | 0.034 | * |
| Complementarity Articulation | +0.260 | 0.099 | |
| Role Anticipation | +0.248 | 0.120 | |
| Information Seeking | +0.370 | 0.215 | |
| Integration Practices | +0.327 | 0.332 | |
| Broader Significance | -0.197 | 0.356 | |
| Speaking Minutes Total | +0.313 | 0.494 | |
| Idea Ownership & Attribution | -0.085 | 0.556 | |
| Knowledge Sharing | +0.199 | 0.565 | |
| Future-Oriented Language | +0.101 | 0.655 | |
| Idea Novelty Signal | +0.089 | 0.676 | |
| Pronoun Framing | +0.231 | 0.731 | |
| Evaluation Practices | +0.072 | 0.817 | |
| is_facilitator | -11.484 | 0.9999 | quasi-complete separation, not a usable estimate |

`outcome_joined_funded_team`: only `Knowledge Sharing` significant
(+0.537, p=0.048). Overall AUC for this target not currently saved to a
file (printed at runtime by `evey_new_analyses.py` Step 7 but not written to
disk) — rerun needed to capture it.

---

## Model 3 — Person-level, low-level codes (subcode model)

**Descriptive stats**
- n = 639 person-within-conference rows (same base population as Model 2)
- `outcome_joined_team` prevalence: 46.0%
- `outcome_joined_funded_team` prevalence: 20.5%
- Feature set: ~70 subcodes + controls, 61 retained in the fitted model

**Visualizations**
- `figures/person_model_roc_curves.png`
- `figures/person-aggregation-features/person_model_results_outcome_joined_team.png`
- `figures/person-aggregation-features/person_model_results_outcome_joined_funded_team.png`
- `figures/person-aggregation-features/slide12_chart_300dpi.png` (non-facilitator, p<0.05 only)

**Model details**

| CV scheme | Target | AUC | AUPRC |
|---|---|---|---|
| LOCO | outcome_joined_team | 0.836 | 0.758 |
| LOCO | outcome_joined_funded_team | 0.755 | 0.374 |
| Stratified K-fold (global person) | outcome_joined_team | 0.859 | 0.810 |
| Stratified K-fold (global person) | outcome_joined_funded_team | 0.792 | 0.478 |

Top 20 of 61 features by p-value, `outcome_joined_team`:

| Feature | coef | p | sig |
|---|---|---|---|
| n_sessions_attended | +0.680 | 0.0003 | *** |
| identifies_common_ground | +0.571 | 0.005 | ** |
| proposes_process | -1.854 | 0.006 | ** |
| expresses_appreciation | -0.838 | 0.009 | ** |
| summarizes_for_group | -0.772 | 0.020 | * |
| checks_consensus | +0.726 | 0.032 | * |
| combines_ideas | +0.461 | 0.034 | * |
| extends_existing_idea | +0.681 | 0.034 | * |
| reframes_cross_disciplinarily | +0.400 | 0.045 | * |
| explicit_role_assignment | +0.384 | 0.065 | |
| shares_personal_experience | -0.418 | 0.071 | |
| asks_for_opinion | +0.571 | 0.071 | |
| connects_methods | +0.323 | 0.076 | |
| uses_humor | -0.488 | 0.083 | |
| setback_response_explores | -0.241 | 0.089 | |
| records_or_documents | +0.375 | 0.097 | |
| ambiguous | -0.661 | 0.131 | |
| returns_to_earlier_idea | +0.335 | 0.138 | |
| critiques_or_challenges | +0.320 | 0.147 | |
| translates_terminology | -0.270 | 0.148 | |

`outcome_joined_funded_team`: significant features are `extends_existing_idea`
(+0.391, p=0.033) and `proposes_process` (-1.314, p=0.010).

Full 61-row table: `data/person-aggregation-features/person_model_coefficients_outcome_joined_team.csv`
and `..._outcome_joined_funded_team.csv`.

---

## Model 4 — Session-level, low-level codes (subcodes)

**Built 2026-07-08.** Same aggregation scheme as Model 1 (mean / beginning /
middle / end / delta per session), applied to the 70 subcodes instead of the
16 categories. Same file-resolution logic as `4-feature_engineering.ipynb`
(`normalize_filename`/`resolve_json_path`, reused verbatim) — 1286/1310
chunk JSONs resolved (98.2%).

**Descriptive stats**
- n = 162 sessions (same population as Model 1)
- 70 distinct subcodes -> 350 session-level features (same 5-way temporal
  aggregation as Model 1)
- `has_teams` / `has_funded_teams` prevalence: same as Model 1 (79.0% / 44.4%)

**Model details** (same specification as Model 1: logistic L2 with
`class_weight='balanced'`, random forest, 100 trees):

| Model | CV | Target | AUC |
|---|---|---|---|
| Logistic | LOCO | has_teams | 0.571 |
| Logistic | LOSO | has_teams | 0.577 |
| Random forest | LOSO | has_teams | 0.655 |
| Logistic | LOCO | has_funded_teams | 0.520 |
| Logistic | LOSO | has_funded_teams | 0.594 |
| Random forest | LOSO | has_funded_teams | 0.584 |

**Univariate screening — methodology note.** The screening test (single
feature + LOSO, asymptotic z-approximation for the p-value) is identical to
Model 1's. Several of the 70 subcode-level features are sparse with heavy
ties at zero; on those, the z-approximation produces AUC exactly 0.000 with
p~1e-10, which is a known approximation-validity failure on near-degenerate
discrete data, not a real effect (Model 1's smoother category-level
features didn't hit this). Raw count: 225 features clear Benjamini-Hochberg
correction for `has_teams`. After filtering to features with >=20% non-zero
sessions: 68. Neither count has been checked with a proper permutation test.

One feature manually checked for non-degeneracy:
`session_mean_num_synthesizes_contributions` — AUC 0.681, p=0.021, q=0.033,
157/162 sessions non-zero, smooth distribution 0-3+.

Correlation of two specific subcodes with session-level outcomes (for
comparison against Models 2/3, see cross-model section below):
- `session_mean_num_expresses_appreciation`: r=+0.199 with `has_teams`,
  r=+0.121 with `has_funded_teams`
- `session_mean_num_summarizes_for_group`: r=+0.017 with `has_teams`,
  r=+0.099 with `has_funded_teams`

Full session-subcode table: `results/tables/4-feature_engineering/session_subcode_features.csv`.
Screening table: `results/tables/4-feature_engineering/model4_univariate_screening.csv`.

---

## Cross-model comparison (data only)

Model 2 (category) vs. Model 3 (subcode) coefficients, same construct:

| Category (Model 2) | coef | Subcodes within it (Model 3) | coef |
|---|---|---|---|
| Relational Climate | -0.759 | expresses_appreciation | -0.838 |
| Participation Dynamics | -1.640 | summarizes_for_group | -0.772 |
| Coordination & Decision Practices | -0.984 | checks_consensus | +0.726 |
| | | proposes_process | -1.854 |

Model 1 (session, category) vs. Models 2/3 (person) direction by theme:

| Theme | Model 1 (session) | Model 2/3 (person) |
|---|---|---|
| Integration Practices / identifies_common_ground | + | + |
| Individual Framing | + | + (n.s.) |
| Epistemic Bridging / reframes_cross_disciplinarily | + | + |
| Idea Management / combines_ideas, extends_existing_idea | + | + |
| Relational Climate / expresses_appreciation | + | − |
| Participation Dynamics / summarizes_for_group | + | − |

Significant at person level (Models 2/3) for `outcome_joined_team` only,
not `outcome_joined_funded_team`: `identifies_common_ground`,
`checks_consensus`, `combines_ideas`. Significant for both outcomes:
`extends_existing_idea`, `proposes_process`.

## Guest / Fellow / Facilitator (data only)

Attendee-role data exists for 2 of 8 conferences (2021ABI, 2021NES; n=164
combined). Team-joining rate by role (`global_person_id` matching, one
identity-collision false positive corrected — "Brad Smith" facilitator vs.
"Barbara Smith" team member, incorrectly sharing a global_person_id):

| Role | n | joined a team | rate |
|---|---|---|---|
| Fellow | 116 | 96 | 82.8% |
| Guest | 28 | 0 | 0% |
| Facilitator | 20 | 0 | 0% |

97 unique team members/authors across these 2 conferences: 95 matched
directly to a Fellow, 2 unmatched (name-formatting variants, not a
different role), 0 Guests, 0 Facilitators.

Guest institutions (from attendee list "Institution" field, 2021ABI
sample): Chan Zuckerberg Initiative (multiple), Allen Institute, Beckman
Foundation, Walder Foundation, Cottrell Foundation ("Area of Expertise"
field for one entry: "we are a meeting sponsor").

### Guest behavioral comparison

8 of 28 Guests resolve to a `global_person_id` (the other 20 never spoke in
a coded session or authored a proposal). Mean raw subcode counts per
person, known-role subset:

| | Fellow (n=98) | Facilitator (n=17) | Guest (n=8) |
|---|---|---|---|
| Utterance count | 31.2 | 93.9 | 2.0 |
| Sessions attended | 2.96 | 3.24 | 1.50 |
| expresses_appreciation | 0.59 | 3.24 | 0.12 |
| summarizes_for_group | 0.06 | 1.12 | 0.00 |
| proposes_process | 1.02 | 16.24 | 0.00 |
| extends_existing_idea | 2.95 | 3.53 | 0.00 |
| identifies_common_ground | 0.20 | 0.41 | 0.00 |
| combines_ideas | 0.24 | 0.18 | 0.00 |
| checks_consensus | 0.12 | 0.59 | 0.00 |

n=8 — no statistical test run, descriptive only.

### Model exclusion: Facilitators + known Guests vs. Facilitators only

8-feature model (`n_sessions_attended` + the 7 key subcodes from Models
2/3), refit on two samples — excludes Facilitators only (n=579, matches the
existing non-facilitator sensitivity check) vs. excludes Facilitators and
the 8 known Guests (n=571). Not the same as the full 61-feature Model 3
table; coefficients here aren't directly comparable to that table, only to
each other.

| Feature | coef, excl. facilitators | coef, excl. facilitators + guests |
|---|---|---|
| proposes_process | -0.336 (p<0.001) | -0.337 (p<0.001) |
| expresses_appreciation | -0.166 (p=0.163) | -0.163 (p=0.172) |
| summarizes_for_group | -0.321 (p=0.239) | -0.316 (p=0.240) |
| identifies_common_ground | +0.737 (p=0.012) | +0.725 (p=0.013) |
| checks_consensus | +0.662 (p=0.016) | +0.658 (p=0.016) |
| combines_ideas | +0.652 (p=0.019) | +0.647 (p=0.020) |
| extends_existing_idea | +0.299 (p<0.001) | +0.292 (p<0.001) |
| reframes_cross_disciplinarily | +0.712 (p=0.015) | +0.703 (p=0.016) |

Only 2 of 8 conferences have Guest labels; scoped to that coverage.

## Change log

- 2026-07-06/07: Session-level name matching rebuilt to align with
  person-level identity resolution; 6 previously-invisible sessions
  recovered (156 -> 162); chunk_registry rebuilt; stages 4-6 rerun.
  Alias-path bug fixed in `evey_new_analyses.py` and
  `7-person_level_modeling.ipynb`. Pushed to `update-session-lvl-name-matching`.
- 2026-07-08: Models 1-3 documented. Guest/Facilitator team-joining rate
  computed (0%); one identity-collision false positive caught and
  corrected.
- 2026-07-08 (later): Model 4 built (session-level subcodes). Guest
  behavioral comparison and exclusion rerun completed.
