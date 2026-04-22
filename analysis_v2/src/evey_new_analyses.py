"""
evey_new_analyses.py
Generates all new figures and data for Evey's slide feedback:
  1. feature_correlation_heatmap.png  – behavioral features × [n_sessions, is_facilitator]
  2. person_codename_features.csv     – person-level code_name aggregate features
  3. coef_chart_codename_model.png    – coefficient chart for high-level category model
  4. coef_chart_subcode_model.png     – re-generated detailed subcode chart (with is_facilitator shown)
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

try:
    import statsmodels.api as sm
except ImportError:
    sm = None

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ─── paths ────────────────────────────────────────────────────────────────────
BASE        = Path("/Users/maxchalekson/Projects/NICO-Research/gemini_data_analysis")
FEATURE_DIR = BASE / "analysis_v2/data/person-aggregation-features"
FIG_DIR     = BASE / "analysis_v2/figures/person-aggregation-features"
OUTPUTS_DIR = BASE / "outputs"
BY_CONF     = BASE / "finalized_matching_csvs/global_participant_identity_by_conference.csv"
ALIAS_PATH  = BASE / "participant_alias_mapping.csv"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ─── color palette ────────────────────────────────────────────────────────────
NAVY     = "#1E2761"
TEAL     = "#028090"
ORANGE   = "#E07B4A"
RED      = "#E53E3E"
GRAY     = "#AAAAAA"
LIGHT_BG = "#F4F7FB"

# ─── helpers ──────────────────────────────────────────────────────────────────
def norm_name(text: object) -> str:
    s = "" if text is None else str(text)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def p_to_stars(p: float) -> str:
    if pd.isna(p): return ""
    if p < 0.001:  return "***"
    if p < 0.01:   return "**"
    if p < 0.05:   return "*"
    return ""

# ─── alias + identity lookup ──────────────────────────────────────────────────
alias_map: dict[str, str] = {}
if ALIAS_PATH.exists():
    for _, r in pd.read_csv(ALIAS_PATH).iterrows():
        a, c = norm_name(r.get("alias_name")), norm_name(r.get("canonical_name"))
        if a and c:
            alias_map[a] = c

def resolve_alias(name: str) -> str:
    cur, seen = name, set()
    while cur in alias_map and cur not in seen:
        seen.add(cur); cur = alias_map[cur]
    return cur

map_df = pd.read_csv(BY_CONF)
map_df["conference"] = map_df["conference"].astype(str).str.strip()
map_df["normalized_name"] = map_df["normalized_name"].astype(str).map(norm_name).map(resolve_alias)
lookup = {(r["conference"], r["normalized_name"]): r["global_person_id"]
          for _, r in map_df.iterrows()}

# ─── canonical code_names ─────────────────────────────────────────────────────
CANONICAL = {
    "broader significance",
    "complementarity articulation",
    "coordination and decision practices",
    "epistemic bridging",
    "evaluation practices",
    "future oriented language",
    "idea management",
    "idea novelty signal",
    "idea ownership and attribution",
    "information seeking",
    "integration practices",
    "knowledge sharing",
    "participation dynamics",
    "pronoun framing",
    "relational climate",
    "role anticipation",
}

# Display labels (title-case, short)
CODENAME_LABELS = {
    "broader_significance":              "Broader Significance",
    "complementarity_articulation":      "Complementarity Articulation",
    "coordination_and_decision_practices": "Coordination & Decision",
    "epistemic_bridging":                "Epistemic Bridging",
    "evaluation_practices":              "Evaluation Practices",
    "future_oriented_language":          "Future-Oriented Language",
    "idea_management":                   "Idea Management",
    "idea_novelty_signal":               "Idea Novelty Signal",
    "idea_ownership_and_attribution":    "Idea Ownership & Attribution",
    "information_seeking":               "Information Seeking",
    "integration_practices":             "Integration Practices",
    "knowledge_sharing":                 "Knowledge Sharing",
    "participation_dynamics":            "Participation Dynamics",
    "pronoun_framing":                   "Pronoun Framing",
    "relational_climate":                "Relational Climate",
    "role_anticipation":                 "Role Anticipation",
}

def normalize_codename(cn: object) -> str | None:
    if not cn: return None
    s = str(cn).lower().strip()
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s if s in CANONICAL else None

def cn_to_col(cn: str) -> str:
    return "codename__" + re.sub(r"[\s\-]+", "_", cn)

# ════════════════════════════════════════════════════════════════════════════════
# STEP 1 – scan outputs and build person × code_name counts
# ════════════════════════════════════════════════════════════════════════════════
print("Step 1: scanning output JSONs for code_name annotations …")
rows: list[dict] = []
for conf_dir in sorted(OUTPUTS_DIR.glob("*")):
    if not conf_dir.is_dir():
        continue
    conference = conf_dir.name
    for jp in conf_dir.rglob("*.json"):
        try:
            payload = json.loads(jp.read_text())
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        for utt in payload.get("utterance_annotations", []):
            sp = resolve_alias(norm_name(utt.get("speaker")))
            pid = lookup.get((conference, sp))
            if not pid:
                continue
            for c in (utt.get("codes") or []):
                cn = normalize_codename(c.get("code_name"))
                if cn:
                    rows.append({"conference": conference, "global_person_id": pid, "code_name": cn})

print(f"  {len(rows):,} code_name annotations across {len(set(r['conference'] for r in rows))} conferences")
raw_df = pd.DataFrame(rows)

counts = raw_df.groupby(["conference", "global_person_id", "code_name"]).size().reset_index(name="count")
pivot  = counts.pivot_table(
    index=["conference", "global_person_id"],
    columns="code_name", values="count", fill_value=0
).reset_index()
pivot.columns.name = None
pivot = pivot.rename(columns={cn: cn_to_col(cn) for cn in CANONICAL if cn in pivot.columns})

# ─── load base features ───────────────────────────────────────────────────────
base = pd.read_csv(FEATURE_DIR / "person_features_detailed_subcodes.csv").copy()
if "speaking_minutes_total" not in base.columns and "speaking_seconds_total" in base.columns:
    base["speaking_minutes_total"] = base["speaking_seconds_total"] / 60.0

# ─── load facilitator flag ────────────────────────────────────────────────────
fac_path = FEATURE_DIR / "person_df_with_fac.csv"
if fac_path.exists():
    fac = pd.read_csv(fac_path)[["conference", "global_person_id", "is_facilitator"]]
    base = base.merge(fac, on=["conference", "global_person_id"], how="left")
    base["is_facilitator"] = base["is_facilitator"].fillna(0).astype(int)
else:
    base["is_facilitator"] = 0

# ─── merge code_name features into base ──────────────────────────────────────
merged = base.merge(pivot, on=["conference", "global_person_id"], how="left")
codename_cols = [cn_to_col(cn) for cn in CANONICAL if cn_to_col(cn) in merged.columns]
for c in codename_cols:
    merged[c] = merged[c].fillna(0)

merged.to_csv(FEATURE_DIR / "person_codename_features.csv", index=False)
print(f"  Saved person_codename_features.csv  ({len(merged)} rows, {len(codename_cols)} code_name features)")

# convenience
subcode_cols     = [c for c in merged.columns if c.startswith("subcode__") and c != "subcode__none"]
subcode_eligible = [c for c in subcode_cols if merged[c].sum() >= 10]
controls         = [c for c in ["n_sessions_attended", "speaking_minutes_total", "is_facilitator"]
                    if c in merged.columns]
TARGET = "outcome_joined_team"

# ════════════════════════════════════════════════════════════════════════════════
# STEP 2 – heatmap: behavioral features × [n_sessions_attended, is_facilitator]
# ════════════════════════════════════════════════════════════════════════════════
print("\nStep 2: building correlation heatmap …")

corr_targets = [c for c in ["n_sessions_attended", "is_facilitator"] if c in merged.columns]
corr_rows: list[dict] = []
for feat in subcode_eligible:
    row: dict = {"feature": feat.replace("subcode__", "")}
    for ct in corr_targets:
        x = merged[feat].values
        y = merged[ct].values
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 3:
            row[ct] = np.nan
        else:
            r, _ = stats.pearsonr(x[mask], y[mask])
            row[ct] = r
    corr_rows.append(row)

corr_df = pd.DataFrame(corr_rows).set_index("feature")
corr_df = corr_df.sort_values("n_sessions_attended", ascending=False)

# rename display labels
col_labels = {"n_sessions_attended": "Session\nAttendance", "is_facilitator": "Facilitator\nStatus"}
corr_display = corr_df.rename(columns=col_labels)

fig, ax = plt.subplots(figsize=(5, 18))
fig.patch.set_facecolor(LIGHT_BG)
ax.set_facecolor(LIGHT_BG)

sns.heatmap(
    corr_display,
    ax=ax,
    cmap="RdBu_r",
    center=0,
    vmin=-0.6, vmax=0.6,
    annot=True, fmt=".2f",
    annot_kws={"size": 7, "weight": "bold"},
    linewidths=0.3,
    linecolor="#dddddd",
    cbar_kws={"shrink": 0.4, "label": "Pearson r"},
)
ax.set_title(
    "Behavioral Features\nCorrelation with Control Variables",
    fontsize=11, fontweight="bold", color=NAVY, pad=10
)
ax.set_xlabel("Control Variable", fontsize=9, color=NAVY)
ax.set_ylabel("")
ax.tick_params(axis="y", labelsize=7)
ax.tick_params(axis="x", labelsize=9, rotation=0)

fig.tight_layout()
heatmap_path = FIG_DIR / "feature_correlation_heatmap.png"
fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {heatmap_path}")

# ════════════════════════════════════════════════════════════════════════════════
# STEP 3 – LOCO + statsmodels for code_name model
# ════════════════════════════════════════════════════════════════════════════════
print("\nStep 3: running code_name-level logistic regression …")

def run_loco(data: pd.DataFrame, feature_cols: list[str], target: str):
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, class_weight="balanced")),
    ])
    y_true_all, y_score_all, y_pred_all = [], [], []
    for holdout in sorted(data["conference"].dropna().unique()):
        tr = data[data["conference"] != holdout]
        te = data[data["conference"] == holdout]
        if tr.empty or te.empty or tr[target].nunique() < 2:
            continue
        pipe.fit(tr[feature_cols], tr[target].astype(int))
        p = pipe.predict_proba(te[feature_cols])[:, 1]
        y_true_all.extend(te[target].astype(int).tolist())
        y_score_all.extend(p.tolist())
        y_pred_all.extend((p >= 0.5).astype(int).tolist())
    return (np.array(y_true_all, int), np.array(y_score_all, float),
            np.array(y_pred_all, int))

def fit_sm(data: pd.DataFrame, feature_cols: list[str], target: str) -> pd.DataFrame:
    if sm is None:
        return pd.DataFrame(columns=["feature", "coef", "pvalue", "stars"])
    X = data[feature_cols].copy().fillna(data[feature_cols].median(numeric_only=True))
    std = X.std(ddof=0).replace(0, 1.0)
    Xz = sm.add_constant((X - X.mean()) / std, has_constant="add")
    y = data[target].astype(int).values
    try:
        fit = sm.Logit(y, Xz).fit(disp=0, maxiter=400)
    except Exception as e:
        print(f"  statsmodels failed: {e}")
        return pd.DataFrame(columns=["feature", "coef", "pvalue", "stars"])
    params = fit.params.drop("const", errors="ignore")
    pvals  = fit.pvalues.drop("const", errors="ignore")
    out = pd.DataFrame({"feature": params.index, "coef": params.values,
                        "pvalue": pvals.reindex(params.index).values})
    out["stars"] = out["pvalue"].map(p_to_stars)
    return out

# ── code_name model
cn_model_features = controls + codename_cols
cn_yt, cn_ys, cn_yp = run_loco(merged, cn_model_features, TARGET)
cn_auc   = roc_auc_score(cn_yt, cn_ys)   if len(np.unique(cn_yt)) >= 2 else np.nan
cn_auprc = average_precision_score(cn_yt, cn_ys) if len(np.unique(cn_yt)) >= 2 else np.nan
cn_acc   = accuracy_score(cn_yt, cn_yp)
cn_f1    = f1_score(cn_yt, cn_yp, zero_division=0)
print(f"  Code_name model: AUC={cn_auc:.3f}  AUPRC={cn_auprc:.3f}  Acc={cn_acc:.3f}  F1={cn_f1:.3f}")

cn_coef_df = fit_sm(merged, cn_model_features, TARGET)
cn_coef_df["feature_label"] = cn_coef_df["feature"].map(
    lambda f: CODENAME_LABELS.get(f.replace("codename__", ""), f.replace("codename__", "").replace("_", " ").title())
)
cn_coef_df.to_csv(FEATURE_DIR / "person_codename_model_coefficients.csv", index=False)

# ── subcode model (for comparison row)
sc_model_features = controls + subcode_eligible
sc_yt, sc_ys, sc_yp = run_loco(merged, sc_model_features, TARGET)
sc_auc   = roc_auc_score(sc_yt, sc_ys)   if len(np.unique(sc_yt)) >= 2 else np.nan
sc_auprc = average_precision_score(sc_yt, sc_ys) if len(np.unique(sc_yt)) >= 2 else np.nan
sc_acc   = accuracy_score(sc_yt, sc_yp)
sc_f1    = f1_score(sc_yt, sc_yp, zero_division=0)
print(f"  Subcode model:   AUC={sc_auc:.3f}  AUPRC={sc_auprc:.3f}  Acc={sc_acc:.3f}  F1={sc_f1:.3f}")

# Use pre-computed coefficients CSV (statsmodels may fail due to is_facilitator separation)
_sc_coef_path = FEATURE_DIR / "person_model_coefficients_outcome_joined_team.csv"
if _sc_coef_path.exists():
    sc_coef_df = pd.read_csv(_sc_coef_path)
    sc_coef_df["stars"] = sc_coef_df["pvalue"].map(p_to_stars)
else:
    sc_coef_df = fit_sm(merged, sc_model_features, TARGET)
sc_coef_df["feature_label"] = sc_coef_df["feature"].map(
    lambda f: f.replace("subcode__", "").replace("_", " ").title()
)

# ════════════════════════════════════════════════════════════════════════════════
# STEP 4 – coefficient chart: code_name model
# ════════════════════════════════════════════════════════════════════════════════
print("\nStep 4: generating code_name coefficient chart …")

cn_sig = cn_coef_df[~cn_coef_df["feature"].isin(["n_sessions_attended", "speaking_minutes_total",
                                                   "is_facilitator"])].copy()
cn_sig = cn_sig.sort_values("coef")

fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor(LIGHT_BG)
ax.set_facecolor(LIGHT_BG)

colors = [TEAL if c >= 0 else ORANGE for c in cn_sig["coef"]]
stars_flag = cn_sig["pvalue"] < 0.05
colors = [TEAL if (c >= 0 and s) else ORANGE if (c < 0 and s) else GRAY
          for c, s in zip(cn_sig["coef"], stars_flag)]

bars = ax.barh(cn_sig["feature_label"], cn_sig["coef"], color=colors, height=0.65, edgecolor="white", linewidth=0.5)
ax.axvline(0, color=NAVY, linewidth=1.2, zorder=3)

for yi, (_, row) in enumerate(cn_sig.reset_index(drop=True).iterrows()):
    if row["stars"]:
        x = row["coef"] + (0.02 if row["coef"] >= 0 else -0.02)
        ha = "left" if row["coef"] >= 0 else "right"
        ax.text(x, yi, row["stars"], va="center", ha=ha, fontsize=11, color=NAVY, fontweight="bold")

ax.set_xlabel("Standardized Logistic Coefficient", fontsize=10, color=NAVY)
ax.set_title(
    f"What High-Level Behavior Categories Predict Joining a Team?\n"
    f"LOCO-CV: AUC={cn_auc:.3f}  AUPRC={cn_auprc:.3f}  Acc={cn_acc:.3f}  F1={cn_f1:.3f}  (n={len(merged)})",
    fontsize=11, fontweight="bold", color=NAVY, pad=10
)

# legend
teal_patch  = mpatches.Patch(color=TEAL,   label="Positive predictor (p<.05)")
orange_patch = mpatches.Patch(color=ORANGE, label="Negative predictor (p<.05)")
gray_patch  = mpatches.Patch(color=GRAY,   label="Not significant (p≥.05)")
ax.legend(handles=[teal_patch, orange_patch, gray_patch], loc="lower right",
          frameon=True, framealpha=0.9, fontsize=8)

ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="y", labelsize=9)
ax.tick_params(axis="x", labelsize=9)

fig.tight_layout()
cn_chart_path = FIG_DIR / "coef_chart_codename_model.png"
fig.savefig(cn_chart_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {cn_chart_path}")

# ════════════════════════════════════════════════════════════════════════════════
# STEP 5 – coefficient chart: subcode model (re-gen with is_facilitator shown)
# ════════════════════════════════════════════════════════════════════════════════
print("\nStep 5: regenerating subcode coefficient chart with is_facilitator …")

sig_sc = sc_coef_df[sc_coef_df["pvalue"] < 0.05].copy()
sig_sc = sig_sc[~sig_sc["feature"].isin(["n_sessions_attended", "speaking_minutes_total", "is_facilitator"])]

# always include is_facilitator row
fac_row = sc_coef_df[sc_coef_df["feature"] == "is_facilitator"].copy()
if not fac_row.empty:
    fac_row["feature_label"] = "is_facilitator"
    plot_df = pd.concat([sig_sc, fac_row]).drop_duplicates("feature").sort_values("coef")
else:
    plot_df = sig_sc.sort_values("coef")

# Separate is_facilitator from the rest for a clean chart
plot_df_no_fac = plot_df[plot_df["feature"] != "is_facilitator"].copy()
fac_coef = fac_row["coef"].values[0] if not fac_row.empty else -10.4

fig, ax = plt.subplots(figsize=(10, max(6, len(plot_df_no_fac) * 0.65 + 1.5)))
fig.patch.set_facecolor(LIGHT_BG)
ax.set_facecolor(LIGHT_BG)

for i, (_, row) in enumerate(plot_df_no_fac.reset_index(drop=True).iterrows()):
    color = TEAL if row["coef"] >= 0 else ORANGE
    ax.barh(row["feature_label"], row["coef"], color=color, height=0.65,
            edgecolor="white", linewidth=0.5)
    x = row["coef"] + (0.02 if row["coef"] >= 0 else -0.02)
    ha = "left" if row["coef"] >= 0 else "right"
    if row["stars"]:
        ax.text(x, i, row["stars"], va="center", ha=ha, fontsize=10, color=NAVY, fontweight="bold")

ax.axvline(0, color=NAVY, linewidth=1.2, zorder=3)

# is_facilitator note box at the bottom
n_bars = len(plot_df_no_fac)
ax.text(
    0.02, -0.09,
    "▼  is_facilitator: coef = −10.4, p ≈ 1.0  (off scale)\n"
    "   All 60 facilitators outcome = 0 → perfect separation → coefficient unreliable",
    transform=ax.transAxes, fontsize=8, color=ORANGE, va="top",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=ORANGE, alpha=0.9),
)

ax.set_xlabel("Standardized Logistic Coefficient", fontsize=10, color=NAVY)
ax.set_title(
    f"What Behaviors Predict Joining a Team? Facilitators Included (n={len(merged)})\n"
    f"LOCO-CV: AUC={sc_auc:.3f}  AUPRC={sc_auprc:.3f}  Acc={sc_acc:.3f}  F1={sc_f1:.3f}",
    fontsize=11, fontweight="bold", color=NAVY, pad=10
)

teal_patch   = mpatches.Patch(color=TEAL,   label="Positive predictor (p<.05)")
orange_patch = mpatches.Patch(color=ORANGE, label="Negative predictor (p<.05)")
ax.legend(handles=[teal_patch, orange_patch], loc="lower right",
          frameon=True, framealpha=0.9, fontsize=8)

ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="y", labelsize=9)

fig.tight_layout()
fig.subplots_adjust(bottom=0.18)
sc_chart_path = FIG_DIR / "coef_chart_with_facilitators.png"
fig.savefig(sc_chart_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {sc_chart_path}")

# ════════════════════════════════════════════════════════════════════════════════
# STEP 6 – comparison table (code_name vs subcode model)
# ════════════════════════════════════════════════════════════════════════════════
comp = pd.DataFrame([
    {"model": "High-level (16 code_name categories)", "n_features": len(cn_model_features),
     "AUC": cn_auc, "AUPRC": cn_auprc, "Accuracy": cn_acc, "F1": cn_f1},
    {"model": "Detailed (58 subcodes)",               "n_features": len(sc_model_features),
     "AUC": sc_auc, "AUPRC": sc_auprc, "Accuracy": sc_acc, "F1": sc_f1},
])
comp.to_csv(FEATURE_DIR / "person_model_highlevel_vs_detailed.csv", index=False)
print("\nModel comparison:")
print(comp.to_string(index=False))

# ════════════════════════════════════════════════════════════════════════════════
# STEP 7 – code_name model: Outcome 2 (funded team), 2-stage conditional
#           Sample: non-facilitators who joined a team  (outcome_joined_team == 1)
#           Target: outcome_joined_funded_team
# ════════════════════════════════════════════════════════════════════════════════
print("\nStep 7: code_name model – Outcome 2 (funded team, conditional on joining) …")

TARGET2     = "outcome_joined_funded_team"
controls_nf = [c for c in ["n_sessions_attended", "speaking_minutes_total"] if c in merged.columns]

# 2-stage conditional sample: non-fac team joiners only
cond_mask = (merged["is_facilitator"] == 0) & (merged[TARGET2].notna())
if "outcome_joined_team" in merged.columns:
    cond_mask = cond_mask & (merged["outcome_joined_team"] == 1)
cond_df = merged[cond_mask].copy()
print(f"  Conditional sample: n={len(cond_df)}  (non-fac team joiners)")
print(f"  Positive rate: {cond_df[TARGET2].mean():.3f}  "
      f"({int(cond_df[TARGET2].sum())} funded / {len(cond_df)} joiners)")

cn2_features = controls_nf + codename_cols
cn2_yt, cn2_ys, cn2_yp = run_loco(cond_df, cn2_features, TARGET2)

if len(np.unique(cn2_yt)) >= 2:
    cn2_auc   = roc_auc_score(cn2_yt, cn2_ys)
    cn2_auprc = average_precision_score(cn2_yt, cn2_ys)
    cn2_acc   = accuracy_score(cn2_yt, cn2_yp)
    cn2_f1    = f1_score(cn2_yt, cn2_yp, zero_division=0)
    print(f"  Code_name Outcome 2: AUC={cn2_auc:.3f}  AUPRC={cn2_auprc:.3f}  "
          f"Acc={cn2_acc:.3f}  F1={cn2_f1:.3f}")
else:
    cn2_auc = cn2_auprc = cn2_acc = cn2_f1 = float("nan")
    print("  Warning: not enough variation in outcome to compute metrics")

cn2_coef_df = fit_sm(cond_df, cn2_features, TARGET2)
cn2_coef_df["feature_label"] = cn2_coef_df["feature"].map(
    lambda f: CODENAME_LABELS.get(f.replace("codename__", ""),
                                  f.replace("codename__", "").replace("_", " ").title())
)
cn2_coef_df.to_csv(FEATURE_DIR / "person_codename_model_coefficients_outcome2.csv", index=False)
print(f"  Saved person_codename_model_coefficients_outcome2.csv")

# ── chart ─────────────────────────────────────────────────────────────────────
cn2_plot = cn2_coef_df[~cn2_coef_df["feature"].isin(
    ["n_sessions_attended", "speaking_minutes_total", "is_facilitator"])].copy()
cn2_plot = cn2_plot.sort_values("coef")

colors2 = [TEAL  if (c >= 0 and p < 0.05) else
           ORANGE if (c <  0 and p < 0.05) else GRAY
           for c, p in zip(cn2_plot["coef"], cn2_plot["pvalue"])]

fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor(LIGHT_BG)
ax.set_facecolor(LIGHT_BG)

ax.barh(cn2_plot["feature_label"], cn2_plot["coef"],
        color=colors2, height=0.65, edgecolor="white", linewidth=0.5)
ax.axvline(0, color=NAVY, linewidth=1.2, zorder=3)

for yi, (_, row) in enumerate(cn2_plot.reset_index(drop=True).iterrows()):
    if row["stars"]:
        x  = row["coef"] + (0.02 if row["coef"] >= 0 else -0.02)
        ha = "left" if row["coef"] >= 0 else "right"
        ax.text(x, yi, row["stars"], va="center", ha=ha,
                fontsize=11, color=NAVY, fontweight="bold")

ax.set_xlabel("Standardized Logistic Coefficient", fontsize=10, color=NAVY)
n_cond = len(cond_df)
ax.set_title(
    f"What High-Level Behavior Categories Predict Joining a Funded Team?\n"
    f"LOCO-CV: AUC={cn2_auc:.3f}  AUPRC={cn2_auprc:.3f}  "
    f"Acc={cn2_acc:.3f}  F1={cn2_f1:.3f}  (n={n_cond}, team joiners only)",
    fontsize=11, fontweight="bold", color=NAVY, pad=10,
)

teal_patch2   = mpatches.Patch(color=TEAL,   label="Positive predictor (p<.05)")
orange_patch2 = mpatches.Patch(color=ORANGE, label="Negative predictor (p<.05)")
gray_patch2   = mpatches.Patch(color=GRAY,   label="Not significant (p≥.05)")
ax.legend(handles=[teal_patch2, orange_patch2, gray_patch2],
          loc="lower right", frameon=True, framealpha=0.9, fontsize=8)

ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="y", labelsize=9)
ax.tick_params(axis="x", labelsize=9)

fig.tight_layout()
cn2_chart_path = FIG_DIR / "coef_chart_codename_outcome2.png"
fig.savefig(cn2_chart_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {cn2_chart_path}")

print("\n✓ All done.")
print(f"  Heatmap:               {heatmap_path}")
print(f"  Code_name coef chart:  {cn_chart_path}")
print(f"  Subcode coef chart:    {sc_chart_path}")
print(f"  Code_name Outcome 2:   {cn2_chart_path}")
print(f"  Code_name features:    {FEATURE_DIR}/person_codename_features.csv")
