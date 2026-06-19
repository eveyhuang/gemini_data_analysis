"""
chunk_summary_analysis.py
Extracts all chunk_summary fields, organizes into buckets, and generates:
  1. coef_chart_chunk_features.png  – model results by bucket (Outcome 1)
  2. chunk_prevalence_overview.png  – descriptive overview of all fields
"""

from __future__ import annotations
import json, re, unicodedata
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
try:
    import statsmodels.api as sm
except ImportError:
    sm = None

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ── paths ──────────────────────────────────────────────────────────────────────
BASE        = Path("/Users/maxchalekson/Projects/NICO-Research/gemini_data_analysis")
FEATURE_DIR = BASE / "analysis_v2/data/person-aggregation-features"
FIG_DIR     = BASE / "analysis_v2/figures/person-aggregation-features"
OUTPUTS_DIR = BASE / "outputs"
BY_CONF     = BASE / "analysis_v2/notebooks/finalized_matching_csvs/global_participant_identity_by_conference.csv"
ALIAS_PATH  = BASE / "analysis_v2/notebooks/participant_alias_mapping.csv"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── colors ─────────────────────────────────────────────────────────────────────
NAVY     = "#1E2761"
TEAL     = "#028090"
ORANGE   = "#E07B4A"
GRAY     = "#AAAAAA"
LIGHT_BG = "#F4F7FB"

BUCKET_COLORS = {
    "Idea Development":       "#2196F3",
    "Collaboration Signals":  "#4CAF50",
    "Commitment & Funding":   "#FF9800",
    "Social & Relational":    "#9C27B0",
    "Participation Dynamics": "#F44336",
}

# ── helpers ────────────────────────────────────────────────────────────────────
def norm_name(text):
    s = "" if text is None else str(text)
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def p_to_stars(p):
    if pd.isna(p): return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""

# ── alias + lookup ─────────────────────────────────────────────────────────────
alias_map: dict[str,str] = {}
if ALIAS_PATH.exists():
    for _, r in pd.read_csv(ALIAS_PATH).iterrows():
        a, c = norm_name(r.get("alias_name")), norm_name(r.get("canonical_name"))
        if a and c: alias_map[a] = c

def resolve_alias(name):
    cur, seen = name, set()
    while cur in alias_map and cur not in seen:
        seen.add(cur); cur = alias_map[cur]
    return cur

map_df = pd.read_csv(BY_CONF)
map_df["conference"] = map_df["conference"].astype(str).str.strip()
map_df["normalized_name"] = map_df["normalized_name"].astype(str).map(norm_name).map(resolve_alias)
lookup = {(r["conference"], r["normalized_name"]): r["global_person_id"]
          for _, r in map_df.iterrows()}

# ── feature buckets ────────────────────────────────────────────────────────────
# (field_name, feature_col, display_label, bucket, feature_type)
# feature_type: "binary" (Yes/No), "numeric" (1-5), "rate_of_value" (categorical == target_value)
FEATURES = [
    # Idea Development
    ("idea_trajectory",            "idea_convergent",          "Idea Trajectory: Convergent",    "Idea Development",       "rate_of_value", "convergent"),
    ("problem_specificity_level",  "problem_specificity",      "Problem Specificity (avg)",       "Idea Development",       "numeric",       None),
    ("decision_crystallization_level","decision_crystallization","Decision Crystallization (avg)", "Idea Development",       "numeric",       None),
    ("ambition_level",             "ambition_novel",            "Ambition: Novel+",               "Idea Development",       "rate_of_value", ("novel_combination","novel_application","paradigm_challenging")),
    # Collaboration Signals
    ("cross_disciplinary_bridging",       "cross_disc_bridging",   "Cross-Disc. Bridging in Session","Collaboration Signals","binary",        None),
    ("explicit_complementarity_recognition","complementarity_recog","Complementarity Recognized",    "Collaboration Signals","binary",        None),
    ("skill_gap_identification",          "skill_gap",             "Skill Gap Identified",           "Collaboration Signals","binary",        None),
    ("shared_vision_indicator",           "shared_vision",         "Shared Vision Signal",           "Collaboration Signals","binary",        None),
    # Commitment & Funding
    ("explicit_commitment_signal",        "commitment_signal",     "Commitment Signal",              "Commitment & Funding", "binary",        None),
    ("risk_acknowledgment_with_enthusiasm","risk_enthusiasm",      "Risk + Enthusiasm",              "Commitment & Funding", "binary",        None),
    ("funding_awareness_signal",          "funding_awareness",     "Funding Awareness",              "Commitment & Funding", "binary",        None),
    # Social & Relational
    ("pronoun_shift_flag",                "pronoun_shift",         "Pronoun Shift (we/us)",          "Social & Relational",  "binary",        None),
    ("personal_disclosure",               "personal_disclosure",   "Personal Disclosure",            "Social & Relational",  "binary",        None),
    ("prior_relationship_signal",         "prior_relationship",    "Prior Relationship Signal",      "Social & Relational",  "binary",        None),
    ("laughter_quality",                  "shared_humor",          "Shared Humor / Laughter",        "Social & Relational",  "rate_of_value", ("shared_humor","appreciative","social_lubricant")),
    # Participation Dynamics
    ("dominant_speaker_flag",             "dominant_speaker",      "Dominant Speaker",               "Participation Dynamics","binary",       None),
    ("screenshare_active",                "screenshare",           "Screenshare Active",             "Participation Dynamics","binary",       None),
    ("collective_engagement_level",       "engagement_level",      "Collective Engagement (avg)",    "Participation Dynamics","numeric",      None),
    ("meeting_structure_quality",         "structured_meeting",    "Meeting: Structured",            "Participation Dynamics","rate_of_value","structured"),
]

FEAT_COLS   = [f[1] for f in FEATURES]
FEAT_LABELS = {f[1]: f[2] for f in FEATURES}
FEAT_BUCKET = {f[1]: f[3] for f in FEATURES}

# ── scan JSONs ─────────────────────────────────────────────────────────────────
print("Scanning chunk_summary fields …")
chunk_rows = []   # one row per (person, chunk)
chunk_global = [] # chunk-level stats (for prevalence figure, no person needed)

for conf_dir in sorted(OUTPUTS_DIR.glob("*")):
    if not conf_dir.is_dir(): continue
    conference = conf_dir.name
    for jp in conf_dir.rglob("*.json"):
        try:
            payload = json.loads(jp.read_text())
        except: continue
        if not isinstance(payload, dict): continue
        cs = payload.get("chunk_summary", {})
        if not cs: continue

        # who spoke in this chunk?
        speakers_raw = cs.get("speaking_time_seconds", {})
        pids = []
        for raw_sp in speakers_raw:
            sp  = resolve_alias(norm_name(raw_sp))
            pid = lookup.get((conference, sp))
            if pid: pids.append((pid, conference))

        # compute feature values for this chunk
        feat_vals = {}
        for (field, col, label, bucket, ftype, target) in FEATURES:
            raw = cs.get(field, None)
            if ftype == "binary":
                feat_vals[col] = 1 if str(raw).strip().lower() in ("yes","true","1") else 0
            elif ftype == "numeric":
                try:    feat_vals[col] = float(raw)
                except: feat_vals[col] = np.nan
            elif ftype == "rate_of_value":
                v = str(raw).strip().lower()
                if isinstance(target, tuple):
                    feat_vals[col] = 1 if v in target else 0
                else:
                    feat_vals[col] = 1 if v == target else 0

        chunk_global.append(feat_vals)

        for (pid, conf) in pids:
            row = {"conference": conf, "global_person_id": pid}
            row.update(feat_vals)
            chunk_rows.append(row)

print(f"  {len(chunk_global)} chunks, {len(chunk_rows)} person-chunk rows")

# ── chunk-level prevalence (for overview figure) ───────────────────────────────
chunk_df = pd.DataFrame(chunk_global)

# ── aggregate to person level ─────────────────────────────────────────────────
person_chunk = pd.DataFrame(chunk_rows)
person_agg = person_chunk.groupby(["conference", "global_person_id"])[FEAT_COLS].mean().reset_index()

# merge with base features
base = pd.read_csv(FEATURE_DIR / "person_codename_features.csv")
merged = base.merge(person_agg, on=["conference","global_person_id"], how="left")
for col in FEAT_COLS:
    merged[col] = merged[col].fillna(0)

print(f"  Person-level merged: {len(merged)} rows")
merged.to_csv(FEATURE_DIR / "person_chunk_features.csv", index=False)

# ── LOCO helpers ───────────────────────────────────────────────────────────────
def run_loco(data, feature_cols, target):
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, class_weight="balanced")),
    ])
    yt, ys, yp = [], [], []
    for holdout in sorted(data["conference"].dropna().unique()):
        tr = data[data["conference"] != holdout]
        te = data[data["conference"] == holdout]
        if tr.empty or te.empty or tr[target].nunique() < 2: continue
        pipe.fit(tr[feature_cols], tr[target].astype(int))
        p = pipe.predict_proba(te[feature_cols])[:,1]
        yt.extend(te[target].astype(int).tolist())
        ys.extend(p.tolist())
        yp.extend((p>=0.5).astype(int).tolist())
    return np.array(yt,int), np.array(ys,float), np.array(yp,int)

def fit_sm(data, feature_cols, target):
    if sm is None: return pd.DataFrame(columns=["feature","coef","pvalue","stars"])
    X = data[feature_cols].copy().fillna(data[feature_cols].median(numeric_only=True))
    std = X.std(ddof=0).replace(0,1.0)
    Xz = sm.add_constant((X-X.mean())/std, has_constant="add")
    y = data[target].astype(int).values
    try:
        fit = sm.Logit(y, Xz).fit(disp=0, maxiter=400)
    except Exception as e:
        print(f"  statsmodels failed: {e}"); return pd.DataFrame(columns=["feature","coef","pvalue","stars"])
    params = fit.params.drop("const", errors="ignore")
    pvals  = fit.pvalues.drop("const", errors="ignore")
    out = pd.DataFrame({"feature": params.index, "coef": params.values,
                        "pvalue": pvals.reindex(params.index).values})
    out["stars"] = out["pvalue"].map(p_to_stars)
    return out

# ── run model: Outcome 1, non-fac ─────────────────────────────────────────────
TARGET   = "outcome_joined_team"
controls = [c for c in ["n_sessions_attended","speaking_minutes_total"] if c in merged.columns]
nonfac   = merged[merged["is_facilitator"] == 0].copy()

print(f"\nRunning LOCO on {len(nonfac)} non-fac participants …")
yt, ys, yp = run_loco(nonfac, controls + FEAT_COLS, TARGET)

if len(np.unique(yt)) >= 2:
    auc   = roc_auc_score(yt, ys)
    auprc = average_precision_score(yt, ys)
    acc   = accuracy_score(yt, yp)
    f1    = f1_score(yt, yp, zero_division=0)
    print(f"  AUC={auc:.3f}  AUPRC={auprc:.3f}  Acc={acc:.3f}  F1={f1:.3f}")
else:
    auc = auprc = acc = f1 = float("nan")

coef_df = fit_sm(nonfac, controls + FEAT_COLS, TARGET)
coef_df["label"]  = coef_df["feature"].map(lambda f: FEAT_LABELS.get(f, f))
coef_df["bucket"] = coef_df["feature"].map(lambda f: FEAT_BUCKET.get(f, "Controls"))
coef_df.to_csv(FEATURE_DIR / "person_chunk_model_coefs.csv", index=False)

# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 1 – coefficient chart colored by bucket
# ════════════════════════════════════════════════════════════════════════════════
print("\nGenerating coefficient chart …")

plot_df = coef_df[coef_df["feature"].isin(FEAT_COLS)].copy()
plot_df = plot_df.sort_values("coef")

sig_mask = plot_df["pvalue"] < 0.05

# color: teal/orange if sig, light version of bucket color if not
def bar_color(row):
    bcolor = BUCKET_COLORS.get(row["bucket"], GRAY)
    if row["pvalue"] < 0.05:
        return TEAL if row["coef"] >= 0 else ORANGE
    return bcolor + "55"   # translucent bucket color for non-sig

colors = [bar_color(row) for _, row in plot_df.iterrows()]

fig, ax = plt.subplots(figsize=(11, 9))
fig.patch.set_facecolor(LIGHT_BG)
ax.set_facecolor(LIGHT_BG)

bars = ax.barh(plot_df["label"], plot_df["coef"],
               color=colors, height=0.7, edgecolor="white", linewidth=0.4)
ax.axvline(0, color=NAVY, linewidth=1.2, zorder=3)

for yi, (_, row) in enumerate(plot_df.reset_index(drop=True).iterrows()):
    if row["stars"]:
        x  = row["coef"] + (0.015 if row["coef"] >= 0 else -0.015)
        ha = "left" if row["coef"] >= 0 else "right"
        ax.text(x, yi, row["stars"], va="center", ha=ha,
                fontsize=11, color=NAVY, fontweight="bold")

# bucket legend
bucket_patches = [mpatches.Patch(color=c, label=b) for b, c in BUCKET_COLORS.items()]
sig_patches = [
    mpatches.Patch(color=TEAL,   label="Sig. positive (p<.05)"),
    mpatches.Patch(color=ORANGE, label="Sig. negative (p<.05)"),
]
ax.legend(handles=sig_patches + bucket_patches, loc="lower right",
          frameon=True, framealpha=0.9, fontsize=7.5,
          title="Color = significance (bright) or bucket (muted)", title_fontsize=7)

ax.set_xlabel("Standardized Logistic Coefficient", fontsize=10, color=NAVY)
ax.set_title(
    f"Do Chunk-Level Session Features Predict Team Joining?\n"
    f"LOCO-CV: AUC={auc:.3f}  AUPRC={auprc:.3f}  Acc={acc:.3f}  F1={f1:.3f}  "
    f"(n={len(nonfac)}, non-fac)",
    fontsize=11, fontweight="bold", color=NAVY, pad=10,
)
ax.spines[["top","right"]].set_visible(False)
ax.tick_params(axis="y", labelsize=8.5)
ax.tick_params(axis="x", labelsize=9)
fig.tight_layout()

path1 = FIG_DIR / "coef_chart_chunk_features.png"
fig.savefig(path1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {path1}")

# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 2 – descriptive prevalence overview (% of chunks where flag=Yes / avg)
# ════════════════════════════════════════════════════════════════════════════════
print("Generating prevalence overview …")

binary_feats   = [(f[1], f[2], f[3]) for f in FEATURES if f[4] in ("binary","rate_of_value")]
numeric_feats  = [(f[1], f[2], f[3]) for f in FEATURES if f[4] == "numeric"]

# compute prevalence
prev = {}
for col, label, bucket in binary_feats:
    if col in chunk_df.columns:
        prev[col] = {"label": label, "bucket": bucket,
                     "value": chunk_df[col].mean() * 100,  # as %
                     "type": "binary"}
for col, label, bucket in numeric_feats:
    if col in chunk_df.columns:
        prev[col] = {"label": label, "bucket": bucket,
                     "value": chunk_df[col].mean(),
                     "type": "numeric"}

prev_df = pd.DataFrame(prev).T
prev_df["value"] = prev_df["value"].astype(float)

BUCKET_ORDER = ["Idea Development","Collaboration Signals",
                "Commitment & Funding","Social & Relational","Participation Dynamics"]

fig, axes = plt.subplots(1, 5, figsize=(18, 5))
fig.patch.set_facecolor(LIGHT_BG)
fig.suptitle("Chunk-Level Session Features: Prevalence Across 1,286 Chunks",
             fontsize=13, fontweight="bold", color=NAVY, y=1.02)

for ax, bucket in zip(axes, BUCKET_ORDER):
    ax.set_facecolor(LIGHT_BG)
    bdf = prev_df[prev_df["bucket"] == bucket].copy()
    bdf = bdf.sort_values("value", ascending=True)
    bcolor = BUCKET_COLORS[bucket]
    bars = ax.barh(bdf["label"], bdf["value"], color=bcolor, height=0.6,
                   edgecolor="white", linewidth=0.4)
    for bar, (_, row) in zip(bars, bdf.iterrows()):
        v = row["value"]
        unit = "%" if row["type"] in ("binary","rate_of_value") else " (avg)"
        ax.text(v + 0.5, bar.get_y() + bar.get_height()/2,
                f"{v:.1f}{unit}", va="center", fontsize=7.5, color=NAVY)
    ax.set_title(bucket, fontsize=9, fontweight="bold", color=bcolor)
    ax.set_xlabel("% of chunks" if True else "mean", fontsize=8, color=NAVY)
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.tick_params(axis="x", labelsize=7.5)

fig.tight_layout()
path2 = FIG_DIR / "chunk_prevalence_overview.png"
fig.savefig(path2, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {path2}")

print("\n✓ Done.")
print(f"  Coef chart:   {path1}")
print(f"  Overview:     {path2}")
