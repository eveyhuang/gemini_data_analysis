"""
role_speaker_analysis.py
Builds person-level features from chunk-level "role speaker" fields:
  - dominant_speaker_name       (when dominant_speaker_flag == "Yes")
  - complementarity_recognition_speaker
  - cross_disciplinary_bridging_speaker
  - commitment_signal_speaker

Produces:
  person_role_speaker_features.csv
  coef_chart_role_speakers.png
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

# ─── colors ───────────────────────────────────────────────────────────────────
NAVY     = "#1E2761"
TEAL     = "#028090"
ORANGE   = "#E07B4A"
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
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
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

# ─── role speaker fields to extract ──────────────────────────────────────────
# (field_in_json, feature_col_name, display_label, conditional_flag_field)
ROLE_FIELDS = [
    ("dominant_speaker_name",           "n_dominant_speaker",
     "Dominant Speaker",                "dominant_speaker_flag"),
    ("complementarity_recognition_speaker", "n_complementarity_recognition",
     "Complementarity Recognition",      "explicit_complementarity_recognition"),
    ("cross_disciplinary_bridging_speaker", "n_cross_disciplinary_bridging",
     "Cross-Disciplinary Bridging",      "cross_disciplinary_bridging"),
    ("commitment_signal_speaker",        "n_commitment_signal",
     "Commitment Signal",                "explicit_commitment_signal"),
]

ROLE_COLS   = [r[1] for r in ROLE_FIELDS]
ROLE_LABELS = {r[1]: r[2] for r in ROLE_FIELDS}

# ─── scan JSONs ───────────────────────────────────────────────────────────────
print("Scanning output JSONs for role speaker fields …")
rows: list[dict] = []
n_chunks_total = 0

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
        cs = payload.get("chunk_summary", {})
        if not cs:
            continue
        n_chunks_total += 1

        for (speaker_field, feat_col, _, flag_field) in ROLE_FIELDS:
            # Only credit when the flag says something happened
            flag_val = str(cs.get(flag_field, "No")).strip().lower()
            if flag_val not in ("yes", "true", "1"):
                continue
            raw_speaker = cs.get(speaker_field, "")
            if not raw_speaker or str(raw_speaker).strip().lower() in ("none", "na", "n/a", ""):
                continue
            sp = resolve_alias(norm_name(raw_speaker))
            pid = lookup.get((conference, sp))
            if pid:
                rows.append({
                    "conference": conference,
                    "global_person_id": pid,
                    "role_col": feat_col,
                })

print(f"  {n_chunks_total:,} chunks scanned  |  {len(rows):,} role-speaker hits")

# ─── aggregate to person level ────────────────────────────────────────────────
role_df = pd.DataFrame(rows)
if role_df.empty:
    print("No role-speaker data found — exiting.")
    raise SystemExit(1)

# count how many chunks each person was flagged in each role (per conference)
agg = (role_df.groupby(["conference", "global_person_id", "role_col"])
              .size().reset_index(name="count"))
pivot = agg.pivot_table(
    index=["conference", "global_person_id"],
    columns="role_col", values="count", fill_value=0
).reset_index()
pivot.columns.name = None
for col in ROLE_COLS:
    if col not in pivot.columns:
        pivot[col] = 0

print("\nPer-person role counts (non-zero rows):")
for col in ROLE_COLS:
    n_nonzero = (pivot[col] > 0).sum()
    print(f"  {col:40s}: {n_nonzero} people with ≥1 flag")

# ─── merge with base features ─────────────────────────────────────────────────
base = pd.read_csv(FEATURE_DIR / "person_codename_features.csv")
merged = base.merge(pivot, on=["conference", "global_person_id"], how="left")
for col in ROLE_COLS:
    merged[col] = merged[col].fillna(0)

# ─── also build binary (ever-flagged) versions ────────────────────────────────
binary_cols = []
for col in ROLE_COLS:
    bcol = col.replace("n_", "ever_")
    merged[bcol] = (merged[col] > 0).astype(int)
    binary_cols.append(bcol)

print(f"\nMerged dataset: {len(merged)} rows")
merged.to_csv(FEATURE_DIR / "person_role_speaker_features.csv", index=False)
print("Saved person_role_speaker_features.csv")

# ─── LOCO-CV helper ──────────────────────────────────────────────────────────
def run_loco(data, feature_cols, target):
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

def fit_sm(data, feature_cols, target):
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
    out = pd.DataFrame({"feature": params.index,
                        "coef": params.values,
                        "pvalue": pvals.reindex(params.index).values})
    out["stars"] = out["pvalue"].map(p_to_stars)
    return out

# ─── run models ───────────────────────────────────────────────────────────────
TARGET    = "outcome_joined_team"
controls  = [c for c in ["n_sessions_attended", "speaking_minutes_total"]
             if c in merged.columns]

# non-facilitator sample
nonfac = merged[merged["is_facilitator"] == 0].copy()
print(f"\nNon-fac sample: n={len(nonfac)}")

# --- COUNT model (continuous) ---
count_features = controls + ROLE_COLS
yt_c, ys_c, yp_c = run_loco(nonfac, count_features, TARGET)
if len(np.unique(yt_c)) >= 2:
    auc_c   = roc_auc_score(yt_c, ys_c)
    auprc_c = average_precision_score(yt_c, ys_c)
    acc_c   = accuracy_score(yt_c, yp_c)
    f1_c    = f1_score(yt_c, yp_c, zero_division=0)
    print(f"  Count model:  AUC={auc_c:.3f}  AUPRC={auprc_c:.3f}  Acc={acc_c:.3f}  F1={f1_c:.3f}")
else:
    auc_c = auprc_c = acc_c = f1_c = float("nan")

coef_c = fit_sm(nonfac, count_features, TARGET)
coef_c["label"] = coef_c["feature"].map(
    lambda f: ROLE_LABELS.get(f, f.replace("_", " ").title()))
coef_c.to_csv(FEATURE_DIR / "person_role_speaker_coefs_count.csv", index=False)

# --- BINARY model (ever flagged) ---
binary_features = controls + binary_cols
yt_b, ys_b, yp_b = run_loco(nonfac, binary_features, TARGET)
if len(np.unique(yt_b)) >= 2:
    auc_b   = roc_auc_score(yt_b, ys_b)
    auprc_b = average_precision_score(yt_b, ys_b)
    acc_b   = accuracy_score(yt_b, yp_b)
    f1_b    = f1_score(yt_b, yp_b, zero_division=0)
    print(f"  Binary model: AUC={auc_b:.3f}  AUPRC={auprc_b:.3f}  Acc={acc_b:.3f}  F1={f1_b:.3f}")
else:
    auc_b = auprc_b = acc_b = f1_b = float("nan")

coef_b = fit_sm(nonfac, binary_features, TARGET)
coef_b["label"] = coef_b["feature"].map(
    lambda f: ROLE_LABELS.get(f.replace("ever_", "n_"), f.replace("_", " ").title()))
coef_b.to_csv(FEATURE_DIR / "person_role_speaker_coefs_binary.csv", index=False)

# ─── chart: side-by-side or stacked role coefs (COUNT model) ─────────────────
print("\nGenerating role speaker coefficient charts …")

for (coef_df, suffix, auc, auprc, acc, f1, feat_cols, title_tag) in [
    (coef_c, "count",  auc_c, auprc_c, acc_c, f1_c,
     ROLE_COLS, "Times Flagged in Role (Count)"),
    (coef_b, "binary", auc_b, auprc_b, acc_b, f1_b,
     binary_cols, "Ever Flagged in Role (Binary)"),
]:
    plot_df = coef_df[coef_df["feature"].isin(feat_cols)].copy()
    plot_df = plot_df.sort_values("coef")

    colors = [TEAL  if (c >= 0 and p < 0.05) else
              ORANGE if (c <  0 and p < 0.05) else GRAY
              for c, p in zip(plot_df["coef"], plot_df["pvalue"])]

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor(LIGHT_BG)
    ax.set_facecolor(LIGHT_BG)

    ax.barh(plot_df["label"], plot_df["coef"],
            color=colors, height=0.55, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color=NAVY, linewidth=1.2, zorder=3)

    for yi, (_, row) in enumerate(plot_df.reset_index(drop=True).iterrows()):
        if row["stars"]:
            x  = row["coef"] + (0.02 if row["coef"] >= 0 else -0.02)
            ha = "left" if row["coef"] >= 0 else "right"
            ax.text(x, yi, row["stars"], va="center", ha=ha,
                    fontsize=12, color=NAVY, fontweight="bold")

    ax.set_xlabel("Standardized Logistic Coefficient", fontsize=10, color=NAVY)
    ax.set_title(
        f"Do Gemini-Assigned Speaker Roles Predict Team Joining?\n"
        f"{title_tag} — LOCO-CV: AUC={auc:.3f}  AUPRC={auprc:.3f}  "
        f"Acc={acc:.3f}  F1={f1:.3f}  (n={len(nonfac)}, non-fac)",
        fontsize=10, fontweight="bold", color=NAVY, pad=10,
    )

    teal_p   = mpatches.Patch(color=TEAL,   label="Positive predictor (p<.05)")
    orange_p = mpatches.Patch(color=ORANGE, label="Negative predictor (p<.05)")
    gray_p   = mpatches.Patch(color=GRAY,   label="Not significant (p≥.05)")
    ax.legend(handles=[teal_p, orange_p, gray_p],
              loc="lower right", frameon=True, framealpha=0.9, fontsize=8)

    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", labelsize=9)

    fig.tight_layout()
    path = FIG_DIR / f"coef_chart_role_speakers_{suffix}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

print("\n✓ Done.")
