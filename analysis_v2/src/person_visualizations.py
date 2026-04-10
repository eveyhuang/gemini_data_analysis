from __future__ import annotations

import math
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
)

try:
    import statsmodels.api as sm
except Exception:  # pragma: no cover
    sm = None


BASE = Path(__file__).resolve().parents[2]
DATA_DIR = BASE / "analysis_v2" / "data"
FEATURE_DIR = DATA_DIR / "person-aggregation-features"
if not FEATURE_DIR.exists():
    FEATURE_DIR = DATA_DIR

FIG_DIR = BASE / "analysis_v2" / "figures" / "person-aggregation-features"
FIG_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = FEATURE_DIR / "person_features_detailed_subcodes.csv"
if not INPUT_PATH.exists():
    INPUT_PATH = DATA_DIR / "person_features_detailed_subcodes.csv"
if not INPUT_PATH.exists():
    INPUT_PATH = FEATURE_DIR / "person_features.csv"
if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Missing expected input: {INPUT_PATH}")

df = pd.read_csv(INPUT_PATH).copy()

if "n_sessions_attended" not in df.columns and "n_sessions_seen" in df.columns:
    df["n_sessions_attended"] = df["n_sessions_seen"]
if "speaking_minutes_total" not in df.columns and "speaking_seconds_total" in df.columns:
    df["speaking_minutes_total"] = df["speaking_seconds_total"] / 60.0

targets = ["outcome_joined_team", "outcome_joined_funded_team"]
subcode_cols = [c for c in df.columns if c.startswith("subcode__") and c != "subcode__none"]

controls = [c for c in ["n_sessions_attended", "speaking_minutes_total"] if c in df.columns]
if len(controls) < 2:
    raise ValueError("Expected controls n_sessions_attended and speaking_minutes_total")

core_behavior_cols = [
    c
    for c in [
        "n_sessions_attended",
        "n_chunks_seen",
        "utterance_count",
        "speaking_minutes_total",
        "dominant_speaker_rate",
        "build_per_utterance",
        "feedback_per_utterance",
        "question_per_utterance",
    ]
    if c in df.columns
]


def norm_name(text: object) -> str:
    s = "" if text is None else str(text)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def build_facilitator_flag(base_df: pd.DataFrame) -> pd.DataFrame:
    """Create conference+person facilitator flag from session_data role labels."""
    by_conf = BASE / "finalized_matching_csvs" / "global_participant_identity_by_conference.csv"
    alias_path = BASE / "participant_alias_mapping.csv"
    session_dir = BASE / "analysis_v1" / "data"
    if not by_conf.exists() or not session_dir.exists():
        out = base_df.copy()
        out["is_facilitator"] = 0
        return out

    alias_map: dict[str, str] = {}
    if alias_path.exists():
        adf = pd.read_csv(alias_path)
        for _, r in adf.iterrows():
            a = norm_name(r.get("alias_name"))
            c = norm_name(r.get("canonical_name"))
            if a and c:
                alias_map[a] = c

    def resolve_alias(name: str) -> str:
        cur = name
        seen = set()
        while cur in alias_map and cur not in seen:
            seen.add(cur)
            cur = alias_map[cur]
        return cur

    map_df = pd.read_csv(by_conf).copy()
    map_df["conference"] = map_df["conference"].astype(str).str.strip()
    map_df["normalized_name"] = map_df["normalized_name"].astype(str).map(norm_name).map(resolve_alias)
    lookup = {(r["conference"], r["normalized_name"]): r["global_person_id"] for _, r in map_df.iterrows()}

    def role_is_facilitator(role: object) -> int:
        rs = str(role).strip().lower()
        if not rs:
            return 0
        return int(
            ("facilitator" in rs)
            or ("program director" in rs)
            or ("program officer" in rs)
        )

    rows = []
    for jp in sorted(session_dir.glob("*/session_data/*.json")):
        conference = jp.parent.parent.name.strip()
        try:
            payload = json.loads(jp.read_text())
        except Exception:
            continue
        for row in payload.get("all_data", []):
            if not isinstance(row, dict):
                continue
            speaker = resolve_alias(norm_name(row.get("speaker")))
            if not speaker:
                continue
            pid = lookup.get((conference, speaker))
            if not pid:
                continue
            rows.append(
                {
                    "conference": conference,
                    "global_person_id": pid,
                    "fac_count": role_is_facilitator(row.get("role")),
                    "n_rows": 1,
                }
            )

    out = base_df.copy()
    if not rows:
        out["is_facilitator"] = 0
        return out

    role_df = pd.DataFrame(rows)
    role_df = role_df.groupby(["conference", "global_person_id"], as_index=False)[["fac_count", "n_rows"]].sum()
    role_df["is_facilitator"] = (role_df["fac_count"] > 0).astype(int)
    out = out.merge(
        role_df[["conference", "global_person_id", "is_facilitator"]],
        on=["conference", "global_person_id"],
        how="left",
    )
    out["is_facilitator"] = out["is_facilitator"].fillna(0).astype(int)
    return out


df = build_facilitator_flag(df)
if "is_facilitator" not in controls:
    controls.append("is_facilitator")


def build_within_session_matrix(target: str):
    """Return a within-session heatmap matrix (top subcodes x top sessions)
    showing log-count difference between outcome=1 and outcome=0 utterance codes.
    """
    by_conf = BASE / "finalized_matching_csvs" / "global_participant_identity_by_conference.csv"
    alias_path = BASE / "participant_alias_mapping.csv"
    outputs_dir = BASE / "outputs"
    if not by_conf.exists() or not outputs_dir.exists():
        return pd.DataFrame()

    alias_map: dict[str, str] = {}
    if alias_path.exists():
        adf = pd.read_csv(alias_path)
        for _, r in adf.iterrows():
            a = norm_name(r.get("alias_name"))
            c = norm_name(r.get("canonical_name"))
            if a and c:
                alias_map[a] = c

    def resolve_alias(name: str) -> str:
        cur = name
        seen = set()
        while cur in alias_map and cur not in seen:
            seen.add(cur)
            cur = alias_map[cur]
        return cur

    map_df = pd.read_csv(by_conf).copy()
    map_df["conference"] = map_df["conference"].astype(str).str.strip()
    map_df["normalized_name"] = map_df["normalized_name"].astype(str).map(norm_name).map(resolve_alias)
    lookup = {(r["conference"], r["normalized_name"]): r["global_person_id"] for _, r in map_df.iterrows()}

    outcome_lookup = {
        (r["conference"], r["global_person_id"]): int(r[target])
        for _, r in df[["conference", "global_person_id", target]].drop_duplicates().iterrows()
    }

    rows = []
    for conf_dir in sorted(outputs_dir.glob("*")):
        if not conf_dir.is_dir():
            continue
        conference = conf_dir.name
        for out_dir in sorted(conf_dir.glob("output_*")):
            if not out_dir.is_dir():
                continue
            session_group = out_dir.name.replace("output_", "")
            for jp in out_dir.rglob("*.json"):
                try:
                    payload = json.loads(jp.read_text())
                except Exception:
                    continue
                if not isinstance(payload, dict):
                    continue
                utterances = payload.get("utterance_annotations", [])
                if not isinstance(utterances, list):
                    continue
                for utt in utterances:
                    if not isinstance(utt, dict):
                        continue
                    sp = resolve_alias(norm_name(utt.get("speaker")))
                    pid = lookup.get((conference, sp))
                    if not pid:
                        continue
                    out = outcome_lookup.get((conference, pid))
                    if out not in (0, 1):
                        continue
                    for c in (utt.get("codes") or []):
                        if not isinstance(c, dict):
                            continue
                        sc = str(c.get("subcode", "")).strip().lower()
                        if not sc:
                            continue
                        sc = re.sub(r"[^a-z0-9]+", "_", sc).strip("_")
                        rows.append(
                            {
                                "session_key": f"{conference}:{session_group}",
                                "subcode": f"subcode__{sc}",
                                "outcome": out,
                            }
                        )
    if not rows:
        return pd.DataFrame()

    r = pd.DataFrame(rows)
    g = r.groupby(["session_key", "subcode", "outcome"], as_index=False).size()
    p = g.pivot_table(index=["session_key", "subcode"], columns="outcome", values="size", fill_value=0).reset_index()
    p.columns = [str(c) for c in p.columns]
    if "0" not in p.columns:
        p["0"] = 0
    if "1" not in p.columns:
        p["1"] = 0
    p["score"] = np.log1p(p["1"]) - np.log1p(p["0"])

    top_subcodes = (
        p.groupby("subcode", as_index=False)[["0", "1"]]
        .sum()
        .assign(total=lambda x: x["0"] + x["1"])
        .sort_values("total", ascending=False)
        .head(12)["subcode"]
        .tolist()
    )
    top_sessions = (
        p.groupby("session_key", as_index=False)[["0", "1"]]
        .sum()
        .assign(total=lambda x: x["0"] + x["1"])
        .sort_values("total", ascending=False)
        .head(16)["session_key"]
        .tolist()
    )

    p = p[p["subcode"].isin(top_subcodes) & p["session_key"].isin(top_sessions)]
    m = p.pivot_table(index="subcode", columns="session_key", values="score", fill_value=0.0)
    m = m.reindex(index=top_subcodes, columns=top_sessions)
    return m


def p_to_stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def run_loco_predictions(data: pd.DataFrame, feature_cols: list[str], target: str):
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced")),
        ]
    )

    y_true_all: list[int] = []
    y_score_all: list[float] = []
    y_pred_all: list[int] = []

    for holdout in sorted(data["conference"].dropna().unique()):
        tr = data[data["conference"] != holdout]
        te = data[data["conference"] == holdout]
        if tr.empty or te.empty:
            continue
        if tr[target].nunique() < 2:
            continue

        X_tr = tr[feature_cols]
        X_te = te[feature_cols]
        y_tr = tr[target].astype(int)
        y_te = te[target].astype(int)

        pipe.fit(X_tr, y_tr)
        p = pipe.predict_proba(X_te)[:, 1]
        y_pred = (p >= 0.5).astype(int)

        y_true_all.extend(y_te.tolist())
        y_score_all.extend(p.tolist())
        y_pred_all.extend(y_pred.tolist())

    y_true = np.asarray(y_true_all, dtype=int)
    y_score = np.asarray(y_score_all, dtype=float)
    y_pred = np.asarray(y_pred_all, dtype=int)
    return y_true, y_score, y_pred


metrics_rows = []
sensitivity_rows = []

for target in targets:

    # ----------------------------
    # (1) Exploration figure
    # ----------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    fig.suptitle(
        f"Behavior Exploration ({target})\nPerson-within-conference view",
        fontsize=15,
        fontweight="bold",
    )

    # A) Core behavior means by outcome
    means = df.groupby(target)[core_behavior_cols].mean().T
    x = np.arange(len(means.index))
    width = 0.38
    axes[0, 0].barh(x - width / 2, means.get(0, pd.Series(index=means.index, data=0)), height=width, label="Outcome=0")
    axes[0, 0].barh(x + width / 2, means.get(1, pd.Series(index=means.index, data=0)), height=width, label="Outcome=1")
    axes[0, 0].set_yticks(x)
    axes[0, 0].set_yticklabels(means.index)
    axes[0, 0].set_title("Core Behavior Means by Outcome")
    axes[0, 0].legend(frameon=False)

    # B) Top subcodes by outcome separation
    if subcode_cols:
        sc_means = df.groupby(target)[subcode_cols].mean().T
        sc_means["abs_diff"] = (sc_means.get(1, 0) - sc_means.get(0, 0)).abs()
        top_sc = sc_means.sort_values("abs_diff", ascending=False).head(12)
        y_pos = np.arange(len(top_sc.index))
        axes[0, 1].barh(y_pos - width / 2, top_sc.get(0, pd.Series(index=top_sc.index, data=0)), height=width, label="Outcome=0")
        axes[0, 1].barh(y_pos + width / 2, top_sc.get(1, pd.Series(index=top_sc.index, data=0)), height=width, label="Outcome=1")
        axes[0, 1].set_yticks(y_pos)
        axes[0, 1].set_yticklabels(top_sc.index)
        axes[0, 1].set_title("Top Detailed Subcodes by Mean Difference")
        axes[0, 1].legend(frameon=False)
    else:
        axes[0, 1].text(0.5, 0.5, "No detailed subcodes found", ha="center", va="center")
        axes[0, 1].set_title("Top Detailed Subcodes")

    # C) Clean control-variable distributions by outcome (split panels to avoid scale compression)
    ctrl_plot_cols = [c for c in ["n_sessions_attended", "speaking_minutes_total"] if c in df.columns]
    if ctrl_plot_cols:
        axes[1, 0].set_title("Control Variable Distributions by Outcome")
        axes[1, 0].axis("off")

        # Left mini-panel: n_sessions_attended
        left_ax = axes[1, 0].inset_axes([0.06, 0.17, 0.42, 0.72])
        ns0 = df.loc[df[target] == 0, "n_sessions_attended"].dropna().values if "n_sessions_attended" in ctrl_plot_cols else []
        ns1 = df.loc[df[target] == 1, "n_sessions_attended"].dropna().values if "n_sessions_attended" in ctrl_plot_cols else []
        if len(ns0) and len(ns1):
            bp_left = left_ax.boxplot([ns0, ns1], patch_artist=True, tick_labels=["Y=0", "Y=1"], showfliers=False)
            bp_left["boxes"][0].set_facecolor("#90caf9")
            bp_left["boxes"][1].set_facecolor("#ffcc80")
        left_ax.set_title("n_sessions_attended", fontsize=10)
        left_ax.set_ylabel("Count")

        # Right mini-panel: speaking_minutes_total
        right_ax = axes[1, 0].inset_axes([0.56, 0.17, 0.42, 0.72])
        ss0 = df.loc[df[target] == 0, "speaking_minutes_total"].dropna().values if "speaking_minutes_total" in ctrl_plot_cols else []
        ss1 = df.loc[df[target] == 1, "speaking_minutes_total"].dropna().values if "speaking_minutes_total" in ctrl_plot_cols else []
        if len(ss0) and len(ss1):
            bp_right = right_ax.boxplot([ss0, ss1], patch_artist=True, tick_labels=["Y=0", "Y=1"], showfliers=False)
            bp_right["boxes"][0].set_facecolor("#90caf9")
            bp_right["boxes"][1].set_facecolor("#ffcc80")
        right_ax.set_title("speaking_minutes_total", fontsize=10)
        right_ax.set_ylabel("Minutes")
    else:
        axes[1, 0].text(0.5, 0.5, "Control distributions unavailable", ha="center", va="center")
        axes[1, 0].set_title("Control Variable Distributions")

    # D) Outcome prevalence by conference
    conf_prev = df.groupby("conference", as_index=False)[target].mean().sort_values(target, ascending=False)
    axes[1, 1].bar(conf_prev["conference"], conf_prev[target], color="#6aaed6")
    axes[1, 1].set_title("Outcome Rate by Conference")
    axes[1, 1].set_ylabel("Positive Rate")
    axes[1, 1].tick_params(axis="x", rotation=45)

    exp_fig_path = FIG_DIR / f"person_behavior_exploration_{target}.png"
    fig.savefig(exp_fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ----------------------------
    # (2) Model + effect-size/significance figure
    # ----------------------------
    subcode_eligible = [c for c in subcode_cols if df[c].sum() >= 10]
    top_k = 14
    top_subcodes = sorted(subcode_eligible, key=lambda c: df[c].sum(), reverse=True)[:top_k]
    model_features = controls + top_subcodes

    y_true, y_score, y_pred = run_loco_predictions(df, model_features, target)
    if len(np.unique(y_true)) < 2:
        auc = np.nan
        auprc = np.nan
        acc = np.nan
        f1 = np.nan
        cm = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        fpr, tpr = np.array([0, 1]), np.array([0, 1])
    else:
        auc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        fpr, tpr, _ = roc_curve(y_true, y_score)

    # statsmodels for coefficients + p-values
    coef_plot_df = pd.DataFrame({"feature": model_features, "coef": np.nan, "pvalue": np.nan})
    if sm is not None:
        try:
            X = df[model_features].copy()
            X = X.fillna(X.median(numeric_only=True))
            std = X.std(ddof=0).replace(0, 1.0)
            Xz = (X - X.mean()) / std
            Xz = sm.add_constant(Xz, has_constant="add")
            yy = df[target].astype(int).values
            fit = sm.Logit(yy, Xz).fit(disp=0, maxiter=200)
            params = fit.params.drop("const", errors="ignore")
            pvals = fit.pvalues.drop("const", errors="ignore")
            coef_plot_df = pd.DataFrame(
                {"feature": params.index, "coef": params.values, "pvalue": pvals.reindex(params.index).values}
            )
        except Exception:
            pass

    coef_plot_df["abs_coef"] = coef_plot_df["coef"].abs()
    coef_plot_df = coef_plot_df.sort_values("coef")
    coef_plot_df["stars"] = coef_plot_df["pvalue"].map(p_to_stars)

    model_fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    gs = model_fig.add_gridspec(2, 2)
    ax1 = model_fig.add_subplot(gs[0, 0])
    ax2 = model_fig.add_subplot(gs[0, 1])
    ax3 = model_fig.add_subplot(gs[1, 0])
    ax4 = model_fig.add_subplot(gs[1, 1])
    model_fig.suptitle(
        f"Person-Level Model Results ({target})\nControls: n_sessions_attended + speaking_minutes_total + is_facilitator",
        fontsize=15,
        fontweight="bold",
    )

    # performance bars
    metric_names = ["Accuracy", "F1", "AUC", "AUPRC"]
    metric_vals = [acc, f1, auc, auprc]
    ax1.bar(metric_names, metric_vals, color=["#00c853", "#2962ff", "#ff6d00", "#aa00ff"])
    ax1.set_ylim(0, 1.0)
    ax1.set_title("Performance Metrics (LOCO pooled)")
    for i, v in enumerate(metric_vals):
        if pd.notna(v):
            ax1.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=10)

    # ROC
    ax2.plot(fpr, tpr, color="#8e24aa", lw=2, label=f"Model ROC (AUC={auc:.3f})" if pd.notna(auc) else "Model ROC")
    ax2.plot([0, 1], [0, 1], color="red", linestyle="--", lw=1.5, label="Baseline (AUC=0.5)")
    ax2.fill_between(fpr, tpr, alpha=0.15, color="#8e24aa")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend(loc="lower right", frameon=False)

    # confusion matrix
    im = ax3.imshow(cm, cmap="Purples")
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(["Pred 0", "Pred 1"])
    ax3.set_yticklabels(["Actual 0", "Actual 1"])
    ax3.set_title("Confusion Matrix (threshold = 0.5)")
    for i in range(2):
        for j in range(2):
            if np.isfinite(cm[i, j]):
                ax3.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > np.nanmax(cm) / 2 else "black")

    # coefficients with significance
    if not coef_plot_df.empty:
        colors = ["#00c853" if c >= 0 else "#ff1744" for c in coef_plot_df["coef"]]
        ax4.barh(coef_plot_df["feature"], coef_plot_df["coef"], color=colors)
        ax4.axvline(0, color="black", lw=1)
        ax4.set_title("Effect Size (coef) + Significance")
        ax4.set_xlabel("Standardized Logistic Coefficient")
        for yi, (coef, stars) in enumerate(zip(coef_plot_df["coef"], coef_plot_df["stars"])):
            x = coef + (0.02 if coef >= 0 else -0.02)
            ha = "left" if coef >= 0 else "right"
            ax4.text(x, yi, stars, va="center", ha=ha, fontsize=11, color="black")
    else:
        ax4.text(0.5, 0.5, "Coefficient model unavailable", ha="center", va="center")
        ax4.set_title("Effect Size + Significance")

    model_fig_path = FIG_DIR / f"person_model_results_{target}.png"
    model_fig.savefig(model_fig_path, dpi=150, bbox_inches="tight")
    plt.close(model_fig)

    # save coefficient table for this target
    coef_out = FEATURE_DIR / f"person_model_coefficients_{target}.csv"
    coef_plot_df.sort_values("abs_coef", ascending=False).to_csv(coef_out, index=False)

    metrics_rows.append(
        {
            "target": target,
            "n_rows": len(df),
            "feature_count": len(model_features),
            "accuracy_loco_pooled": acc,
            "f1_loco_pooled": f1,
            "auc_loco_pooled": auc,
            "auprc_loco_pooled": auprc,
            "behavior_figure_path": str(exp_fig_path),
            "model_figure_path": str(model_fig_path),
            "coef_table_path": str(coef_out),
        }
    )

    # non-facilitator-only sensitivity check
    nonfac = df[df["is_facilitator"] == 0].copy()
    nonfac_features = [c for c in model_features if c != "is_facilitator"]
    if not nonfac.empty and nonfac[target].nunique() >= 2:
        ny_true, ny_score, ny_pred = run_loco_predictions(nonfac, nonfac_features, target)
        if len(np.unique(ny_true)) >= 2:
            sensitivity_rows.append(
                {
                    "target": target,
                    "sample": "non_facilitator_only",
                    "n_rows": len(nonfac),
                    "feature_count": len(nonfac_features),
                    "accuracy_loco_pooled": accuracy_score(ny_true, ny_pred),
                    "f1_loco_pooled": f1_score(ny_true, ny_pred, zero_division=0),
                    "auc_loco_pooled": roc_auc_score(ny_true, ny_score),
                    "auprc_loco_pooled": average_precision_score(ny_true, ny_score),
                }
            )

    sensitivity_rows.append(
        {
            "target": target,
            "sample": "all_rows",
            "n_rows": len(df),
            "feature_count": len(model_features),
            "accuracy_loco_pooled": acc,
            "f1_loco_pooled": f1,
            "auc_loco_pooled": auc,
            "auprc_loco_pooled": auprc,
        }
    )

summary_df = pd.DataFrame(metrics_rows)
summary_out = FEATURE_DIR / "person_model_visualization_summary.csv"
summary_df.to_csv(summary_out, index=False)
sense_df = pd.DataFrame(sensitivity_rows)
sense_out = FEATURE_DIR / "person_model_non_facilitator_sensitivity.csv"
sense_df.to_csv(sense_out, index=False)

print("Saved summary:", summary_out)
print(summary_df.to_string(index=False))
print("\nSaved non-facilitator sensitivity:", sense_out)
print(sense_df.to_string(index=False))
