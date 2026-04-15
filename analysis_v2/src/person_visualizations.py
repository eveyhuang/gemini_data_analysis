from __future__ import annotations

import math
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
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


def metric_row(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if len(np.unique(y_true)) < 2:
        return {
            "accuracy_loco_pooled": np.nan,
            "f1_loco_pooled": np.nan,
            "auc_loco_pooled": np.nan,
            "auprc_loco_pooled": np.nan,
        }
    return {
        "accuracy_loco_pooled": float(accuracy_score(y_true, y_pred)),
        "f1_loco_pooled": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_loco_pooled": float(roc_auc_score(y_true, y_score)),
        "auprc_loco_pooled": float(average_precision_score(y_true, y_score)),
    }


def fit_full_sample_coefficients(data: pd.DataFrame, feature_cols: list[str], target: str, c_val: float = 0.5) -> pd.DataFrame:
    x = data[feature_cols].copy()
    x = x.fillna(x.median(numeric_only=True))
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x.values)
    y = data[target].astype(int).values
    clf = LogisticRegression(max_iter=5000, class_weight="balanced", C=c_val)
    clf.fit(x_scaled, y)
    out = pd.DataFrame({"feature": feature_cols, "coef": clf.coef_[0]})
    out["abs_coef"] = out["coef"].abs()
    return out.sort_values("abs_coef", ascending=False)


metrics_rows = []
sensitivity_rows = []
supplemental_rows = []
supplemental_coef_rows = []

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

    # ----------------------------
    # (3) Supplemental checks (no figure changes)
    # ----------------------------
    base_metrics = metric_row(y_true, y_score, y_pred)

    # 3a) Conference fixed-effects check (conference dummies)
    df_fe = df.copy()
    conf_dummies = pd.get_dummies(df_fe["conference"], prefix="conf", drop_first=True, dtype=float)
    df_fe = pd.concat([df_fe, conf_dummies], axis=1)
    fe_features = model_features + conf_dummies.columns.tolist()
    fe_true, fe_score, fe_pred = run_loco_predictions(df_fe, fe_features, target)
    fe_metrics = metric_row(fe_true, fe_score, fe_pred)
    supplemental_rows.append(
        {
            "target": target,
            "check_name": "conference_fixed_effects",
            "n_rows": len(df_fe),
            "n_features": len(fe_features),
            **fe_metrics,
            "delta_auc_vs_base": fe_metrics["auc_loco_pooled"] - base_metrics["auc_loco_pooled"],
            "delta_auprc_vs_base": fe_metrics["auprc_loco_pooled"] - base_metrics["auprc_loco_pooled"],
        }
    )
    fe_coef = fit_full_sample_coefficients(df_fe, fe_features, target)
    fe_coef["target"] = target
    fe_coef["check_name"] = "conference_fixed_effects"
    fe_coef["feature_type"] = np.where(fe_coef["feature"].str.startswith("conf_"), "conference_dummy", "behavior_or_control")
    supplemental_coef_rows.append(fe_coef)

    # 3b) Facilitator interaction check (role-modified behavior effects)
    key_interactions = [
        "subcode__proposes_process",
        "subcode__extends_existing_idea",
        "subcode__invites_contribution",
        "subcode__shares_domain_knowledge",
        "subcode__individual_framing",
    ]
    key_interactions = [c for c in key_interactions if c in df.columns]
    df_int = df.copy()
    int_cols: list[str] = []
    for c in key_interactions:
        int_name = f"is_facilitator_x_{c.replace('subcode__', '')}"
        df_int[int_name] = df_int["is_facilitator"] * df_int[c]
        int_cols.append(int_name)

    int_features = model_features + int_cols
    int_true, int_score, int_pred = run_loco_predictions(df_int, int_features, target)
    int_metrics = metric_row(int_true, int_score, int_pred)
    supplemental_rows.append(
        {
            "target": target,
            "check_name": "facilitator_behavior_interactions",
            "n_rows": len(df_int),
            "n_features": len(int_features),
            **int_metrics,
            "delta_auc_vs_base": int_metrics["auc_loco_pooled"] - base_metrics["auc_loco_pooled"],
            "delta_auprc_vs_base": int_metrics["auprc_loco_pooled"] - base_metrics["auprc_loco_pooled"],
        }
    )
    int_coef = fit_full_sample_coefficients(df_int, int_features, target)
    int_coef["target"] = target
    int_coef["check_name"] = "facilitator_behavior_interactions"
    int_coef["feature_type"] = np.where(int_coef["feature"].str.startswith("is_facilitator_x_"), "interaction_term", "behavior_or_control")
    supplemental_coef_rows.append(int_coef)

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

supp_df = pd.DataFrame(supplemental_rows)
supp_out = FEATURE_DIR / "person_model_supplemental_checks.csv"
supp_df.to_csv(supp_out, index=False)
supp_coef_df = pd.concat(supplemental_coef_rows, ignore_index=True)
supp_coef_out = FEATURE_DIR / "person_model_supplemental_coefficients.csv"
supp_coef_df.to_csv(supp_coef_out, index=False)

print("\nSaved supplemental checks:", supp_out)
print(supp_df.to_string(index=False))
print("\nSaved supplemental coefficients:", supp_coef_out)

# ----------------------------
# (4) Heckman-style two-stage selection model
# outcome 1: on a team (selection)
# outcome 2: funded team (conditional on being on a team)
# ----------------------------
heckman_summary_rows: list[dict] = []
stage1_coef_df = pd.DataFrame()
stage2_coef_df = pd.DataFrame()

if sm is not None:
    # keep consistent feature set with existing modeling pipeline
    subcode_eligible = [c for c in subcode_cols if df[c].sum() >= 10]
    top_k = 14
    top_subcodes = sorted(subcode_eligible, key=lambda c: df[c].sum(), reverse=True)[:top_k]
    selection_features = controls + top_subcodes
    outcome_features = controls + top_subcodes

    # stage 1: selection equation (Probit) on full sample
    y_sel = df["outcome_joined_team"].astype(int).values
    x_sel_raw = df[selection_features].copy().fillna(df[selection_features].median(numeric_only=True))
    x_sel = sm.add_constant(x_sel_raw, has_constant="add")
    sel_fit = sm.Probit(y_sel, x_sel).fit(disp=0, maxiter=200)

    # inverse Mills ratio for selected observations
    xb = np.clip(np.asarray(sel_fit.predict(x_sel, which="linear"), dtype=float), -8.0, 8.0)
    phi = norm.pdf(xb)
    Phi = np.clip(norm.cdf(xb), 1e-8, 1 - 1e-8)
    imr = phi / Phi

    work = df.copy()
    work["imr_selection"] = imr

    # stage 2: funded outcome conditional on selection
    obs = work[work["outcome_joined_team"] == 1].copy()
    y_out = obs["outcome_joined_funded_team"].astype(int).values
    x_out_raw = obs[outcome_features + ["imr_selection"]].copy()
    x_out_raw = x_out_raw.fillna(x_out_raw.median(numeric_only=True))
    x_out = sm.add_constant(x_out_raw, has_constant="add")
    try:
        out_fit = sm.Logit(y_out, x_out).fit(disp=0, maxiter=200)
        stage2_model_type = "logit"
    except Exception:
        # Fallback for singular Hessian/separation cases.
        out_fit = sm.GLM(y_out, x_out, family=sm.families.Binomial()).fit(maxiter=200)
        stage2_model_type = "glm_binomial_fallback"

    # stage metrics
    p_sel = np.clip(np.asarray(sel_fit.predict(x_sel), dtype=float), 1e-8, 1 - 1e-8)
    p_out = np.clip(np.asarray(out_fit.predict(x_out), dtype=float), 1e-8, 1 - 1e-8)
    y_out_pred = (p_out >= 0.5).astype(int)

    sel_auc = roc_auc_score(y_sel, p_sel) if len(np.unique(y_sel)) >= 2 else np.nan
    sel_auprc = average_precision_score(y_sel, p_sel) if len(np.unique(y_sel)) >= 2 else np.nan
    out_auc = roc_auc_score(y_out, p_out) if len(np.unique(y_out)) >= 2 else np.nan
    out_auprc = average_precision_score(y_out, p_out) if len(np.unique(y_out)) >= 2 else np.nan
    out_acc = accuracy_score(y_out, y_out_pred) if len(np.unique(y_out)) >= 2 else np.nan
    out_f1 = f1_score(y_out, y_out_pred, zero_division=0) if len(np.unique(y_out)) >= 2 else np.nan

    heckman_summary_rows.append(
        {
            "selection_target": "outcome_joined_team",
            "outcome_target": "outcome_joined_funded_team",
            "n_total": int(len(work)),
            "n_selected": int(len(obs)),
            "selection_n_features": int(len(selection_features)),
            "outcome_n_features_plus_imr": int(len(outcome_features) + 1),
            "selection_auc": sel_auc,
            "selection_auprc": sel_auprc,
            "outcome_auc_selected_only": out_auc,
            "outcome_auprc_selected_only": out_auprc,
            "outcome_accuracy_selected_only": out_acc,
            "outcome_f1_selected_only": out_f1,
            "imr_coef_stage2": float(out_fit.params.get("imr_selection", np.nan)),
            "imr_pvalue_stage2": float(out_fit.pvalues.get("imr_selection", np.nan)),
            "imr_significance": p_to_stars(float(out_fit.pvalues.get("imr_selection", np.nan))),
            "stage2_model_type": stage2_model_type,
        }
    )

    stage1_coef_df = pd.DataFrame(
        {
            "feature": sel_fit.params.index,
            "coef": sel_fit.params.values,
            "pvalue": sel_fit.pvalues.reindex(sel_fit.params.index).values,
        }
    )
    stage1_coef_df["stage"] = "selection_probit"
    stage1_coef_df["stars"] = stage1_coef_df["pvalue"].map(p_to_stars)
    stage1_coef_df["abs_coef"] = stage1_coef_df["coef"].abs()
    stage1_coef_df = stage1_coef_df.sort_values("abs_coef", ascending=False)

    stage2_coef_df = pd.DataFrame(
        {
            "feature": out_fit.params.index,
            "coef": out_fit.params.values,
            "pvalue": out_fit.pvalues.reindex(out_fit.params.index).values,
        }
    )
    stage2_coef_df["stage"] = f"funded_{stage2_model_type}_with_imr"
    stage2_coef_df["stars"] = stage2_coef_df["pvalue"].map(p_to_stars)
    stage2_coef_df["abs_coef"] = stage2_coef_df["coef"].abs()
    stage2_coef_df = stage2_coef_df.sort_values("abs_coef", ascending=False)

heckman_summary_df = pd.DataFrame(heckman_summary_rows)
heckman_summary_out = FEATURE_DIR / "person_model_heckman_two_stage_summary.csv"
heckman_summary_df.to_csv(heckman_summary_out, index=False)
stage1_out = FEATURE_DIR / "person_model_heckman_stage1_coefficients.csv"
stage2_out = FEATURE_DIR / "person_model_heckman_stage2_coefficients.csv"
stage1_coef_df.to_csv(stage1_out, index=False)
stage2_coef_df.to_csv(stage2_out, index=False)

print("\nSaved Heckman-style summary:", heckman_summary_out)
if not heckman_summary_df.empty:
    print(heckman_summary_df.to_string(index=False))
print("Saved Heckman stage-1 coefficients:", stage1_out)
print("Saved Heckman stage-2 coefficients:", stage2_out)

# Heckman summary figure
heckman_fig_out = FIG_DIR / "person_model_heckman_two_stage_results.png"
if not heckman_summary_df.empty:
    row = heckman_summary_df.iloc[0]
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    fig.suptitle(
        "Heckman-Style Two-Stage Results\nSelection: joined team | Outcome: funded team (conditional on selection)",
        fontsize=15,
        fontweight="bold",
    )

    # Panel A: stage metrics
    metric_names = [
        "Sel AUC",
        "Sel AUPRC",
        "Out AUC\n(selected)",
        "Out AUPRC\n(selected)",
        "Out Acc\n(selected)",
        "Out F1\n(selected)",
    ]
    metric_vals = [
        row.get("selection_auc", np.nan),
        row.get("selection_auprc", np.nan),
        row.get("outcome_auc_selected_only", np.nan),
        row.get("outcome_auprc_selected_only", np.nan),
        row.get("outcome_accuracy_selected_only", np.nan),
        row.get("outcome_f1_selected_only", np.nan),
    ]
    ax1.bar(metric_names, metric_vals, color=["#1b9e77", "#66a61e", "#7570b3", "#e7298a", "#1f78b4", "#d95f02"])
    ax1.set_ylim(0, 1.0)
    ax1.set_title("Stage Metrics")
    for i, v in enumerate(metric_vals):
        if pd.notna(v):
            ax1.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=10)

    # Panel B: stage-1 coefficients
    s1 = stage1_coef_df[stage1_coef_df["feature"] != "const"].copy()
    s1 = s1.sort_values("abs_coef", ascending=False).head(12).sort_values("coef")
    colors1 = ["#00c853" if c >= 0 else "#ff1744" for c in s1["coef"]]
    ax2.barh(s1["feature"], s1["coef"], color=colors1)
    ax2.axvline(0, color="black", lw=1)
    ax2.set_title("Stage 1 (Selection Probit) Coefficients")
    ax2.set_xlabel("Coefficient")
    for yi, (_, r) in enumerate(s1.reset_index(drop=True).iterrows()):
        x = r["coef"] + (0.02 if r["coef"] >= 0 else -0.02)
        ha = "left" if r["coef"] >= 0 else "right"
        ax2.text(x, yi, p_to_stars(r["pvalue"]), va="center", ha=ha, fontsize=10)

    # Panel C: IMR summary
    imr_coef = row.get("imr_coef_stage2", np.nan)
    imr_p = row.get("imr_pvalue_stage2", np.nan)
    stars = row.get("imr_significance", "")
    stage2_type = row.get("stage2_model_type", "unknown")
    txt = (
        f"Stage 2 model: {stage2_type}\n\n"
        f"IMR coefficient: {imr_coef:.3f}\n"
        f"IMR p-value: {imr_p:.3f}\n"
        f"IMR significance: {stars if stars else 'ns'}\n\n"
        "Interpretation:\n"
        "IMR captures selection-bias correction term.\n"
        "Non-significant IMR suggests limited selection correction\n"
        "signal in this specification."
    )
    ax3.axis("off")
    ax3.text(0.02, 0.98, txt, va="top", ha="left", fontsize=11)
    ax3.set_title("Selection-Correction (IMR) Summary", loc="left")

    # Panel D: stage-2 coefficients
    s2 = stage2_coef_df[stage2_coef_df["feature"] != "const"].copy()
    s2 = s2.sort_values("abs_coef", ascending=False).head(12).sort_values("coef")
    colors2 = ["#00c853" if c >= 0 else "#ff1744" for c in s2["coef"]]
    ax4.barh(s2["feature"], s2["coef"], color=colors2)
    ax4.axvline(0, color="black", lw=1)
    ax4.set_title("Stage 2 (Funded Outcome + IMR) Coefficients")
    ax4.set_xlabel("Coefficient")
    for yi, (_, r) in enumerate(s2.reset_index(drop=True).iterrows()):
        x = r["coef"] + (0.02 if r["coef"] >= 0 else -0.02)
        ha = "left" if r["coef"] >= 0 else "right"
        ax4.text(x, yi, p_to_stars(r["pvalue"]), va="center", ha=ha, fontsize=10)

    fig.savefig(heckman_fig_out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved Heckman figure:", heckman_fig_out)
