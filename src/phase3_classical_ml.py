"""
Phase 3: Classical ML Baselines
Multi-Target Protein Binding Predictor
This work used Proteinbase by Adaptyv Bio under ODC-BY license
"""

import sys
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    matthews_corrcoef
)
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = Path(__file__).parent.parent
FEAT_DIR     = PROJECT_ROOT / "features"
MODEL_DIR    = PROJECT_ROOT / "models"
OUTPUT_DIR   = PROJECT_ROOT / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

def log(msg):
    print(msg, flush=True)

# ── Load data ─────────────────────────────────────────────────────────────────

log("Loading feature matrix...")
fm       = pd.read_parquet(FEAT_DIR / "feature_matrix.parquet")
col_meta = pd.read_csv(FEAT_DIR / "feature_columns.csv")
feat_cols = col_meta["column"].tolist()

train_df = fm[fm["split"] == "train"].reset_index(drop=True)
val_df   = fm[fm["split"] == "val"].reset_index(drop=True)
test_df  = fm[fm["split"] == "test"].reset_index(drop=True)

X_train, y_train = train_df[feat_cols].values, train_df["binding_label"].values
X_val,   y_val   = val_df[feat_cols].values,   val_df["binding_label"].values
X_test,  y_test  = test_df[feat_cols].values,  test_df["binding_label"].values

pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
log(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
log(f"  Positive class weight: {pos_weight:.2f}")
log(f"  Features: {len(feat_cols)}")

# Check GPU availability
import torch
USE_GPU = torch.cuda.is_available()
log(f"  GPU available: {USE_GPU} ({'RTX 4070 SUPER' if USE_GPU else 'CPU only'})")

# ── Evaluation helper ─────────────────────────────────────────────────────────

results = []

def evaluate(name, model, X, y, split="val", print_result=True):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    else:
        score = model.decision_function(X)
        proba = (score - score.min()) / (score.max() - score.min() + 1e-9)
    preds = (proba >= 0.5).astype(int)
    auroc = roc_auc_score(y, proba)
    auprc = average_precision_score(y, proba)
    f1    = f1_score(y, preds, zero_division=0)
    mcc   = matthews_corrcoef(y, preds)
    r = {"model": name, "split": split, "auroc": auroc, "auprc": auprc, "f1": f1, "mcc": mcc}
    if print_result:
        log(f"  ✓ {name:<25} AUROC={auroc:.4f}  AUPRC={auprc:.4f}  F1={f1:.4f}  MCC={mcc:.4f}")
    return r

# ── 1. Logistic Regression ────────────────────────────────────────────────────

log("\n[1/7] Logistic Regression...")
best_lr_result = None
for penalty, C in [("l2", 1.0), ("l2", 0.1), ("l1", 1.0)]:
    solver = "saga" if penalty == "l1" else "lbfgs"
    m = LogisticRegression(penalty=penalty, C=C, solver=solver,
                           class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED)
    m.fit(X_train, y_train)
    r = evaluate(f"LR_{penalty}_C{C}", m, X_val, y_val)
    results.append(r)
    if best_lr_result is None or r["auprc"] > best_lr_result["auprc"]:
        best_lr_result = r
        joblib.dump(m, MODEL_DIR / "lr_best.pkl")

# ── 2. Random Forest ──────────────────────────────────────────────────────────

log("\n[2/7] Random Forest (500 trees)...")
rf = RandomForestClassifier(n_estimators=500, min_samples_split=5,
                             class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED)
rf.fit(X_train, y_train)
results.append(evaluate("RandomForest", rf, X_val, y_val))
joblib.dump(rf, MODEL_DIR / "rf.pkl")

# ── 3. Extra Trees ────────────────────────────────────────────────────────────

log("\n[3/7] Extra Trees (500 trees)...")
et = ExtraTreesClassifier(n_estimators=500, min_samples_split=5,
                           class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED)
et.fit(X_train, y_train)
results.append(evaluate("ExtraTrees", et, X_val, y_val))
joblib.dump(et, MODEL_DIR / "et.pkl")

# ── 4. XGBoost — Optuna + GPU ─────────────────────────────────────────────────

log("\n[4/7] XGBoost (Optuna 80 trials, GPU)...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

xgb_device = "cuda" if USE_GPU else "cpu"

def xgb_objective(trial):
    params = dict(
        n_estimators     = trial.suggest_int("n_estimators", 200, 1000),
        max_depth        = trial.suggest_int("max_depth", 3, 8),
        learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample        = trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_weight = trial.suggest_int("min_child_weight", 1, 10),
        reg_alpha        = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        reg_lambda       = trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        scale_pos_weight = pos_weight,
        eval_metric      = "aucpr",
        random_state     = RANDOM_SEED,
        n_jobs           = 1,
        tree_method      = "hist",
        device           = xgb_device,
    )
    scores = []
    for tr_idx, vl_idx in skf.split(X_train, y_train):
        m = xgb.XGBClassifier(**params)
        m.fit(X_train[tr_idx], y_train[tr_idx], verbose=False)
        p = m.predict_proba(X_train[vl_idx])[:, 1]
        scores.append(average_precision_score(y_train[vl_idx], p))
    return np.mean(scores)

study_xgb = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
study_xgb.optimize(xgb_objective, n_trials=80, show_progress_bar=True)

log(f"  XGBoost best CV AUPRC: {study_xgb.best_value:.4f}")
log(f"  Best params: {study_xgb.best_params}")

best_xgb_params = dict(**study_xgb.best_params,
                        scale_pos_weight=pos_weight, eval_metric="aucpr",
                        random_state=RANDOM_SEED, n_jobs=-1,
                        tree_method="hist", device=xgb_device)

xgb_model = xgb.XGBClassifier(**best_xgb_params)
xgb_model.fit(X_train, y_train)
results.append(evaluate("XGBoost", xgb_model, X_val, y_val))
joblib.dump(xgb_model, MODEL_DIR / "xgb_best.pkl")

# XGBoost + SMOTE
log("  Training XGBoost + SMOTE...")
sm = SMOTE(random_state=RANDOM_SEED, k_neighbors=min(5, (y_train==1).sum()-1))
X_sm, y_sm = sm.fit_resample(X_train, y_train)
xgb_smote = xgb.XGBClassifier(**{**best_xgb_params, "scale_pos_weight": 1.0})
xgb_smote.fit(X_sm, y_sm)
results.append(evaluate("XGBoost_SMOTE", xgb_smote, X_val, y_val))

# ── 5. LightGBM — Optuna + GPU ────────────────────────────────────────────────

log("\n[5/7] LightGBM (Optuna 80 trials, GPU)...")
lgb_device = "gpu" if USE_GPU else "cpu"

def lgb_objective(trial):
    params = dict(
        n_estimators      = trial.suggest_int("n_estimators", 200, 1000),
        max_depth         = trial.suggest_int("max_depth", 3, 8),
        learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        num_leaves        = trial.suggest_int("num_leaves", 20, 100),
        subsample         = trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_samples = trial.suggest_int("min_child_samples", 5, 50),
        reg_alpha         = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        is_unbalance      = True,
        random_state      = RANDOM_SEED,
        n_jobs            = 1,
        device            = lgb_device,
        verbose           = -1,
    )
    scores = []
    for tr_idx, vl_idx in skf.split(X_train, y_train):
        m = lgb.LGBMClassifier(**params)
        m.fit(X_train[tr_idx], y_train[tr_idx])
        p = m.predict_proba(X_train[vl_idx])[:, 1]
        scores.append(average_precision_score(y_train[vl_idx], p))
    return np.mean(scores)

study_lgb = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
study_lgb.optimize(lgb_objective, n_trials=80, show_progress_bar=True)

log(f"  LightGBM best CV AUPRC: {study_lgb.best_value:.4f}")

best_lgb_params = dict(**study_lgb.best_params, is_unbalance=True,
                        random_state=RANDOM_SEED, n_jobs=-1,
                        device=lgb_device, verbose=-1)
lgb_model = lgb.LGBMClassifier(**best_lgb_params)
lgb_model.fit(X_train, y_train)
results.append(evaluate("LightGBM", lgb_model, X_val, y_val))
joblib.dump(lgb_model, MODEL_DIR / "lgb_best.pkl")

# ── 6. SVM ────────────────────────────────────────────────────────────────────

log("\n[6/7] SVM (RBF kernel)...")
svm = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
          probability=True, random_state=RANDOM_SEED)
svm.fit(X_train, y_train)
results.append(evaluate("SVM_RBF", svm, X_val, y_val))
joblib.dump(svm, MODEL_DIR / "svm.pkl")

# ── 7. Gaussian Naive Bayes ───────────────────────────────────────────────────

log("\n[7/7] Gaussian Naive Bayes...")
gnb = GaussianNB()
gnb.fit(X_train, y_train)
results.append(evaluate("GaussianNB", gnb, X_val, y_val))
joblib.dump(gnb, MODEL_DIR / "gnb.pkl")

# ── Results summary ───────────────────────────────────────────────────────────

results_df = pd.DataFrame(results).sort_values("auprc", ascending=False)
log("\n══ VALIDATION RESULTS (sorted by AUPRC) ═══════════════════════════════")
log(f"  {'Model':<25} {'AUROC':>7} {'AUPRC':>7} {'F1':>7} {'MCC':>7}")
log(f"  {'-'*55}")
for _, row in results_df.iterrows():
    log(f"  {row['model']:<25} {row['auroc']:>7.4f} {row['auprc']:>7.4f} {row['f1']:>7.4f} {row['mcc']:>7.4f}")

# ── Test set — top 4 models ───────────────────────────────────────────────────

model_map = {
    "XGBoost": xgb_model, "XGBoost_SMOTE": xgb_smote,
    "LightGBM": lgb_model, "RandomForest": rf,
    "ExtraTrees": et, "SVM_RBF": svm,
    "GaussianNB": gnb,
    **{k: joblib.load(MODEL_DIR / "lr_best.pkl") for k in ["LR_l2_C1.0", "LR_l2_C0.1", "LR_l1_C1.0"]},
}

log("\n══ TEST SET RESULTS (top 4 by val AUPRC) ══════════════════════════════")
test_results = []
for name in results_df.head(4)["model"].tolist():
    if name in model_map:
        r = evaluate(name, model_map[name], X_test, y_test, split="test")
        test_results.append(r)

# ── Per-target evaluation ─────────────────────────────────────────────────────

best_name  = results_df.iloc[0]["model"]
best_model = model_map.get(best_name, xgb_model)
log(f"\n══ PER-TARGET PERFORMANCE ({best_name}) ════════════════════════════════")

per_target = []
for target, grp in val_df.groupby("target"):
    grp = grp.reset_index(drop=True)
    Xt = grp[feat_cols].values
    yt = grp["binding_label"].values
    if yt.sum() == 0 or yt.sum() == len(yt):
        continue
    proba = best_model.predict_proba(Xt)[:, 1]
    per_target.append({
        "target": target, "n": len(yt), "binders": int(yt.sum()),
        "auroc": roc_auc_score(yt, proba),
        "auprc": average_precision_score(yt, proba),
    })

per_target_df = pd.DataFrame(per_target).sort_values("auroc", ascending=False)
log(f"  {'Target':<35} {'n':>5} {'bind':>5} {'AUROC':>7} {'AUPRC':>7}")
log(f"  {'-'*65}")
for _, row in per_target_df.iterrows():
    log(f"  {row['target']:<35} {int(row['n']):>5} {int(row['binders']):>5} {row['auroc']:>7.4f} {row['auprc']:>7.4f}")

# ── SHAP feature importance ───────────────────────────────────────────────────

log("\nComputing SHAP values (XGBoost)...")
explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_val)
mean_shap   = np.abs(shap_values).mean(axis=0)

shap_df = pd.DataFrame({
    "feature": feat_cols,
    "mean_abs_shap": mean_shap,
    "group": col_meta["group"].values,
}).sort_values("mean_abs_shap", ascending=False)

log("\n  Top 20 features by SHAP importance:")
log(f"  {'Feature':<45} {'Group':<20} {'SHAP':>8}")
log(f"  {'-'*75}")
for _, row in shap_df.head(20).iterrows():
    log(f"  {row['feature']:<45} {row['group']:<20} {row['mean_abs_shap']:>8.4f}")

# ── Plots ─────────────────────────────────────────────────────────────────────

log("\nGenerating plots...")

# SHAP top 30
fig, ax = plt.subplots(figsize=(10, 9))
top30 = shap_df.head(30)
color_map = {
    "dipeptide": "#4C72B0", "boltz2": "#DD8452", "precomputed_seq": "#55A868",
    "physicochemical": "#C44E52", "aa_composition": "#8172B3",
    "design_method": "#937860", "other": "#aaaaaa",
}
bar_colors = [color_map.get(g, "#aaaaaa") for g in top30["group"]]
ax.barh(top30["feature"][::-1], top30["mean_abs_shap"][::-1], color=bar_colors[::-1])
ax.set_xlabel("Mean |SHAP value|")
ax.set_title(f"Top 30 Features by SHAP — XGBoost")
handles = [plt.Rectangle((0,0),1,1, color=v) for v in color_map.values()]
ax.legend(handles, color_map.keys(), loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "shap_top30.png", dpi=150)
plt.close()

# Model comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
rdf_plot = results_df[~results_df["model"].str.contains("SMOTE|LR_l")].copy()
rdf_plot["model_short"] = rdf_plot["model"].str.replace("_", "\n")
palette = sns.color_palette("tab10", len(rdf_plot))
for ax, metric in zip(axes, ["auroc", "auprc"]):
    bars = ax.bar(rdf_plot["model_short"], rdf_plot[metric], color=palette)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(f"Validation {metric.upper()}", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    for bar, val in zip(bars, rdf_plot[metric]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
plt.suptitle("Classical ML Baseline Comparison — Validation Set", fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=150)
plt.close()

# Per-target heatmap
if len(per_target_df) > 1:
    fig, ax = plt.subplots(figsize=(7, max(4, len(per_target_df) * 0.45)))
    hdata = per_target_df.set_index("target")[["auroc", "auprc"]].sort_values("auroc")
    sns.heatmap(hdata, annot=True, fmt=".3f", cmap="RdYlGn",
                vmin=0, vmax=1, ax=ax, linewidths=0.5)
    ax.set_title(f"Per-Target Performance — {best_name} (Val)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "per_target_heatmap.png", dpi=150)
    plt.close()

# ── Save ──────────────────────────────────────────────────────────────────────

results_df.to_csv(OUTPUT_DIR / "val_results.csv", index=False)
pd.DataFrame(test_results).to_csv(OUTPUT_DIR / "test_results.csv", index=False)
shap_df.to_csv(OUTPUT_DIR / "shap_importance.csv", index=False)
per_target_df.to_csv(OUTPUT_DIR / "per_target_results.csv", index=False)

log("\nSaved: outputs/val_results.csv  test_results.csv  shap_importance.csv")
log("Saved: outputs/shap_top30.png  model_comparison.png  per_target_heatmap.png")
log("\n✓ Phase 3 complete.")
