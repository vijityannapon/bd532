"""Training, validation, experiment tracking and model export (Sections 3.1-3.5).

Usage
-----
    python -m credit_risk.train                       # all experiments, real data if present
    python -m credit_risk.train --data data/sample/application_sample.csv
    python -m credit_risk.train --experiments v1_logreg_basic v5_hgb_tuned --no-cv

Validation methodology
----------------------
* Stratified split 70 / 15 / 15 (train / validation / test), seed 42.
* Every model version is fitted on *train* and compared on *validation*
  (ROC-AUC primary, PR-AUC, KS, Brier, expected cost) -> logged to MLflow.
* The decision threshold is chosen on validation by minimising expected cost
  (FN cost 5 : FP cost 1).
* The selected model gets 5-fold stratified CV on train (stability) and is
  evaluated ONCE on the untouched *test* set.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from . import config, data, evaluate, explain, fairness, features
from .model import EXPERIMENTS, ExperimentSpec, build_pipeline, get_experiment, reweighing_weights

AUC_TOLERANCE = 0.003  # models within this ROC-AUC of the best are "equivalent"
BRIER_TOLERANCE = 1.10  # calibration gate: Brier must be within 10% of the best model
# Types inside our own sklearn models that MLflow's skops serializer must be told to trust.
SKOPS_TRUSTED = ["numpy.dtype", "sklearn.tree._tree.Tree",
                 "sklearn._loss.link.Interval", "sklearn._loss.link.LogitLink",
                 "sklearn._loss.loss.HalfBinomialLoss", "sklearn._loss._loss.CyHalfBinomialLoss",
                 "sklearn.ensemble._hist_gradient_boosting.binning._BinMapper",
                 "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]


def split(X, y, s):
    X_tr, X_tmp, y_tr, y_tmp, s_tr, s_tmp = train_test_split(
        X, y, s, test_size=0.30, stratify=y, random_state=config.RANDOM_STATE)
    X_va, X_te, y_va, y_te, s_va, s_te = train_test_split(
        X_tmp, y_tmp, s_tmp, test_size=0.50, stratify=y_tmp, random_state=config.RANDOM_STATE)
    return (X_tr, y_tr, s_tr), (X_va, y_va, s_va), (X_te, y_te, s_te)


def spec_columns(spec: ExperimentSpec) -> list[str]:
    return features.feature_columns(spec.engineered)


def fit_spec(spec: ExperimentSpec, X_tr, y_tr, s_tr):
    pipe = build_pipeline(spec)
    fit_kw = {}
    if spec.reweigh:
        fit_kw["clf__sample_weight"] = reweighing_weights(y_tr, s_tr["gender"])
    t0 = time.perf_counter()
    pipe.fit(X_tr[spec_columns(spec)], y_tr, **fit_kw)
    return pipe, time.perf_counter() - t0


def latency_ms(pipe, X_one: pd.DataFrame, repeats: int = 100) -> float:
    pipe.predict_proba(X_one)  # warm-up
    t0 = time.perf_counter()
    for _ in range(repeats):
        pipe.predict_proba(X_one)
    return (time.perf_counter() - t0) / repeats * 1000


def model_size_mb(obj) -> float:
    with tempfile.NamedTemporaryFile(suffix=".joblib") as f:
        joblib.dump(obj, f.name)
        return os.path.getsize(f.name) / 1e6


def training_profile(X: pd.DataFrame) -> dict:
    """Reference statistics stored with the model for input validation & drift checks."""
    prof = {"numeric": {}, "categorical": {}}
    for c in X.columns:
        if c in config.CATEGORICAL_FEATURES:
            prof["categorical"][c] = sorted(X[c].dropna().astype(str).unique().tolist())
        else:
            q = X[c].quantile([0.01, 0.5, 0.99])
            prof["numeric"][c] = {"mean": float(X[c].mean()), "std": float(X[c].std()),
                                  "p01": float(q.iloc[0]), "p50": float(q.iloc[1]),
                                  "p99": float(q.iloc[2]), "missing_rate": float(X[c].isna().mean())}
    return prof


def run(data_path=None, nrows=None, experiment_names=None, do_cv=True, out_dir=config.REPORTS_DIR):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"; fig_dir.mkdir(exist_ok=True)

    raw = data.load_raw(data_path, nrows)
    synthetic = data.is_synthetic(raw)
    X, y, s = features.prepare_dataset(raw, engineered=True)
    (X_tr, y_tr, s_tr), (X_va, y_va, s_va), (X_te, y_te, s_te) = split(X, y, s)
    print(f"Data: {len(X):,} rows ({'synthetic' if synthetic else 'Home Credit'}), "
          f"default rate {y.mean():.2%} | train {len(X_tr):,} / val {len(X_va):,} / test {len(X_te):,}")

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    specs = [get_experiment(n) for n in experiment_names] if experiment_names else EXPERIMENTS

    results, fitted = [], {}
    for spec in specs:
        with mlflow.start_run(run_name=spec.name) as run_ctx:
            pipe, fit_s = fit_spec(spec, X_tr, y_tr, s_tr)
            cols = spec_columns(spec)
            p_va = pipe.predict_proba(X_va[cols])[:, 1]
            thr = evaluate.select_threshold(y_va, p_va)
            m = evaluate.classification_metrics(y_va, p_va, thr)
            fair = fairness.audit(y_va, p_va, s_va, thr)
            m.update({
                "threshold": thr,
                "fit_seconds": fit_s,
                "latency_ms_single": latency_ms(pipe, X_va[cols].iloc[[0]]),
                "model_size_mb": model_size_mb(pipe),
                "gender_fpr_diff": fair["gender"]["summary"]["fpr_diff"],
                "gender_disparate_impact": fair["gender"]["summary"]["disparate_impact_ratio"],
                "age_disparate_impact": fair["age_group"]["summary"]["disparate_impact_ratio"],
            })
            mlflow.set_tags({"model_type": spec.model_type, "engineered": spec.engineered,
                             "reweigh": spec.reweigh, "data": "synthetic" if synthetic else "home_credit",
                             "mlflow.note.content": spec.description})
            mlflow.log_params({f"clf_{k}": v for k, v in spec.params.items()} |
                              {"n_features": len(cols), "train_rows": len(X_tr)})
            mlflow.log_metrics({f"val_{k}": v for k, v in m.items()})
            for attr, res in fair.items():
                p = fig_dir / f"{spec.name}_fairness_{attr}.csv"
                res["by_group"].to_csv(p)
                mlflow.log_artifact(str(p), artifact_path="fairness")
            mlflow.sklearn.log_model(pipe, name="model", input_example=X_va[cols].head(3),
                                     skops_trusted_types=SKOPS_TRUSTED)
            results.append({"name": spec.name, "run_id": run_ctx.info.run_id,
                            "description": spec.description, **m})
            fitted[spec.name] = (spec, pipe, p_va)
            print(f"  {spec.name:<26} AUC={m['roc_auc']:.4f} PR-AUC={m['pr_auc']:.4f} "
                  f"KS={m['ks']:.3f} cost={m['expected_cost']:.4f} "
                  f"lat={m['latency_ms_single']:.1f}ms size={m['model_size_mb']:.2f}MB")

    comp = pd.DataFrame(results).set_index("name")
    # ---- model selection rule --------------------------------------------
    # 1) Calibration gate: the UI shows a probability of default, so it must be
    #    trustworthy. Models whose Brier score is >10% worse than the best are
    #    excluded (class-weighted models typically fail this: they inflate PD).
    brier_limit = comp["brier"].min() * BRIER_TOLERANCE
    eligible = comp[comp["brier"] <= brier_limit]
    rejected = sorted(set(comp.index) - set(eligible.index))
    # 2) Discrimination: highest ROC-AUC; models within AUC_TOLERANCE are equivalent.
    best_auc = eligible["roc_auc"].max()
    candidates = eligible[eligible["roc_auc"] >= best_auc - AUC_TOLERANCE]
    # 3) Tie-break on fairness (gender FPR gap), then latency.
    best_name = candidates.sort_values(["gender_fpr_diff", "latency_ms_single"]).index[0]
    rationale = (f"Calibration gate (Brier <= {brier_limit:.4f}) excluded {rejected or 'none'}. "
                 f"Highest eligible validation ROC-AUC is {best_auc:.4f} ({eligible['roc_auc'].idxmax()}). "
                 f"Models within {AUC_TOLERANCE} AUC are treated as equivalent: "
                 f"{list(candidates.index)}. Among them the one with the smallest gender FPR gap "
                 f"(then lowest latency) is selected: **{best_name}**.")
    print("Selected:", best_name)
    spec, best_pipe, p_va_best = fitted[best_name]
    cols = spec_columns(spec)
    thr = float(comp.loc[best_name, "threshold"])

    cv = {}
    if do_cv:
        skf = StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE)
        if spec.reweigh:  # sample weights can't be routed through cross_val_score simply
            scores = []
            for tr_i, va_i in skf.split(X_tr, y_tr):
                pp, _ = fit_spec(spec, X_tr.iloc[tr_i], y_tr.iloc[tr_i], s_tr.iloc[tr_i])
                from sklearn.metrics import roc_auc_score
                scores.append(roc_auc_score(y_tr.iloc[va_i], pp.predict_proba(X_tr.iloc[va_i][cols])[:, 1]))
            scores = np.array(scores)
        else:
            scores = cross_val_score(build_pipeline(spec), X_tr[cols], y_tr, cv=skf, scoring="roc_auc", n_jobs=1)
        cv = {"cv_roc_auc_mean": float(scores.mean()), "cv_roc_auc_std": float(scores.std()),
              "cv_folds": scores.round(4).tolist()}
        print(f"  5-fold CV ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")

    # ---- final test evaluation (touched once) -----------------------------
    p_te = best_pipe.predict_proba(X_te[cols])[:, 1]
    test_metrics = evaluate.classification_metrics(y_te, p_te, thr)
    test_fair = fairness.audit(y_te, p_te, s_te, thr)
    bands = pd.Series([evaluate.decision_band(p, thr) for p in p_te])
    band_table = pd.DataFrame({"band": bands, "y": y_te.to_numpy()}).groupby("band")["y"].agg(
        share="size", actual_default_rate="mean")
    band_table["share"] = band_table["share"] / len(bands)
    with mlflow.start_run(run_id=comp.loc[best_name, "run_id"]):
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()} |
                           {k: v for k, v in cv.items() if k != "cv_folds"})
        mlflow.set_tag("selected_model", "true")

    # ---- explainability ---------------------------------------------------
    background = X_tr[cols].sample(min(200, len(X_tr)), random_state=config.RANDOM_STATE)
    expl = explain.CreditExplainer(best_pipe, background)
    X_shap = X_te[cols].sample(min(2000, len(X_te)), random_state=config.RANDOM_STATE)
    shap_imp = expl.global_importance(X_shap)
    explain.plot_global(expl, X_shap, fig_dir)
    perm_n = min(4000, len(X_te))
    perm = permutation_importance(best_pipe, X_te[cols].iloc[:perm_n], y_te.iloc[:perm_n],
                                  scoring="roc_auc", n_repeats=5, random_state=config.RANDOM_STATE, n_jobs=1)
    perm_imp = pd.Series(perm.importances_mean, index=cols).sort_values(ascending=False)

    # Three representative test cases: lowest, median-band and highest risk.
    order = np.argsort(p_te)
    picks = {"low_risk_case": order[len(order) // 20], "borderline_case": int(np.argmin(np.abs(p_te - thr))),
             "high_risk_case": order[-max(1, len(order) // 50)]}
    cases = {}
    for label, i in picks.items():
        row = X_te[cols].iloc[[i]]
        pdv = float(p_te[i])
        cases[label] = {"applicant_index": int(X_te.index[i]), "actual_default": int(y_te.iloc[i]),
                        "probability_of_default": round(pdv, 4), "risk_score": evaluate.risk_score(pdv),
                        "decision_band": evaluate.decision_band(pdv, thr),
                        "inputs": {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
                                   for k, v in row.iloc[0].items()},
                        **expl.explain_applicant(row)}

    # ---- plots ------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for name, (sp, pp, pv) in fitted.items():
        fpr, tpr, _ = roc_curve(y_va, pv); ax[0].plot(fpr, tpr, label=f"{name} ({comp.loc[name,'roc_auc']:.3f})")
        pr, rc, _ = precision_recall_curve(y_va, pv); ax[1].plot(rc, pr, label=f"{name} ({comp.loc[name,'pr_auc']:.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8); ax[0].set(title="ROC (validation)", xlabel="FPR", ylabel="TPR")
    ax[1].axhline(y_va.mean(), color="k", ls="--", lw=0.8); ax[1].set(title="Precision-Recall (validation)", xlabel="Recall", ylabel="Precision")
    for a in ax: a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "roc_pr_comparison.png", dpi=130); plt.close(fig)

    frac, mean_pred = calibration_curve(y_te, p_te, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(mean_pred, frac, "o-", label=best_name); ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(title="Calibration (test)", xlabel="Mean predicted PD", ylabel="Observed default rate",
           xlim=(0, max(0.5, mean_pred.max())), ylim=(0, max(0.5, frac.max()))); ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir / "calibration_test.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    perm_imp.head(15)[::-1].plot.barh(ax=ax, color="#4C78A8")
    ax.set(title="Permutation importance (drop in test ROC-AUC)"); fig.tight_layout()
    fig.savefig(fig_dir / "permutation_importance.png", dpi=130); plt.close(fig)

    # ---- export bundle ----------------------------------------------------
    config.MODELS_DIR.mkdir(exist_ok=True)
    bundle = {
        "pipeline": best_pipe, "experiment": spec.name, "model_type": spec.model_type,
        "feature_columns": cols, "decline_threshold": thr,
        "review_threshold": thr * config.REVIEW_BAND_FACTOR,
        # Background rows are needed only by SHAP LinearExplainer; tree models do not
        # use them, so we avoid shipping training records inside the model file.
        "background": background if spec.model_type == "logreg" else None, "training_profile": training_profile(X_tr[cols]),
        "metrics": {"validation": comp.loc[best_name].drop(["run_id", "description"]).to_dict(),
                    "test": test_metrics, **cv},
        "mlflow_run_id": comp.loc[best_name, "run_id"],
        "data_source": "synthetic" if synthetic else "home_credit",
        "version": config.MODEL_VERSION_TAG,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    joblib.dump(bundle, config.MODEL_BUNDLE_PATH, compress=3)

    # ---- write report artefacts --------------------------------------------
    show = ["roc_auc", "pr_auc", "ks", "brier", "recall", "precision", "flag_rate", "expected_cost",
            "gender_fpr_diff", "gender_disparate_impact", "latency_ms_single", "model_size_mb", "fit_seconds"]
    comp[show].round(4).to_csv(out_dir / "model_comparison.csv")
    summary = {
        "data_source": bundle["data_source"], "rows": len(X), "default_rate": float(y.mean()),
        "selected_model": best_name, "selection_rationale": rationale, "threshold": thr,
        "validation_comparison": comp[show].round(4).to_dict(orient="index"),
        "cross_validation": cv, "test_metrics": test_metrics,
        "decision_bands_test": band_table.round(4).to_dict(orient="index"),
        "fairness_test": {a: {"summary": r["summary"], "by_group": r["by_group"].round(4).to_dict(orient="index")}
                          for a, r in test_fair.items()},
        "shap_global_importance": shap_imp.round(4).to_dict(),
        "permutation_importance": perm_imp.round(4).to_dict(),
        "example_explanations": cases,
    }
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    md = ["# Training results", "",
          f"Data: **{bundle['data_source']}**, {len(X):,} rows, default rate {y.mean():.2%}", "",
          "## Model versions (validation set)", "", comp[show].round(4).to_markdown(), "",
          "## Selection", "", rationale, ""]
    if cv:
        md += [f"5-fold CV ROC-AUC on train: {cv['cv_roc_auc_mean']:.4f} ± {cv['cv_roc_auc_std']:.4f}", ""]
    md += ["## Test set (held out)", "", pd.Series(test_metrics).round(4).to_frame("value").to_markdown(), "",
           "## Decision bands (test)", "", band_table.round(4).to_markdown(), ""]
    for a, r in test_fair.items():
        md += [f"## Fairness - {a} (test)", "", r["by_group"].round(4).to_markdown(), "",
               pd.Series(r["summary"]).round(4).to_frame("value").to_markdown(), ""]
    md += ["## Global importance", "", pd.DataFrame({"mean_abs_shap": shap_imp, "permutation_auc_drop": perm_imp}).round(4)
           .sort_values("mean_abs_shap", ascending=False).to_markdown(), ""]
    md += ["## Example explanations (3 test cases)", ""]
    for label, c in cases.items():
        md += [f"### {label}: PD={c['probability_of_default']:.1%}, score={c['risk_score']}, "
               f"band={c['decision_band']}, actual={c['actual_default']}", ""]
        md += [f"* {r['text']} (SHAP {r['shap_value']:+.3f})" for r in c["reasons"]] + [""]
    (out_dir / "results.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Model saved to {config.MODEL_BUNDLE_PATH}\nReport: {out_dir / 'results.md'}")
    return summary


if __name__ == "__main__":  # pragma: no cover
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--nrows", type=int, default=None)
    ap.add_argument("--experiments", nargs="*", default=None)
    ap.add_argument("--no-cv", action="store_true")
    a = ap.parse_args()
    run(a.data, a.nrows, a.experiments, do_cv=not a.no_cv)
