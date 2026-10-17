"""
Evaluation -- stratified k-fold cross-validation on the development set (week 4).

Up to week 3 the pipeline judged a model on ONE train/test split. That number depends
heavily on which rows happened to land in the test part: change the seed and it moves.
From week 4, models are judged by k-fold cross-validation instead:
  - the development set is split into k folds (stratified: each fold keeps the same
    share of reoffenders as the whole set);
  - the whole Pipeline (preprocessing + model) is fit k times, each time on k-1 folds,
    and scored on the fold it did not see;
  - we report every fold's score, their mean AND their standard deviation. The std is
    the honest "how much would this number move with different data" -- a mean alone
    hides it.

The final test set is not used anywhere in this file -- it stays locked.
See Practical/W4/notebooks/03_cross_validation.ipynb for the full walkthrough.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate


def cross_validate_pipeline(pipeline, X, y, cv, scoring: str = "accuracy", n_jobs: int = 1):
    """
    Fits a fresh copy of `pipeline` on each fold's training part and scores it on that
    fold's validation part. Because `pipeline` contains the preprocessing too, imputation
    medians, encoder statistics and scaler means are re-learned inside every fold -- the
    validation fold never influences its own preprocessing.

    Returns (fold_scores, y_oof):
      - fold_scores: one row per fold -- train score, validation score, and the gap between
        them (a large, consistent gap = overfitting).
      - y_oof: out-of-fold predictions. Every row gets a prediction from the one fold model
        that did NOT train on it, so all of them are "unseen" -- what the classification
        report and fairness check are computed on (more rows, and the locked test set stays
        untouched). Taken from the same fits as the scores, so nothing is fitted twice.
    """
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, return_train_score=True,
                            return_estimator=True, return_indices=True, n_jobs=n_jobs)
    fold_scores = pd.DataFrame({
        "fold": range(1, len(scores["test_score"]) + 1),
        "train": scores["train_score"],
        "validation": scores["test_score"],
    })
    fold_scores["gap"] = fold_scores["train"] - fold_scores["validation"]

    y_oof = np.empty(len(X), dtype=np.asarray(y).dtype)
    for model, val_idx in zip(scores["estimator"], scores["indices"]["test"]):
        y_oof[val_idx] = model.predict(X.iloc[val_idx])
    return fold_scores, y_oof


def cv_report(fold_scores: pd.DataFrame, scoring: str = "accuracy") -> str:
    """Per-fold table + mean +/- std, as text (printed, and saved to results/)."""
    lines = [
        f"Cross-validation ({len(fold_scores)} stratified folds, metric: {scoring})",
        "",
        fold_scores.to_string(index=False, float_format=lambda v: f"{v:.3f}"),
        "",
    ]
    for col in ["train", "validation", "gap"]:
        sign = "+" if col == "gap" else ""
        lines.append(f"{col.capitalize():<11s} mean = {fold_scores[col].mean():{sign}.3f}   "
                     f"std = {fold_scores[col].std(ddof=1):.3f}")
    text = "\n".join(lines)
    print(text)
    return text


def oof_classification_report(y_true, y_pred) -> str:
    """Classification report on the out-of-fold predictions."""
    text = "Classification report (out-of-fold predictions, development set):\n" + \
        classification_report(y_true, y_pred, zero_division=0)
    print(text)
    return text


def fairness_report(y_true, y_pred, extras: pd.DataFrame, sensitive_attr: str = "race") -> str:
    """
    Deliberately simple fairness check -- not a substitute for a real audit, just enough
    to show that "accurate" and "fair" are not the same thing.

    For each race group, the false positive rate (share of people who did NOT reoffend
    but were predicted to) for:
        - our own model (out-of-fold predictions on the development set)
        - COMPAS's own risk score (score_text != "Low" counts as a "high risk"
          prediction), on the same rows, for comparison
    """
    df = extras.copy()
    df["y_true"] = pd.Series(y_true).values
    df["y_pred_model"] = y_pred
    df["y_pred_compas"] = (df["score_text"] != "Low").astype(int)

    lines = [
        "False positive rate by race (development set, out-of-fold)",
        "(share of people who did NOT reoffend, but were predicted to)",
        "",
    ]

    for label, col in [("Our model", "y_pred_model"), ("COMPAS's own score", "y_pred_compas")]:
        lines.append(f"  {label}:")
        for group, g in df.groupby(sensitive_attr):
            negatives = g[g["y_true"] == 0]
            if len(negatives) == 0:
                continue
            fpr = (negatives[col] == 1).mean()
            lines.append(f"    {group:<20s} FPR = {fpr:.2f}  (n={len(negatives)})")
        lines.append("")

    text = "\n".join(lines)
    print(text)
    return text


def corrected_paired_test(scores_a, scores_b, test_train_ratio: float) -> dict:
    """
    Week 6: is model A really better than model B, or is the gap just fold-to-fold noise?

    Corrected paired t-test (Nadeau & Bengio, 2003) for two models scored on the SAME CV folds.
    The folds share most of their training rows, so their scores are correlated and a plain
    paired t-test is overconfident. The variance of the per-fold differences is therefore
    inflated: 1/k becomes 1/k + test_train_ratio, where test_train_ratio is
    (validation rows / training rows) per fold = 1 / (n_splits - 1) for k-fold CV.

    Use it on the per-fold `validation` columns of two nested-CV runs (same folds), e.g.
        corrected_paired_test(nested_rf["validation"], nested_ada["validation"], 1 / (n_splits - 1))

    Returns {"mean_diff": mean(a - b), "wins": folds where a > b, "t": t statistic,
             "p_value": two-sided p-value with k - 1 degrees of freedom}.
    A "winner" is only worth calling when p_value is small (conventionally < 0.05).
    """
    diff = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    k = len(diff)
    se = np.sqrt((1 / k + test_train_ratio) * diff.var(ddof=1))
    t = float(diff.mean() / se) if se > 0 else 0.0
    p_value = float(2 * stats.t.sf(abs(t), df=k - 1))
    return {"mean_diff": float(diff.mean()), "wins": f"{int((diff > 0).sum())}/{k}", "t": t, "p_value": p_value}
