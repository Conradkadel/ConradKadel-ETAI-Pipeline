"""
Data diagnostics (week 3) -- the three techniques from 01_eda_introduction.ipynb:
    1. test_missingness_mechanism -> is a column's missingness random (MCAR) or not (MNAR)?
    2. flag_invalid_values        -> domain rules: impossible values become NaN
    3. find_duplicates            -> duplicates checked two ways (exact rows + repeated ids)

Generic on purpose: no COMPAS column names are written here. Every column list and
rule comes from the `diagnostics` section of config.yaml.
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def cramers_v(confusion_matrix: pd.DataFrame) -> float:
    """Bias-corrected Cramer's V effect size for a chi-square test of association."""
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2_corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    return float(np.sqrt(phi2_corr / min(k_corr - 1, r_corr - 1)))


def test_missingness_mechanism(df: pd.DataFrame, target_col: str, candidate_predictors: list) -> pd.DataFrame:
    """
    For `target_col`'s missing-value indicator, test association against each
    column in `candidate_predictors` via chi-square + Cramer's V.
    Returns one row per predictor, sorted by association strength (strongest first).

    How to read it: max Cramer's V well under 0.1 -> MCAR (safe to impute simply);
    around 0.2 or more -> MAR/MNAR (impute + add a `_was_missing` flag).
    """
    indicator = df[target_col].isna()   # True where the value is missing
    rows = []
    for predictor in candidate_predictors:
        # skip the column itself, and any column this dataset doesn't have
        if predictor == target_col or predictor not in df.columns:
            continue
        sub = pd.DataFrame({"missing": indicator, "predictor": df[predictor]}).dropna(subset=["predictor"])
        # a test needs at least 2 values on each side
        if sub["predictor"].nunique() < 2 or sub["missing"].nunique() < 2:
            continue
        table = pd.crosstab(sub["missing"], sub["predictor"])
        chi2, p, _, _ = chi2_contingency(table)
        v = cramers_v(table)
        rows.append({"predictor": predictor, "cramers_v": round(v, 3), "p_value": p, "n": len(sub)})
    result = pd.DataFrame(rows).sort_values("cramers_v", ascending=False).reset_index(drop=True)
    return result


def flag_invalid_values(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """
    Applies the domain rules from config.yaml (`diagnostics.validity_rules`), e.g.
        age: {min: 18, max: 100}
        juv_fel_count: {min: 0}          # either bound is optional
    Any value outside its rule is converted to NaN -- "impossible but not missing"
    is still missing. Changes `df` in place and returns a small report table.
    """
    report_rows = []
    for column, bounds in rules.items():
        if column not in df.columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")

        # start with "every value is valid", then apply each bound that exists
        valid = pd.Series(True, index=df.index)
        if "min" in bounds:
            valid &= numeric >= bounds["min"]
        if "max" in bounds:
            valid &= numeric <= bounds["max"]

        violations = numeric.notna() & ~valid   # present, but breaks the rule
        report_rows.append({
            "column": column,
            "rule": bounds,
            "violations": int(violations.sum()),
            "examples": sorted(numeric[violations].unique().tolist())[:6],
        })
        df.loc[violations, column] = np.nan
    return pd.DataFrame(report_rows)


def find_duplicates(df: pd.DataFrame, id_column: str = None) -> dict:
    """
    Duplicates checked two ways:
      - exact row duplicates (identical in every column)
      - repeated ids (same record entered twice, maybe with a typo in one field)
    Both are reported, because in general they can disagree.
    """
    result = {"exact_row_duplicates": int(df.duplicated().sum())}
    if id_column and id_column in df.columns:
        result["repeated_ids"] = int(df[id_column].duplicated().sum())
        # if both counts match, every repeated id is also an exact duplicate row
        result["same_rows"] = result["exact_row_duplicates"] == result["repeated_ids"]
    return result
