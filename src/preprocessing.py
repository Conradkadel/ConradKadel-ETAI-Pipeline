"""
Preprocessing (week 3) -- raw data in, model-ready train/test split out.
Built from 02_preprocessing.ipynb. Same file as week 2, it just grew:

    clean_dataset              -> category cleanup, placeholders/invalid values -> NaN,
                                  drop duplicates, drop redundant columns
    add_missingness_indicators -> `<col>_was_missing` flags for the MNAR columns
    split_features_target      -> (X, y, extras); y is None on label-free data
    build_preprocessor         -> leak-safe ColumnTransformer (impute + encode + scale)
    split_train_test           -> stratified train/test split (week 2's original job)

Two rules every function respects:
  - leak-safe: clean_dataset / split_features_target learn nothing from the data, so they
    can run on the whole dataset. Imputing, encoding and scaling (build_preprocessor) are
    only ever FIT on the training rows, inside the sklearn Pipeline in main.py.
  - deployable: nothing before the split needs the target column to be present.

No COMPAS column names are hardcoded -- they all come from config.yaml.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler, RobustScaler
from category_encoders import CountEncoder, TargetEncoder

from src.data_diagnostics import flag_invalid_values


# ---------------------------------------------------------------------------
# Step 1 - cleaning (no fitting, safe on the whole dataset)
# ---------------------------------------------------------------------------

def find_placeholder_rows(series: pd.Series, tokens: set) -> pd.Series:
    """True where the value is a placeholder like '-', '?', 'n/a'."""
    return series.astype(str).str.strip().isin(tokens)


def canonicalize_categories(df: pd.DataFrame, columns_and_maps: dict, placeholder_tokens: set) -> pd.DataFrame:
    """Normalise whitespace/casing, then map known spelling variants to one label
    (e.g. 'MALE', ' male ' -> 'Male'). Placeholder tokens become NaN.
    Values not in the map are kept as-is, so a new category doesn't silently disappear."""
    out = df.copy()
    for col, mapping in columns_and_maps.items():
        if col not in out.columns:
            continue
        cleaned = out[col].astype(str).str.strip()
        lowered = cleaned.str.lower()
        out[col] = lowered.map(mapping).fillna(cleaned)
        out.loc[find_placeholder_rows(out[col], placeholder_tokens), col] = np.nan
    return out


def clean_dataset(df: pd.DataFrame, diagnostics_config: dict) -> pd.DataFrame:
    """
    Applies the EDA notebook's diagnosis, all read from config.yaml `diagnostics`:
    category cleanup, placeholder / domain-rule -> NaN, de-duplication,
    redundant-column removal. Target-agnostic -- safe on label-free inference data.
    """
    out = df.copy()
    placeholder_tokens = set(diagnostics_config["placeholder_tokens"])

    # 1) numeric columns that loaded as text because of placeholder tokens -> real numbers
    for col in diagnostics_config.get("numeric_text_columns", []):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col].replace(list(placeholder_tokens), np.nan), errors="coerce")

    # 2) domain-rule violations -> NaN (e.g. age of -3, decile score of 23)
    flag_invalid_values(out, diagnostics_config.get("validity_rules", {}))

    # 3) category canonicalization (also turns placeholder tokens into NaN)
    out = canonicalize_categories(out, diagnostics_config.get("canonical_maps", {}), placeholder_tokens)

    # 4) duplicates: exact row dupes first, then repeated ids -- keep the first occurrence
    out = out.drop_duplicates()
    id_column = diagnostics_config.get("id_column")
    if id_column and id_column in out.columns:
        out = out.drop_duplicates(subset=id_column, keep="first")

    # 5) redundant columns found via correlation heatmap + VIF
    cols_to_drop = [c for c in diagnostics_config.get("redundant_columns", []) if c in out.columns]
    out = out.drop(columns=cols_to_drop)

    return out


# ---------------------------------------------------------------------------
# Step 2/3 - features, target and the leak-safe preprocessor
# ---------------------------------------------------------------------------

def add_missingness_indicators(df: pd.DataFrame, mnar_indicator_sources: list) -> pd.DataFrame:
    """Adds a `<col>_was_missing` flag for each MNAR column, BEFORE that column gets imputed.
    Target-agnostic -- safe on label-free inference data."""
    out = df.copy()
    for col in mnar_indicator_sources:
        if col in out.columns:
            out[f"{col}_was_missing"] = out[col].isna().astype(int)
    return out


def split_features_target(df: pd.DataFrame, data_config: dict, mnar_indicator_sources: list):
    """
    Splits into (X, y, extras).
      X      -> model features (everything not in drop_columns / target / sensitive attr)
      y      -> the target, or None on label-free inference data
      extras -> sensitive attribute + COMPAS's own score, kept aside for the fairness report only
    """
    target = data_config["target"]
    sensitive_attr = data_config["sensitive_attr"]
    extras_columns = [sensitive_attr] + data_config.get("fairness_extra_columns", [])

    df = add_missingness_indicators(df, mnar_indicator_sources)
    y = df[target] if target in df.columns else None

    extras_cols = [c for c in extras_columns if c in df.columns]
    extras = df[extras_cols].copy() if extras_cols else None

    drop_always = set(data_config.get("drop_columns", [])) | {target, sensitive_attr}
    feature_cols = [c for c in df.columns if c not in drop_always]
    X = df[feature_cols]
    return X, y, extras


def build_preprocessor(preprocessing_config: dict) -> ColumnTransformer:
    """
    Factory: builds a leak-safe ColumnTransformer for the encoder/scaler pair in config.yaml
    (picked by the empirical grid in 02_preprocessing.ipynb). Every encoder tolerates unseen
    categories at transform time -- fit on train, applied unchanged to test/inference.
    """
    encoder_name = preprocessing_config["encoder"]
    scaler_name = preprocessing_config["scaler"]
    imputation = preprocessing_config["imputation"]

    # built inside the function so every call gets fresh, unfitted objects
    scalers = {
        "none": "passthrough",
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
    }
    encoders = {
        "onehot": OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        "ordinal": OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        "count": CountEncoder(handle_unknown=0, handle_missing=0),
        "target": TargetEncoder(handle_unknown="value", handle_missing="value"),
    }

    numeric_indicator_cols = [f"{c}_was_missing" for c in preprocessing_config.get("mnar_indicator_sources", [])]

    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=imputation["numeric_strategy"])),
        ("scale", scalers[scaler_name]),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=imputation["categorical_strategy"])),
        ("encode", encoders[encoder_name]),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, preprocessing_config["numeric_features"]),
        ("categorical", categorical_pipeline, preprocessing_config["categorical_features"]),
        ("indicators", "passthrough", numeric_indicator_cols),   # 0/1 flags, no processing needed
    ])


# ---------------------------------------------------------------------------
# Split (week 2's original job)
# ---------------------------------------------------------------------------

def split_train_test(X, y, extras, test_size: float, random_state: int):
    """Stratified split of X, y and extras together, so all three stay row-aligned.
    This is the leak-safe boundary: from here on, anything fitted sees only X_train."""
    X_train, X_test, y_train, y_test, extras_train, extras_test = train_test_split(
        X, y, extras, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, extras_train, extras_test
