"""
Entry point for the predictive pipeline.

Run with:
    python main.py

This orchestrates the full pipeline:
    load config -> load data -> diagnose/clean (week 3) -> drop duplicate rows (training only, week 4)
    -> split features/target
    -> set the final test set aside, locked (week 4)
    -> stratified k-fold cross-validation of preprocessing + model on the development set (week 4)
    -> week 5: hyperparameter tuning (Optuna) + nested CV, if tuning.enabled
    -> out-of-fold classification report + fairness check
    -> refit the final model on the whole development set -> save results
"""
import optuna
import yaml
from sklearn.model_selection import StratifiedKFold

from src.data import load_data
from src.preprocessing import (add_engineered_features, clean_dataset, drop_duplicate_rows, split_features_target, split_dev_test,)
from src.model import build_pipeline  # week 5: preprocessing + model as ONE estimator
from src.evaluate import (cross_validate_pipeline, cv_report, oof_classification_report, fairness_report,)
from src.results import save_run
from src.tuning import tune_pipeline, nested_cross_validate, tuning_report  # week 5


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    # load + diagnose-and-clean (week 3): nothing here is learned from the data, so it's
    # safe to run on the whole dataset -- see src/preprocessing.py
    df_raw = load_data(config["data"]["path"])
    df_clean = clean_dataset(df_raw, config["diagnostics"])          # row-preserving: also safe for new data
    # training data only: the same person must not count twice, or sit in both dev and test
    df_clean = drop_duplicate_rows(df_clean, config["diagnostics"].get("id_column"))

    # week 6: hand-made features chosen in config (preprocessing.engineered_features; [] = none)
    df_clean = add_engineered_features(df_clean, config["preprocessing"].get("engineered_features", []))
    
    mnar_sources = config["preprocessing"].get("mnar_indicator_sources", [])
    X, y, extras = split_features_target(df_clean, config["data"], mnar_sources)

    # week 4: the final test set is set aside HERE and never used again in this script.
    # Every decision from now on (preprocessing, model, hyperparameters) is made on the
    # development set only. The test set is used only for the final assessment.
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = split_dev_test(
        X, y, extras,
        test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"],
    )

    # preprocessing lives INSIDE the pipeline, so cross-validation re-fits it on the
    # training part of every fold -- the validation fold never leaks into its own preprocessing
    # week 5: built by build_pipeline() so tuning can address its steps by name (model__max_depth, ...)
    pipeline = build_pipeline(config["preprocessing"], config["model"])

    # a fixed random_state = the same folds on every run and for every model, so comparing
    # two models' fold scores is a like-for-like (paired) comparison
    cv_config = config["cv"]
    shuffle = cv_config.get("shuffle", True)
    cv = StratifiedKFold(n_splits=cv_config["n_splits"], shuffle=shuffle,
                         random_state=cv_config.get("random_state") if shuffle else None)
    scoring = cv_config.get("scoring", "accuracy")

    # fold scores + out-of-fold predictions (each row predicted by the fold model that did NOT train on it)
    fold_scores, y_oof = cross_validate_pipeline(
        pipeline, X_dev, y_dev, cv, scoring, n_jobs=cv_config.get("n_jobs", 1)
    )

    # logging: record which hyperparameters the default CV used, so a results file is self-explanatory
    header = f"Hyperparameters from config.yaml: {config['model'].get('params')}"
    print(header)
    report = header + "\n" + cv_report(fold_scores, scoring)

    # week 5: hyperparameter tuning -- on the development set only, every candidate scored by CV of the whole pipeline 
    tuning_config = config.get("tuning", {})
    tuning_enabled = tuning_config.get("enabled", False)
    nested_scores = None
    if tuning_enabled:
        model_type = config["model"]["type"]
        search_spaces = tuning_config.get("search_spaces") or {}
        if model_type not in search_spaces:
            raise ValueError(f"tuning.enabled is true but tuning.search_spaces has no entry for "
                             f"'{model_type}'. Options: {list(search_spaces)}")
        search_space = search_spaces[model_type]
        n_trials = tuning_config["n_trials"]
        tuning_seed = tuning_config["random_state"]
        n_jobs = cv_config.get("n_jobs", 1)
        inner_cv = StratifiedKFold(n_splits=tuning_config["n_splits"], shuffle=True,
                                   random_state=tuning_seed)
        optuna.logging.set_verbosity(optuna.logging.WARNING)   # logging: hide per-trial lines (the report lists the top trials)
        print(f"\nNested cross-validation: {cv.get_n_splits()} outer folds x {n_trials} trials x "
              f"{inner_cv.get_n_splits()} inner folds ...")      # logging: what is about to run (can take minutes)

        # 1. the honest estimate (its out-of-fold predictions replace the untuned ones)
        nested_scores, y_oof = nested_cross_validate(
            pipeline, X_dev, y_dev, cv, inner_cv, scoring, search_space, n_trials, tuning_seed,
            n_jobs=n_jobs
        )
        report += "\n\nNested cross-validation (tuned model, honest estimate)\n" + cv_report(nested_scores, scoring)

        # 2. the hyperparameters we keep: the same procedure, once, on the whole development set
        pipeline, study = tune_pipeline(
            pipeline, X_dev, y_dev, inner_cv, scoring, search_space, n_trials, tuning_seed,
            n_jobs=n_jobs
        )
        print()
        report += "\n\n" + tuning_report(study, nested_scores, scoring)

    report += "\n\n" + oof_classification_report(y_dev, y_oof)
    report += "\n" + fairness_report(
        y_dev, y_oof, extras_dev, sensitive_attr=config["data"]["sensitive_attr"]
    )

    # the model we'd actually use: same pipeline, refit on EVERY development row. CV above
    # estimated how well this recipe does; it didn't produce a model.
    final_model = pipeline.fit(X_dev, y_dev)
    refit = f"Final model: {config['model']['type']} refit on all {len(X_dev)} development rows."
    if tuning_enabled:
        refit += f" Tuned hyperparameters: {study.best_params}"
    print(refit)
    report += "\n" + refit + "\n"

    locked = (f"Locked test set: {len(X_test)} rows set aside, not evaluated. "
              f"Development set: {len(X_dev)} rows.")
    print(locked)
    report += "\n" + locked + "\n"

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"Full results saved to {path}")
    if nested_scores is not None:
        # per-fold nested scores, so two runs can be compared with corrected_paired_test()
        nested_path = path.replace(".txt", "_nested.csv")
        nested_scores.to_csv(nested_path, index=False)
        print(f"Nested fold scores saved to {nested_path}")


if __name__ == "__main__":
    main()
