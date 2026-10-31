"""
Saving each run's results to disk.

Writes each run's full report (cross-validation scores, classification report, fairness table)
to a timestamped file in `results/`, so runs can be compared after changing config.yaml.
"""
import os
import pandas as pd


def _tuning_line(tuning) -> str:
    """Week 5: records how (and whether) the model was tuned, so a results file is reproducible."""
    if not tuning or not tuning.get("enabled", False):
        return "Tuning: off\n"
    return (f"Tuning: Optuna  n_trials={tuning['n_trials']}  inner_folds={tuning['n_splits']}  "
            f"random_state={tuning['random_state']}\n")


def save_run(results_dir: str, run: str, config: dict, report_text: str) -> str:
    """
    Writes the run's settings and full report to `results_dir/run_<run>.txt` (the folder is
    created if needed) and returns the path. `run` is the run id made by main.py.
    """
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"run_{run}.txt")

    cv = config["cv"]
    header = (
        f"Run: {run}\n"
        f"Model: {config['model']['type']}  params={config['model'].get('params')}\n"
        f"Preprocessing: encoder={config['preprocessing']['encoder']}  "
        f"scaler={config['preprocessing']['scaler']}\n"
        f"Engineered features: {config['preprocessing'].get('engineered_features') or 'none'}\n"
        f"CV: {cv['n_splits']} stratified folds  shuffle={cv.get('shuffle', True)}  "
        f"random_state={cv.get('random_state')}\n"
        f"Metrics: {cv['metrics']}  selection metric: {cv['selection_metric']}\n"
        f"{_tuning_line(config.get('tuning'))}"
        f"Locked test set: size={config['test_set']['size']}  "
        f"random_state={config['test_set']['random_state']}  (not evaluated)\n"
        + "=" * 60 + "\n\n"
    )

    with open(path, "w") as f:
        f.write(header + report_text)

    return path


def save_scores(path: str, row: dict) -> pd.DataFrame:
    """Appends one run to the CSV at `path` (created if needed) and returns all runs."""
    runs = pd.read_csv(path, dtype={"run": str}) if os.path.exists(path) else pd.DataFrame()
    runs = pd.concat([runs, pd.DataFrame([row])], ignore_index=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    runs.to_csv(path, index=False)
    return runs