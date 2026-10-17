"""
Saving each run's results to disk.

Writes each run's full report (cross-validation scores, classification report, fairness table)
to a timestamped file in `results/`, so runs can be compared after changing config.yaml.
"""
import os
from datetime import datetime


def _tuning_line(tuning) -> str:
    """Week 5: records how (and whether) the model was tuned, so a results file is reproducible."""
    if not tuning or not tuning.get("enabled", False):
        return "Tuning: off\n"
    return (f"Tuning: Optuna  n_trials={tuning['n_trials']}  inner_folds={tuning['n_splits']}  "
            f"random_state={tuning['random_state']}\n")


def save_run(results_dir: str, config: dict, report_text: str) -> str:
    """
    Writes one run's full report to a timestamped .txt file inside
    `results_dir` (created automatically if it doesn't exist yet) and
    returns the path that was written.
    """
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(results_dir, f"run_{timestamp}.txt")

    cv = config["cv"]
    header = (
        f"Run: {timestamp}\n"
        f"Model: {config['model']['type']}  params={config['model'].get('params')}\n"
        f"Preprocessing: encoder={config['preprocessing']['encoder']}  "
        f"scaler={config['preprocessing']['scaler']}\n"
        f"Engineered features: {config['preprocessing'].get('engineered_features') or 'none'}\n"
        f"CV: {cv['n_splits']} stratified folds  shuffle={cv.get('shuffle', True)}  "
        f"random_state={cv.get('random_state')}  metric={cv.get('scoring', 'accuracy')}\n"
        f"{_tuning_line(config.get('tuning'))}"
        f"Locked test set: size={config['test_set']['size']}  "
        f"random_state={config['test_set']['random_state']}  (not evaluated)\n"
        + "=" * 60 + "\n\n"
    )

    with open(path, "w") as f:
        f.write(header + report_text)

    return path
