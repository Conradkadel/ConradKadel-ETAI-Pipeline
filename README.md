# Baseline Predictive Pipeline -- ETAI

- Student 20260680 - Conrad Kadel

First Test Class 1 - 16th of September

Precision Decision Tree :
- Train accuracy: 0.829
- Test accuracy:  0.627
Precision Logisic Regression : 
- Train accuracy: 0.678
- Test accuracy:  0.679

-  Decision Tree is overfitting so Logisic Regression is better

Class 2 - 23rd of September (EDA + Preprocessing)

After adding the diagnosis-driven cleaning + leak-safe preprocessing (target encoding + standard scaling):

Decision Tree :
- Train accuracy: 0.792
- Test accuracy:  0.609
Logistic Regression :
- Train accuracy: 0.676
- Test accuracy:  0.657

- Decision Tree still overfits (gap +0.18). Logistic Regression generalises well (gap +0.02) and is still the best model.

This is the **starting point** for your semester project: a small but *complete* predictive pipeline -- every piece a real project needs (entry point, config, data loading, preprocessing, model, evaluation), just kept as simple as possible for now.

The task: predict two-year recidivism using ProPublica's COMPAS
dataset -- the data behind a real 2016 investigation into a risk-
assessment algorithm actually used by US courts to help inform bail and sentencing decisions. See `data/README.md` for the full problem description and a complete data dictionary before you start.

It has some **deliberately weak spots**. Part of your work this
semester is finding them and making them better -- see the pipeline progress table below, which tracks what changes and why as the weeks
go on.

## Project structure

```
.
├── main.py                # entry point: run the whole pipeline
├── config.yaml             # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py             # loading
│   ├── data_diagnostics.py # missingness test, domain rules, duplicate check (week 3)
│   ├── preprocessing.py    # cleaning, leak-safe preprocessor + train/test split (grown in week 3)
│   ├── model.py             # model construction
│   ├── evaluate.py         # accuracy metrics + fairness check
│   └── results.py          # saves each run's report to disk
├── results/                # created automatically -- one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md            # problem description + full data dictionary
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |
| 3 | EDA + preprocessing: diagnose the data, then fix it | New `src/data_diagnostics.py` (missingness-mechanism test via chi-square + Cramér's V, domain-rule invalid-value detection, two-way duplicate check). `src/preprocessing.py` now has `clean_dataset`, `add_missingness_indicators`, `split_features_target`, `build_preprocessor` (leak-safe `ColumnTransformer`) and `split_train_test`, replacing the naive `dropna()` / `get_dummies()`. Encoder/scaler (target + standard) picked by a 4x4 grid over 15 repeated splits. 3 redundant columns dropped. `config.yaml` gains `diagnostics` and `preprocessing` sections, so no column names are hardcoded. |

## Preprocessing decisions

From the week 3 diagnosis (`01_eda_introduction.ipynb`) and preprocessing (`02_preprocessing.ipynb`) notebooks.

| Column(s) | Issue found | Mechanism | What was done |
|---|---|---|---|
| `age` | 2.0% missing + invalid values (< 18 or > 100) | MCAR / domain rule | invalid -> NaN, then median impute, no indicator |
| `juv_fel_count` | 3.0% missing + negative values | MCAR / domain rule | invalid -> NaN, then median impute, no indicator |
| `priors_count` | ~7% missing (incl. `-` placeholders) + values > 60 | MNAR (tied to `age_cat`, V ≈ 0.36) / domain rule | invalid -> NaN, median impute + `priors_count_was_missing` flag |
| `c_charge_degree` | 3.2% missing | MNAR (tied to `age_cat`, V ≈ 0.35) | mode impute + `c_charge_degree_was_missing` flag |
| `sex` | ~1.5% missing (incl. placeholders) | MCAR | mode impute, no indicator |
| `race` | ~2% missing (placeholders) | MCAR | not a model feature, kept aside for the fairness check only |
| `decile_score` | values outside 1-10 | domain rule | -> NaN (not a model feature, only used for comparison) |
| `sex` / `race` / `c_charge_degree` / `score_text` | same category spelled many ways | data entry | canonicalized to one label |
| (whole row) | 72 exact duplicates = 72 repeated ids | data entry | dropped, keep first occurrence |
| `prior_offenses` | same as `priors_count` (r = 1.00) | multicollinearity | dropped |
| `age_in_months` | same as `age` (r = 1.00) | multicollinearity | dropped |
| `juvenile_total` | sum of the 3 `juv_*_count` columns (VIF) | multicollinearity | dropped |

Encoder / scaler: the grid of 4 encoders x 4 scalers with logistic regression over 15 repeated splits picked **target encoding + standard scaling** (mean accuracy 0.670). The runner-up (target + none) is within noise on a paired check.

## Best Model

| Week | Model | Train acc | Test acc | Gap |
|---|---|---|---|---|
| 2 | Decision Tree | 0.829 | 0.627 | +0.202 |
| 2 | Logistic Regression | 0.678 | 0.679 | -0.001 |
| 3 | Decision Tree | 0.792 | 0.609 | +0.183 |
| 3 | Logistic Regression | 0.676 | 0.657 | +0.019 |

Current best: **Logistic Regression**. Test accuracy is a bit lower than in week 2, but week 2 dropped every row with any missing value (and the invalid values stayed in), so it was tested on a smaller, "easier" subset. Week 3 keeps all 7,214 de-duplicated rows.

## Environment setup

You only need to do this once per machine.

### macOS / Linux
```bash
python3 -m venv venv                 # creates an isolated Python environment in a folder called "venv"
source venv/bin/activate             # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```

### Windows -- PowerShell
```powershell
python -m venv venv                  # creates an isolated Python environment in a folder called "venv"
venv\Scripts\activate                # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```
If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Windows -- cmd.exe
Same three steps as above, just with cmd's own activation command:
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Once the environment is active you'll see `(venv)` at the start of your prompt. To leave it later, run `deactivate` (same command on every OS).

### Every time after the first

Creating the environment and installing packages only needs to happen once, ever. Every other time you sit down to work -- a new terminal window, the next practical class, tomorrow -- you don't repeat any of the steps above. From the project's root folder, you just need to:

**macOS / Linux**
```bash
source venv/bin/activate
python main.py
```

**Windows**
```powershell
venv\Scripts\activate
python main.py
```

That's it -- activate, then run. If you don't see `(venv)` at the start of your prompt, the environment isn't active and `python main.py` may use the wrong Python (or fail to find a package) entirely.

## Running the pipeline

With the environment active (see above), from the project's root
folder, on any OS:
```bash
python main.py
```

This loads `config.yaml`, loads and preprocesses the data, trains the model, and prints:
- **train accuracy and test accuracy, side by side.** Comparing the two is how you catch overfitting: if the model looks much better on the data it was trained on than on data it's never seen, it has memorised rather than learned something that generalises. 
- a classification report on the test set
- a false-positive-rate-by-race comparison between our model and
  COMPAS's own score

All of this is also saved to a timestamped file in `results/` (e.g.`results/run_20260916_143012.txt`), so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.
`results/` is created automatically the first time you run the
pipeline, and isn't tracked in git (see `.gitignore`) since it's
generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

## Dataset

See `data/README.md`.
