# ev

**Electric vehicle purchase prediction**

Classifies whether a survey respondent will buy an electric vehicle using demographic, commute, charging access, financial, and attitude features.

## Evaluation results and findings

The selected **weighted soft-voting ensemble** achieved **89.7647% accuracy**,
**0.701570 buyer F1**, and **0.941676 ROC-AUC** on **100,300 labeled holdout
rows**. It also generated predictions for **286,571 external test rows**.
Those external rows had no true `Will_Buy_EV` labels, so their accuracy and F1
could not be calculated.

The final results below come from the supplied output of [ev.py](ev.py).
Development comparisons and the selected settings are recorded in
[ev.ipynb](ev.ipynb). The final run fitted the selected ensemble components
once, using the previously selected weights and threshold. It did not repeat
the model, weight, or threshold searches.

### Data split and target

The target was `Will_Buy_EV`, with `No` mapped to class 0 and `Yes` to class 1.
The labeled dataset contained **668,665 rows**, divided using stratified splits
and random seed 42:

| Partition | Rows | Purpose |
| --- | ---: | --- |
| Training, approximately 70% | 468,065 | Fit preprocessing and the three ensemble components |
| Validation, approximately 15% | 100,300 | Development model/weight/threshold selection; CatBoost iteration selection |
| Internal labeled holdout, approximately 15% | 100,300 | Evaluate the frozen ensemble and threshold |
| Separate external test file | 286,571 | Generate predictions; true labels unavailable |

The holdout contained **82,783 non-buyers** and **17,517 buyers**. Buyers made
up only **17.46%** of the holdout, so buyer recall, buyer F1, and balanced
accuracy are useful alongside overall accuracy.

### Feature engineering used by the final ensemble

The model used 13 original predictors: seven numeric variables (`Age`,
`Annual_Income_USD`, `Daily_Commute_km`, `Number_of_Cars_Owned`,
`Charging_Stations_Near_Home`, `Charging_Stations_Near_Work`, and
`Environmental_Concern_Level`) and six categorical variables (`Gender`,
`City_Type`, `Current_Car_Type`, `Home_Charging_Possible`, `Subsidy_Available`,
and `Range_Anxiety_Level`). IDs and the target were excluded from predictors.

Eleven additional features were created:

| Engineered feature | Calculation |
| --- | --- |
| `Total_Charging_Stations` | Home stations + work stations |
| `Stations_per_Commute` | Total stations / (daily commute + 1) |
| `Commute_per_Station` | Daily commute / (total stations + 1) |
| `Income_per_Car` | Annual income / (cars owned + 1) |
| `HomeCharge_Binary` | Home charging: Yes = 1, No = 0; missing/unmapped = 0 |
| `Subsidy_Binary` | Subsidy available: Yes = 1, No = 0; missing/unmapped = 0 |
| `HomeCharge_And_Subsidy` | Home-charging indicator × subsidy indicator |
| `Concern_x_Subsidy` | Environmental concern × subsidy indicator |
| `Concern_x_HomeCharge` | Environmental concern × home-charging indicator |
| `Commute_x_HomeCharge` | Daily commute × home-charging indicator |
| `Range_Anxiety_Ordinal` | Low = 0, Medium = 1, High = 2; missing/unmapped = -1 |

Features were calculated before numeric imputation. Missing numeric values,
including missing derived values, were filled with medians learned only from
training rows. Entirely missing numeric columns had a zero fallback. Missing
categorical values became `Missing`.

LightGBM and XGBoost used one-hot encoding, with prediction columns aligned to
the training columns. CatBoost used the categorical columns directly. The
recorded development fit had **24 features before encoding** and **35 encoded
columns** for LightGBM and XGBoost. No feature scaling was used for these tree
models. The later expanded logistic-regression feature recipe was not used by
this ensemble.

### Model selection: what worked and what did not

These are the notebook's recorded validation comparisons. Selection prioritized
accuracy while requiring at least 65% buyer recall; buyer F1 and average
precision were used to break accuracy ties.

| Candidate | Threshold | Validation accuracy | Validation buyer F1 |
| --- | ---: | ---: | ---: |
| **Weighted soft voting** | **0.48** | **0.8995** | **0.7064** |
| LightGBM | 0.49 | 0.8993 | 0.7037 |
| Stacking | 0.43 | 0.8993 | 0.7037 |
| Equal soft voting | 0.49 | 0.8992 | 0.7029 |
| CatBoost | 0.50 | 0.8992 | 0.6997 |
| XGBoost | 0.49 | 0.8990 | 0.7032 |

- **Weighting the ensemble helped modestly.** It produced the highest recorded
  validation accuracy and buyer F1 among these candidates. LightGBM supplied
  most of the final probability estimate.
- **Equal voting and stacking did not exceed the weighted ensemble.** Adding
  these alternatives did not produce a better recorded validation result.
- **SMOTE did not improve the LightGBM experiment.** Its holdout accuracy and
  buyer F1 were 0.8972 and 0.6977, compared with 0.8978 and 0.6998 for the
  original LightGBM.
- **Expanded logistic-regression features did not overtake the ensemble.** That
  experiment achieved holdout accuracy 0.8940 and buyer F1 0.6873. These model
  comparisons do not isolate the effect of each engineered feature.
- **The ensemble's holdout improvement involved a tradeoff.** Compared with
  LightGBM alone, buyer F1 increased from 0.6998 to 0.7016 and buyer recall from
  0.6822 to 0.6889, while precision fell from 0.7182 to 0.7147 and accuracy
  decreased slightly from 0.8978 to 0.8976. The gains were small; no statistical
  significance claim is made.

### Final ensemble configuration

| Component | Weight | Main settings |
| --- | ---: | --- |
| LightGBM | 70% | 700 trees, learning rate 0.04, 31 leaves, minimum child samples 30 |
| XGBoost | 15% | 700 trees, learning rate 0.05, maximum depth 5, minimum child weight 3 |
| CatBoost | 15% | Up to 800 iterations, learning rate 0.05, depth 7, L2 leaf regularization 5 |

The predicted buyer probability was:

```text
P(Yes) = 0.70 × P_LightGBM + 0.15 × P_XGBoost + 0.15 × P_CatBoost
```

Rows with probability **at least 0.48** received `Yes`; lower probabilities
received `No`. The weights and threshold were fixed before holdout evaluation.

CatBoost completed 800 iterations and retained the first **799 trees**. Its
best validation ROC-AUC was **0.9415680941**, at zero-based iteration **798**.
The `test` and `bestTest` labels in its training output refer to the supplied
validation set, not the external test file. This component score is distinct
from the final ensemble's holdout ROC-AUC.

### Final labeled holdout metrics

| Metric | Result |
| --- | ---: |
| Evaluated rows | 100,300 |
| Decision threshold | 0.48 |
| Accuracy | 0.897647 |
| Buyer precision | 0.714743 |
| Buyer recall | 0.688874 |
| Buyer F1 | 0.701570 |
| Balanced accuracy | 0.815349 |
| ROC-AUC | 0.941676 |
| Average precision | 0.754587 |

| Class / average | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Not Buyer (0) | 0.93467 | 0.94182 | 0.93823 | 82,783 |
| Buyer (1) | 0.71474 | 0.68887 | 0.70157 | 17,517 |
| Macro average | 0.82470 | 0.81535 | 0.81990 | 100,300 |
| Weighted average | 0.89626 | 0.89765 | 0.89690 | 100,300 |

### Confusion matrix and error findings

Rows represent actual classes; columns represent predicted classes.

| Actual class | Predicted No | Predicted Yes |
| --- | ---: | ---: |
| Not Buyer (0) | 77,967 true negatives | 4,816 false positives |
| Buyer (1) | 5,450 false negatives | 12,067 true positives |

The ensemble correctly classified **90,034 rows** and misclassified **10,266**.
It identified **68.89% of actual buyers**, missing **31.11%** of them. Buyer
precision of **71.47%** means that proportion of predicted buyers actually had
the buyer label.

It identified **94.18% of non-buyers**, while **5.82%** of actual non-buyers
were incorrectly predicted as buyers. The main measured weakness was missing
buyers: buyer F1 was 0.70157, compared with 0.93823 for non-buyers. Macro F1 of
0.81990 and balanced accuracy of 0.815349 show this difference more clearly
than overall accuracy alone.

For context, predicting `No` for every holdout row would yield **82.54%
accuracy** but zero buyer recall. The ensemble improved accuracy by about
**7.23 percentage points** while detecting 12,067 buyers. Its confusion matrix
matches the notebook's recorded weighted-ensemble result, reproducing that
evaluation on the same split.

### External test predictions

Prediction completed for **286,571 external test rows**, preserving input order
and IDs. Each row received a `Will_Buy_EV` prediction and a
`probability_class_1`, where class 1 means `Yes`.

| ID | Predicted `Will_Buy_EV` | Probability of Yes |
| --- | --- | ---: |
| 668665 | No | 0.011480 |
| 668666 | No | 0.015096 |
| 668667 | No | 0.004454 |
| 668668 | No | 0.002984 |
| 668669 | No | 0.019100 |
| 668670 | No | 0.003375 |
| 668671 | No | 0.373308 |
| 668672 | No | 0.000591 |
| 668673 | Yes | 0.653537 |
| 668674 | No | 0.004916 |

For example, ID `668673` received `Yes` because 0.653537 exceeded the threshold
of 0.48, while ID `668671` received `No` because 0.373308 was below it.
`submission.csv` contains the IDs and predicted labels; `test_predictions.csv`
also contains the probabilities.

The external file had no true `Will_Buy_EV` labels. **No external-test accuracy,
F1, confusion matrix, or ROC-AUC was measured.** The 89.7647% accuracy reported
above belongs to the labeled holdout. The ten-row preview does not establish
the full test set's class distribution or prediction correctness.

## Package reference

The sections below describe the separate modular package. Its default model
selection and feature configuration differ from the selected ensemble in
[ev.py](ev.py); the reported results above belong to that standalone run.

## Run the Python files

This project follows the `freight-rate-ml-assessment` layout: runnable `.py` files
in `scripts/`, reusable modules in `src/ev/`, CSV inputs in `data/`, and
models/reports in `outputs/`. Run the following commands from this project folder.

Python 3.11 or newer is required. Install dependencies, then put your original
training CSV at `data/train.csv` and prediction rows at `data/new_rows.csv`.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS instead: source .venv/bin/activate
python -m pip install -r requirements.txt

python scripts/run_eda.py
python scripts/run_train.py
python scripts/run_predict.py
python scripts/build_report.py
```

The scripts import this folder's `src/` directly. An editable package install and
notebook execution are not required. The source CSVs were not supplied, so real
data must be added before running these default commands.

| File | What it does |
| --- | --- |
| `scripts/run_eda.py` | Saves data summaries, missing values, correlations, target associations, and figures. |
| `scripts/run_train.py` | Performs cross-validation, selects the model, evaluates holdout, and exports it. |
| `scripts/run_predict.py` | Loads the fitted export and predicts directly from raw CSV rows. |
| `scripts/build_report.py` | Builds `REPORT.md` and `REPORT.html` from a completed training run. |
| `scripts/make_demo_data.py` | Generates synthetic rows for checking script execution. |

## Folder structure

```text
ev/
├── scripts/
│   ├── run_eda.py
│   ├── run_train.py
│   ├── run_predict.py
│   ├── build_report.py
│   └── make_demo_data.py
├── src/ev/
│   ├── config.py       # paths, settings, default/full model profiles
│   ├── data.py         # CSV loading, validation, target parsing
│   ├── schema.py       # required raw input columns
│   ├── features.py     # feature engineering and preprocessing
│   ├── models.py       # model families, pipelines, search spaces
│   ├── validation.py   # splits, metrics, threshold selection
│   ├── pipeline.py     # training, evaluation, model export
│   ├── predict.py      # saved-model inference
│   ├── artifacts.py    # persistence and model metadata
│   ├── eda.py          # analysis tables and plots
│   ├── report.py       # readable reports from saved results
│   ├── demo.py         # synthetic example generator
│   └── __init__.py
├── data/
├── outputs/
├── examples/predict.csv
├── tests/
├── .github/workflows/ci.yml
├── .gitignore
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Training settings

Edit `src/ev/config.py` to change defaults or pass command-line overrides.
The `TrainConfig` class contains the default settings; `FULL_PROFILE` contains
the broader notebook-derived model comparison. `EV_DATA_DIR` and
`EV_OUTPUT_DIR` environment variables can change the default folders.

Default: always-positive baseline, logistic regression, and random forest. Full: also LightGBM, XGBoost, CatBoost, soft voting, and stacking, with original/engineered features and parameter search.

```bash
python scripts/run_train.py --data data/train.csv --output outputs/run
python scripts/run_train.py --profile full --data data/train.csv --output outputs/full
python scripts/run_train.py --help
```

Available overrides include `--models`, `--feature-sets`, `--n-estimators`,
`--cv-folds`, `--search-iterations`, `--seed`, and `--n-jobs`. The full profile can
be expensive. Use a new output directory for each training/EDA run; existing run
files are preserved. Defaults resolve relative to this project directory; paths
explicitly passed on the command line resolve relative to your working directory.

## Model export and prediction

Training writes the evaluated fitted model to `outputs/run/model.joblib`.
It includes the feature transformer, learned preprocessing, estimator(s), and
class mapping and validation-selected decision threshold.
No separate manual imputation or encoding is needed for prediction.

```bash
python scripts/run_predict.py --model outputs/run/model.joblib --data data/new_rows.csv --output outputs/predictions.csv
python scripts/build_report.py --run outputs/run
```

The training output directory contains:

- `model.joblib`: complete fitted workflow.
- `metadata.json`: raw input schema, model choice, configuration, dependency versions, and data/source/model hashes.
- `requirements.lock.txt`: recorded runtime/model package versions.
- `config.json`: settings captured from this particular run.
- `cv_results.csv`, `cv_folds.csv`, and optional `search_*.csv`: development model comparisons.
- `split_assignments.csv`: original row indices, IDs, and evaluation partitions.
- `holdout_predictions.csv`, `metrics.json`: final evaluation results.
- `threshold_search.csv`: validation threshold scores.

`build_report.py` adds `REPORT.md` and `REPORT.html` without retraining the model.
Only load model files you trust, using the same source and dependency versions as
the export environment. See the [scikit-learn persistence documentation](https://scikit-learn.org/stable/model_persistence.html).

## Input data

Training target: **`Will_Buy_EV`**. Prediction CSVs do not need the target.
Columns may appear in any order; extra columns are ignored unless explicitly
allowlisted. Column names are case sensitive, with surrounding CSV header spaces
trimmed. A required column may contain missing values, but must be present.

| Column | Expected value |
| --- | --- |
| `Age` | numeric; missing allowed |
| `Annual_Income_USD` | numeric; missing allowed |
| `Daily_Commute_km` | numeric; missing allowed |
| `Number_of_Cars_Owned` | numeric; missing allowed |
| `Charging_Stations_Near_Home` | numeric; missing allowed |
| `Charging_Stations_Near_Work` | numeric; missing allowed |
| `Environmental_Concern_Level` | numeric; missing allowed |
| `Gender` | category; missing allowed |
| `City_Type` | category; missing allowed |
| `Current_Car_Type` | category; missing allowed |
| `Home_Charging_Possible` | category; missing allowed |
| `Subsidy_Available` | category; missing allowed |
| `Range_Anxiety_Level` | category; missing allowed |

`id` is optional and is never a predictor. Category values seen in the notebook include `Yes`/`No`, `Low`/`Medium`/`High`, and `Urban`/`Suburban`/`Rural`. Binary predictor values use the notebook's capitalization for engineered mappings.

Training requires at least 30 rows in each class,
plus enough rows/groups for the requested splits and CV folds. Exact duplicate
rows are removed before splitting. CSV IDs are loaded as strings to preserve
large numeric identifiers.

## Features

`original` uses the raw feature allowlist. `engineered` preserves the notebook's
charging station totals/shares, stations per commute/car, income ratios and logs,
home charging/subsidy indicators, range anxiety encoding, interaction terms,
polynomial terms, and fixed age/income/commute bands. Missing numeric values use
training-fold medians. Categorical inputs use fitted one-hot encoding; unknown
categories are accepted at prediction time. Scaling is fitted inside the pipeline
for logistic regression and Gaussian Naive Bayes.

## Evaluation

1. Validate targets (`Yes`/`No` or `1`/`0`), deduplicate exact rows, and reject unresolved repeated IDs. IDs and the target are excluded from predictors.
2. Create stratified train/validation/holdout partitions of approximately 70%/15%/15%.
3. Select model and feature recipes using stratified training CV (positive-class F1 by default). All imputation, encoding, scaling, optional feature processing, and optional SMOTE stay inside each training fold.
4. Fit the winner on training rows. On validation only, select a probability threshold that maximizes precision while achieving at least `min_recall=0.65`; use the primary metric and closeness to 0.5 to break ties.
5. Freeze model and threshold, evaluate holdout once, and export the exact evaluated model. Do not refit on validation after choosing the threshold.

Prediction also includes the notebook's strict probability buckets: at least 0.85
is `high_probability_buyer`, at most 0.15 is `low_probability_buyer`, otherwise
`uncertain`. These are probability buckets, not a measured confidence guarantee.
A validation recall constraint is not a guarantee about holdout or future recall.

CV scores are used for selecting candidates and are not unbiased external-test
estimates. `metrics.json` records the holdout results of this run. No previously
observed notebook metric is reused as a claimed result for these modules.

## EDA outputs

`run_eda.py` writes CSVs for numeric summaries, missing values, category counts,
correlation/covariance matrices, and target associations, plus PNGs for target and
numeric distributions and correlations. Target associations use numeric ANOVA and categorical chi-square tests.
These summaries describe the supplied dataset; EDA results are not used to select
the model automatically.

## Try the scripts without the original dataset

```bash
python scripts/make_demo_data.py --output data/demo.csv --rows 400
python scripts/run_eda.py --data data/demo.csv --output outputs/demo_eda
python scripts/run_train.py --data data/demo.csv --output outputs/demo --n-estimators 20 --cv-folds 2
python scripts/run_predict.py --model outputs/demo/model.joblib --data examples/predict.csv --output outputs/demo/predictions.csv
python scripts/build_report.py --run outputs/demo
```

These inputs are **synthetic**. They check execution, export, and reload; their
scores do not establish real model performance.

## Tests and GitHub

```bash
python -m pip install pytest ruff
python -m pytest
ruff check src scripts tests
```

Tests cover direct script execution, fresh-process export/reload, missing/unseen
inputs, split isolation, invalid data, optional model families, and reports.
Each folder can be uploaded as its own same-named GitHub repository:

```bash
git init
git add .
git commit -m "Add ev Python ML project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ev.git
git push -u origin main
```

Local CSVs, generated outputs, environments, and caches are ignored by Git. No
license was inferred from the supplied notebook. If you also want an installable
wheel, `pyproject.toml` remains available (`pip install -e .` or `python -m build`).

## Source and limitations

The source CSV, dataset provenance, and verified source URL were not included. Evaluation assumes independent respondents; repeated people or time-dependent data need a suitable split. The modular pipeline intentionally replaces test-set threshold experiments and globally prepared encodings with fold-fitted preprocessing and separate validation. Its metrics are therefore not claimed to match historical notebook outputs. SMOTE on encoded categories follows the notebook approach and can interpolate categorical indicators; compare it on validation rather than assuming it helps.

Converted from `ev.ipynb`. This repository includes the source notebook and
`ev.py`, the standalone selected-ensemble workflow, alongside the modular
package and supporting files. Notebook-only installation/display cells and obsolete duplicate
experiments are not part of the runnable workflow. No previous notebook score is
presented as a result of this conversion.

Original notebook SHA-256: `38b5e73cf56004821be2e348f59283f874d71d04ecc4fecd04ea84d48c2cbdd9`.

For the notebook's SMOTE experiment, install `imbalanced-learn>=0.14,<1` and run
`python scripts/run_train.py --smote --output outputs/smote`. Resampling stays
inside training folds; validation and holdout are not resampled.
