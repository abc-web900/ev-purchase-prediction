# Electric vehicle purchase prediction: results and findings

## Data split and target

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

## Feature engineering used by the final ensemble

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

## Model selection: what worked and what did not

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

## Final ensemble configuration

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

## Final labeled holdout metrics

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

## Confusion matrix and error findings

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

## External test predictions

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
