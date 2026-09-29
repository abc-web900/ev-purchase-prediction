import numpy as np
from sklearn.metrics import f1_score

from ev.validation import select_threshold, threshold_table


def test_threshold_table_matches_predictions_with_ties():
    y = np.array([0, 1, 1, 0, 0, 1])
    probability = np.array([0.0, 0.2, 0.2, 0.7, 1.0, 1.0])
    table = threshold_table(y, probability)
    for row in table.itertuples():
        assert np.isclose(row.f1, f1_score(y, probability >= row.threshold, zero_division=0))
    chosen, table = select_threshold(y, probability, min_recall=0.65)
    assert table.loc[table.threshold == chosen, "recall_1"].iloc[0] >= 0.65
