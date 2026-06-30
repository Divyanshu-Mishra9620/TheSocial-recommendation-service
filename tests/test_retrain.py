"""Unit tests for the retraining regression guard (pure logic)."""

from app.jobs import retrain


def test_is_regression_true_when_auc_drops_beyond_tolerance():
    assert retrain.is_regression(0.70, 0.80, tolerance=0.02) is True


def test_is_regression_false_within_tolerance():
    # 0.79 + 0.02 = 0.81 >= 0.80 → not a regression
    assert retrain.is_regression(0.79, 0.80, tolerance=0.02) is False


def test_is_regression_false_when_improved():
    assert retrain.is_regression(0.85, 0.80) is False


def test_default_tolerance_is_positive():
    assert retrain._AUC_REGRESSION_TOLERANCE > 0
