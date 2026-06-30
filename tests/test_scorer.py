"""Unit tests for the heuristic prior scorer (pure functions)."""

from app.ranking.scorer import WEIGHTS, ScoreResult, score_features


def test_each_component_uses_its_weight():
    features = {
        "avg_watch_ratio": 1.0,
        "like_rate": 1.0,
        "share_rate": 1.0,
        "comment_rate": 1.0,
        "save_rate": 1.0,
        "completion_rate": 1.0,
        "freshness": 1.0,
        "skip_rate": 1.0,
    }
    result = score_features(features)
    assert isinstance(result, ScoreResult)
    for component, weight in WEIGHTS.items():
        assert result.components[component] == weight
    assert result.score == sum(WEIGHTS.values())


def test_skip_is_a_negative_signal():
    base = {
        "avg_watch_ratio": 0.0,
        "like_rate": 0.0,
        "share_rate": 0.0,
        "comment_rate": 0.0,
        "save_rate": 0.0,
        "completion_rate": 0.0,
        "freshness": 0.0,
    }
    skipped = score_features({**base, "skip_rate": 1.0}).score
    not_skipped = score_features({**base, "skip_rate": 0.0}).score
    assert skipped < not_skipped


def test_missing_features_default_to_zero():
    result = score_features({})
    assert result.score == 0.0
    assert all(v == 0.0 for v in result.components.values())


def test_non_numeric_values_are_ignored():
    result = score_features({"like_rate": "abc", "share_rate": None})
    assert result.score == 0.0


def test_nan_and_infinity_are_treated_as_zero():
    assert score_features({"like_rate": float("nan")}).score == 0.0
    assert score_features({"like_rate": float("inf")}).score == 0.0
    assert score_features({"like_rate": float("-inf")}).score == 0.0
    # Strings that float() would parse to NaN/Inf are also neutralized.
    assert score_features({"like_rate": "nan"}).score == 0.0
    assert score_features({"like_rate": "inf"}).score == 0.0


def test_higher_engagement_scores_higher():
    assert score_features({"like_rate": 0.9}).score > score_features({"like_rate": 0.1}).score
