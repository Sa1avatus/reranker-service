import pytest

from reranker_service.parity import ranking_parity, score_deviation


def test_identical_rankings_have_perfect_parity() -> None:
    result = ranking_parity(["a", "b", "c"], ["a", "b", "c"])

    assert result["spearman"] == 1
    assert result["kendall_tau"] == 1
    assert result["top_1_agreement"] == 1
    assert result["top_3_overlap"] == 1


def test_reversed_rankings_report_disagreement() -> None:
    result = ranking_parity(["a", "b", "c"], ["c", "b", "a"])

    assert result["spearman"] == -1
    assert result["kendall_tau"] == -1
    assert result["top_1_agreement"] == 0


def test_score_deviation_and_invalid_inputs() -> None:
    assert score_deviation({"a": 1.0, "b": 0.0}, {"a": 0.9, "b": 0.2}) == pytest.approx(
        {"mean_absolute_deviation": 0.15, "max_absolute_deviation": 0.2}
    )
    with pytest.raises(ValueError):
        ranking_parity(["a"], ["b"])
    with pytest.raises(ValueError):
        score_deviation({}, {})
