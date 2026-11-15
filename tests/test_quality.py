import pytest

from optimization_explorer.quality import detection_metrics, score_cases


def test_score_cases_distinguishes_fallback_and_unmatched_predicates():
    source = {"source_file": "f.py", "source_function": "dispatch"}
    gold = [
        dict(source, id="fused", predicate="enabled", expected_optimization=True),
        dict(source, id="fallback", predicate="not enabled", expected_optimization=False),
    ]
    cases = [
        dict(source, case_id="a", applicability_predicate=" enabled ", fusion_type="epilogue"),
        dict(source, case_id="b", applicability_predicate="not enabled", fusion_type="vertical"),
        dict(source, case_id="c", applicability_predicate="UNKNOWN", fusion_type="none"),
    ]
    result = score_cases(gold, cases)
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (1, 1, 0, 0)
    assert result["unmatched_case_ids"] == ["c"]


def test_detection_metrics_with_known_confusion_counts():
    result = detection_metrics(
        {"found": True, "missed": True, "extra": False, "correctly_ignored": False},
        ["found", "extra", "found"],
    )
    assert result == {
        "tp": 1, "fp": 1, "fn": 1, "tn": 1,
        "precision": 0.5, "recall": 0.5, "f1": 0.5,
        "f2": 0.5,
        "false_positive_rate": 0.5, "false_negative_rate": 0.5,
    }


def test_detection_metrics_requires_labels_and_preserves_undefined_rates():
    with pytest.raises(ValueError, match="outside the labeled set"):
        detection_metrics({"known": True}, ["unknown"])
    assert detection_metrics({"negative": False}, [])["recall"] is None
