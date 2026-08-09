from scripts.evaluate_affected_part_identity import evaluate


def gt(image_id, parts, *, composite=False, error_type="missing", view="front"):
    return {"image_id": image_id, "ground_truth_part_ids": parts, "review_status": "confirmed", "is_composite": str(composite).lower(), "error_type": error_type, "view_angle": view, "case_id": image_id}


def pred(image_id, parts, confidences="0.95", **extra):
    return {"image_id": image_id, "predicted_part_ids": parts, "predicted_confidence": confidences, **extra}


def test_exact_partial_composite_and_part_metrics():
    metrics = evaluate(
        [gt("one", "A|B", composite=True), gt("two", "C")],
        [pred("one", "A|X", "0.95|0.95"), pred("two", "C")],
    )
    summary = metrics["summary"]
    assert summary["exact_set_match_accuracy"] == 0.5
    assert summary["at_least_one_part_recall"] == 1.0
    assert summary["all_parts_recall"] == 0.5
    assert summary["part_level_precision"] == 2 / 3
    assert summary["part_level_recall"] == 2 / 3
    assert summary["part_level_f1"] == 2 / 3
    assert summary["composite_full_recall"] == 0.0


def test_empty_correct_set_unknown_and_correct_false_positive():
    metrics = evaluate(
        [gt("correct", "", error_type="correct"), gt("error", "A")],
        [pred("correct", "X"), pred("error", "UNKNOWN")],
    )
    assert metrics["summary"]["correct_control_false_positive_rate"] == 1.0
    assert metrics["summary"]["unknown_part_rate"] == 0.5


def test_false_confident_thresholds_use_identity_and_case_denominators():
    metrics = evaluate(
        [gt("one", "A"), gt("two", "B")],
        [pred("one", "X|A", "0.75|0.95"), pred("two", "Y", "0.85")],
    )
    assert metrics["false_confident"]["0.70"]["false_confident_count"] == 2
    assert metrics["false_confident"]["0.70"]["high_confidence_prediction_count"] == 3
    assert metrics["false_confident"]["0.80"]["false_confident_identity_rate"] == 0.5
    assert metrics["false_confident"]["0.80"]["false_confident_case_rate"] == 0.5
    assert metrics["false_confident"]["0.90"]["false_confident_count"] == 0


def test_zero_denominators_are_null_not_zero_percent():
    metrics = evaluate([gt("correct", "", error_type="correct")], [pred("correct", "", "")])
    summary = metrics["summary"]
    assert summary["at_least_one_part_recall"] is None
    assert summary["part_level_precision"] is None
    assert summary["unknown_part_rate"] is None
    assert metrics["false_confident"]["0.80"]["false_confident_identity_rate"] is None
    assert all(item["empirical_accuracy"] is None for item in metrics["confidence_bins"])


def test_verifier_escape_is_counted_without_relaxing_verifier():
    metrics = evaluate(
        [gt("one", "A")],
        [pred("one", "X", verifier_status="verified", verified_part_ids="X")],
    )
    assert metrics["verifier"]["acceptance_rate"] == 1.0
    assert metrics["verifier"]["wrong_identity_escaped_verifier_count"] == 1


def test_candidate_violation_metrics_distinguish_high_confidence():
    metrics = evaluate(
        [gt("one", "A"), gt("two", "B")],
        [pred("one", "X", "0.95", candidate_constraint_status="violation"), pred("two", "B", "0.70", candidate_constraint_status="valid")],
    )
    assert metrics["summary"]["candidate_violation_rate"] == 0.5
    assert metrics["summary"]["high_confidence_candidate_violation_rate"] == 1.0
