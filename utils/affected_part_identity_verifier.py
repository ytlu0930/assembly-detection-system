"""Evidence-based affected-part identity verification.

The verifier never maps a case name to a part and never treats model confidence,
taxonomy membership, or expected-inventory presence as sufficient proof.  It
requires relation evidence that compares the test assembly with its correct
reference before an identity can be marked verified.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


IdentityStatus = Literal["verified", "conflict", "uncertain", "unresolved"]
VALID_IDENTITY_STATUSES = {"verified", "conflict", "uncertain", "unresolved"}


@dataclass(frozen=True)
class IdentityVerificationResult:
    predicted_part_id: str
    verified_part_id: str | None
    identity_status: IdentityStatus
    identity_confidence: float
    evidence: list[dict[str, Any]] = field(default_factory=list)
    alternative_candidates: list[dict[str, Any]] = field(default_factory=list)
    requires_manual_review: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AffectedPartIdentityVerifier:
    """Verify a predicted affected-part identity from contrastive evidence."""

    def __init__(
        self,
        *,
        verification_threshold: float = 0.80,
        candidate_threshold: float = 0.85,
        candidate_margin: float = 0.10,
        localization_score_threshold: float = 0.15,
        nms_iou_threshold: float = 0.50,
    ) -> None:
        self.verification_threshold = self._unit(verification_threshold)
        self.candidate_threshold = self._unit(candidate_threshold)
        self.candidate_margin = self._unit(candidate_margin)
        self.localization_score_threshold = self._unit(localization_score_threshold)
        self.nms_iou_threshold = self._unit(nms_iou_threshold)

    def verify(
        self,
        *,
        error_report: dict[str, Any],
        test_image_metadata: dict[str, Any] | None,
        reference_image_metadata: dict[str, Any] | None,
        expected_state: dict[str, Any] | None,
        localization_evidence: dict[str, Any] | None,
        part_library: dict[str, Any] | None,
    ) -> IdentityVerificationResult:
        predicted = str(error_report.get("part_id") or "unknown_part").strip()
        error_type = str(error_report.get("error_type") or "uncertain").lower()
        expected = expected_state if isinstance(expected_state, dict) else {}
        library = part_library if isinstance(part_library, dict) else {}
        raw_evidence = localization_evidence if isinstance(localization_evidence, dict) else {}
        evidence: list[dict[str, Any]] = [
            {
                "kind": "input_metadata",
                "test": self._safe_metadata(test_image_metadata),
                "reference": self._safe_metadata(reference_image_metadata),
            }
        ]

        expected_count = sum(
            1
            for part in expected.get("expected_parts", [])
            if isinstance(part, dict) and str(part.get("part_id") or "") == predicted
        )
        evidence.append(
            {
                "kind": "expected_inventory",
                "part_id": predicted,
                "expected_count": expected_count,
                "note": "Inventory presence alone is not identity verification.",
            }
        )

        alternatives = self._rank_candidates(raw_evidence.get("candidate_evidence"), library)

        if not predicted or predicted.lower().startswith("unknown") or predicted not in library:
            evidence.append({"kind": "identity", "result": "unknown_or_out_of_taxonomy"})
            return self._result(predicted, None, "unresolved", 0.0, evidence, alternatives)

        reference = self._summarize_localization(raw_evidence.get("reference_localization"))
        test = self._summarize_localization(raw_evidence.get("test_localization"))
        evidence.extend(
            [
                {"kind": "reference_localization", **reference},
                {"kind": "test_localization", **test},
            ]
        )

        explicit_relation = raw_evidence.get("relation_supported")
        if isinstance(explicit_relation, bool):
            relation_score = self._unit(raw_evidence.get("relation_confidence", 1.0))
            evidence.append(
                {
                    "kind": "explicit_relation",
                    "supported": explicit_relation,
                    "confidence": relation_score,
                }
            )
            if explicit_relation and relation_score >= self.verification_threshold:
                return self._result(predicted, predicted, "verified", relation_score, evidence, alternatives)
            if not explicit_relation and relation_score >= self.verification_threshold:
                return self._result(predicted, None, "conflict", relation_score, evidence, alternatives)

        cross_view = self._cross_view_support(raw_evidence.get("cross_view_consistency"), predicted)
        if cross_view is not None:
            evidence.append({"kind": "cross_view_consistency", **cross_view})

        if error_type == "missingpart":
            status, confidence = self._verify_missing(expected_count, reference, test)
        elif error_type == "extrapart":
            status, confidence = self._verify_extra(reference, test)
        elif error_type in {"wrongpart", "positionerror", "criticalerror"}:
            status, confidence = self._verify_relation_only(cross_view)
        else:
            status, confidence = "uncertain", 0.25

        if status == "verified":
            return self._result(predicted, predicted, status, confidence, evidence, alternatives)
        if status == "conflict":
            replacement = self._verified_alternative(alternatives)
            if replacement is not None:
                evidence.append({"kind": "alternative_selected", **replacement})
                return self._result(
                    predicted,
                    str(replacement["part_id"]),
                    "verified",
                    float(replacement["score"]),
                    evidence,
                    alternatives,
                )
            return self._result(predicted, None, status, confidence, evidence, alternatives)
        return self._result(predicted, None, status, confidence, evidence, alternatives)

    def _verify_missing(
        self,
        expected_count: int,
        reference: dict[str, Any],
        test: dict[str, Any],
    ) -> tuple[IdentityStatus, float]:
        if expected_count <= 0:
            return "conflict", 1.0
        if not reference["reliable"] or not test["reliable"]:
            return "uncertain", 0.30
        reference_count = reference.get("count")
        test_count = test.get("count")
        if reference_count is None or test_count is None:
            return "uncertain", 0.40
        confidence = min(float(reference["confidence"]), float(test["confidence"]))
        if reference_count > test_count:
            return ("verified", confidence) if confidence >= self.verification_threshold else ("uncertain", confidence)
        return "conflict", confidence

    def _verify_extra(
        self,
        reference: dict[str, Any],
        test: dict[str, Any],
    ) -> tuple[IdentityStatus, float]:
        if not reference["reliable"] or not test["reliable"]:
            return "uncertain", 0.30
        reference_count = reference.get("count")
        test_count = test.get("count")
        if reference_count is None or test_count is None:
            return "uncertain", 0.40
        confidence = min(float(reference["confidence"]), float(test["confidence"]))
        if test_count > reference_count:
            return ("verified", confidence) if confidence >= self.verification_threshold else ("uncertain", confidence)
        return "conflict", confidence

    def _verify_relation_only(self, cross_view: dict[str, Any] | None) -> tuple[IdentityStatus, float]:
        if not cross_view:
            return "uncertain", 0.30
        confidence = float(cross_view["confidence"])
        if cross_view["supports_prediction"] and confidence >= self.verification_threshold:
            return "verified", confidence
        if not cross_view["supports_prediction"] and confidence >= self.verification_threshold:
            return "conflict", confidence
        return "uncertain", confidence

    def _summarize_localization(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {"status": "missing", "reliable": False, "present": None, "count": None, "confidence": 0.0}
        status = str(value.get("status") or "unknown").lower()
        explicit_count = value.get("estimated_count", value.get("distinct_detection_count"))
        count: int | None
        try:
            count = max(0, int(explicit_count)) if explicit_count is not None else self._distinct_detection_count(value)
        except (TypeError, ValueError):
            count = None
        score = self._unit(
            value.get(
                "identity_evidence_confidence",
                value.get("selected_detection_score", value.get("detection_score", 0.0)),
            )
        )
        present_value = value.get("present")
        present = bool(present_value) if isinstance(present_value, bool) else (count > 0 if count is not None else None)
        reliable_value = value.get("identity_evidence_reliable")
        reliable = (
            bool(reliable_value)
            if isinstance(reliable_value, bool)
            else status == "success" and count is not None and score >= self.localization_score_threshold
        )
        return {
            "status": status,
            "reliable": reliable,
            "present": present,
            "count": count,
            "confidence": score,
            "selected_bbox": value.get("selected_bbox"),
        }

    def _distinct_detection_count(self, value: dict[str, Any]) -> int | None:
        detections = value.get("all_detections")
        if not isinstance(detections, list):
            bbox = value.get("selected_bbox")
            return 1 if isinstance(bbox, (list, tuple)) and len(bbox) == 4 else None
        candidates: list[tuple[float, list[float]]] = []
        for item in detections:
            if not isinstance(item, dict):
                continue
            try:
                score = float(item.get("score", 0.0))
                bbox = [float(number) for number in item.get("bbox", [])]
            except (TypeError, ValueError):
                continue
            if score >= self.localization_score_threshold and len(bbox) == 4:
                candidates.append((score, bbox))
        selected: list[list[float]] = []
        for _, bbox in sorted(candidates, reverse=True):
            if all(self._iou(bbox, kept) < self.nms_iou_threshold for kept in selected):
                selected.append(bbox)
        return len(selected)

    def _rank_candidates(self, value: Any, library: dict[str, Any]) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        ranked: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            part_id = str(item.get("part_id") or "").strip()
            if not part_id or part_id not in library:
                continue
            score = self._unit(item.get("score", 0.0))
            raw_evidence = item.get("evidence")
            candidate_evidence = raw_evidence if isinstance(raw_evidence, list) else [raw_evidence] if raw_evidence else []
            ranked.append(
                {
                    "part_id": part_id,
                    "score": score,
                    "evidence": candidate_evidence,
                    "difference_supported": bool(item.get("difference_supported", False)),
                }
            )
        return sorted(ranked, key=lambda item: (-float(item["score"]), str(item["part_id"])))

    def _verified_alternative(self, alternatives: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not alternatives:
            return None
        first = alternatives[0]
        runner_up = float(alternatives[1]["score"]) if len(alternatives) > 1 else 0.0
        if (
            bool(first.get("difference_supported"))
            and float(first["score"]) >= self.candidate_threshold
            and float(first["score"]) - runner_up >= self.candidate_margin
        ):
            return first
        return None

    def _cross_view_support(self, value: Any, predicted: str) -> dict[str, Any] | None:
        if not isinstance(value, list) or not value:
            return None
        relevant = [item for item in value if isinstance(item, dict) and str(item.get("part_id") or "") == predicted]
        if not relevant:
            return None
        supports = [bool(item.get("supports_prediction")) for item in relevant]
        confidences = [self._unit(item.get("confidence", 0.0)) for item in relevant]
        support_ratio = sum(supports) / len(supports)
        confidence = sum(confidences) / len(confidences)
        return {
            "views": len(relevant),
            "support_ratio": support_ratio,
            "supports_prediction": support_ratio >= 0.75,
            "confidence": confidence,
        }

    @staticmethod
    def _safe_metadata(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        allowed = {"image_name", "relative_path", "model_id", "step_id", "view_angle", "path"}
        return {key: value[key] for key in allowed if key in value}

    @staticmethod
    def _iou(left: list[float], right: list[float]) -> float:
        x1, y1 = max(left[0], right[0]), max(left[1], right[1])
        x2, y2 = min(left[2], right[2]), min(left[3], right[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if intersection <= 0:
            return 0.0
        left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
        right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
        union = left_area + right_area - intersection
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _unit(value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _result(
        predicted: str,
        verified: str | None,
        status: IdentityStatus,
        confidence: float,
        evidence: list[dict[str, Any]],
        alternatives: list[dict[str, Any]],
    ) -> IdentityVerificationResult:
        if status not in VALID_IDENTITY_STATUSES:
            raise ValueError(f"Unsupported identity status: {status}")
        return IdentityVerificationResult(
            predicted_part_id=predicted or "unknown_part",
            verified_part_id=verified,
            identity_status=status,
            identity_confidence=max(0.0, min(1.0, float(confidence))),
            evidence=evidence,
            alternative_candidates=alternatives,
            requires_manual_review=status != "verified",
        )
