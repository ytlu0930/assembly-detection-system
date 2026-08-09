from pathlib import Path

import cv2
import numpy as np

from utils.deterministic_correction_annotator import annotate_correction, file_sha256


def test_annotation_created_without_modifying_source(tmp_path):
    source = tmp_path / "source.png"
    assert cv2.imwrite(str(source), np.full((240, 320, 3), 220, dtype=np.uint8))
    before = file_sha256(source)
    output = Path(annotate_correction(
        test_image=source,
        bbox_evidence=[{"candidate_part_id": "PIN", "bbox": [20, 30, 100, 130], "role": "test"}],
        affected_parts=[{"part_id": "PIN", "confidence": .9}], error_type="wrongpart",
        output_path=tmp_path / "result.png", requires_manual_review=False,
    ))
    assert output.is_file() and file_sha256(source) == before
    rendered = cv2.imread(str(output))
    assert rendered.shape[0] == 420


def test_uncertain_annotation_does_not_invent_bbox(tmp_path):
    source = tmp_path / "source.png"
    assert cv2.imwrite(str(source), np.full((100, 120, 3), 220, dtype=np.uint8))
    output = annotate_correction(test_image=source, bbox_evidence=[], affected_parts=[], error_type="uncertain", output_path=tmp_path / "uncertain.png", requires_manual_review=True)
    assert Path(output).is_file()


def test_reference_only_bbox_is_not_drawn_on_test_frame(tmp_path):
    source = tmp_path / "source.png"
    original = np.full((100, 120, 3), 220, dtype=np.uint8)
    assert cv2.imwrite(str(source), original)
    output = annotate_correction(
        test_image=source,
        bbox_evidence=[{"candidate_part_id": "PIN", "bbox": [10, 10, 50, 50], "role": "reference"}],
        affected_parts=[{"part_id": "PIN", "confidence": .9}], error_type="missingpart",
        output_path=tmp_path / "result.png", requires_manual_review=True,
    )
    rendered = cv2.imread(str(output))
    assert np.array_equal(rendered[:100], original)
