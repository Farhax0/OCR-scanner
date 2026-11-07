import numpy as np
import pytest
import synth

from ocrscanner.detect import auto_crop, detect_document, order_corners


def test_order_corners():
    pts = np.array([[10, 90], [90, 10], [10, 10], [90, 90]], np.float32)
    assert order_corners(pts).tolist() == [[10, 10], [90, 10], [90, 90], [10, 90]]


@pytest.mark.parametrize("seed,difficulty", [(1, "easy"), (2, "medium"), (4, "hard"), (8, "hard")])
def test_finds_page_precisely(seed, difficulty):
    photo, corners, _ = synth.sample(seed, difficulty)
    page, det = auto_crop(photo)
    assert det.found
    err = np.abs(det.corners - order_corners(corners)).max()
    assert err < 5, f"corner error {err:.1f}px"
    assert page.shape[0] > page.shape[1]  # portrait page recovered


def test_flat_scan_is_left_alone(flat_page):
    img, _ = flat_page
    assert not detect_document(img).found
