import cv2
import pytest

from ocrscanner import scan_image
from ocrscanner.metrics import cer, wer


def test_metrics():
    assert cer("hello world", "hello world") == 0
    assert cer("abcd", "abce") == 0.25
    assert wer("a b c d", "a b x d") == 0.25
    assert cer("a  b\nc", "a b c") == 0  # whitespace-normalised


def test_photo_end_to_end(photo_sample):
    photo, _, truth = photo_sample
    r = scan_image(photo)
    assert r.document_found
    assert cer(truth, r.text) <= 0.001, r.text  # <= 0.1 % character errors
    assert r.confidence > 90
    assert r.quality["grade"] in ("excellent", "good")


@pytest.mark.parametrize("rot", [cv2.ROTATE_180, cv2.ROTATE_90_CLOCKWISE])
def test_rotated_pages(flat_page, rot):
    img, truth = flat_page
    r = scan_image(cv2.rotate(img, rot))
    assert cer(truth, r.text) <= 0.001
