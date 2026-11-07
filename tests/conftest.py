import random
import sys
from pathlib import Path

import cv2
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

import synth  # noqa: E402


@pytest.fixture(scope="session")
def photo_sample():
    """A realistic phone photo of a document, plus ground truth."""
    return synth.sample(3, "medium")


@pytest.fixture(scope="session")
def flat_page():
    rng = random.Random(11)
    text = synth.make_text(rng)
    page = synth.render_page(text, rng)
    return cv2.cvtColor(page, cv2.COLOR_GRAY2BGR), text
