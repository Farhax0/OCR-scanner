"""Image cleanup that makes OCR as accurate as possible."""

from __future__ import annotations

import re

import cv2
import numpy as np
import pytesseract

TARGET_XHEIGHT = 22  # px; Tesseract LSTM is most accurate around 20-30 px x-height


def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


_ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def _rotate(img: np.ndarray, rot: int) -> np.ndarray:
    return img if rot == 0 else cv2.rotate(img, _ROT[rot])


def _read_score(gray: np.ndarray) -> float:
    """How well does Tesseract read this as horizontal text? (conf x amount).

    PSM 6 is used because it does not auto-rotate vertical text lines, so a
    sideways page cannot score well.
    """
    d = pytesseract.image_to_data(gray, config="--oem 1 --psm 6", output_type=pytesseract.Output.DICT)
    score = 0.0
    for t, c in zip(d["text"], d["conf"]):
        t, c = t.strip(), float(c)
        if t and c > 0 and sum(ch.isalnum() for ch in t) >= 2:
            score += len(t) * (c / 100.0) ** 2
    return score


def _lines_are_horizontal(gray: np.ndarray) -> bool:
    """Text lines make the row-ink profile 'spiky'; compare against columns."""
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    def spikiness(prof: np.ndarray) -> float:
        prof = prof.astype(np.float64)
        return float(np.sum(np.diff(prof) ** 2) / (np.sum(prof ** 2) + 1e-6))

    return spikiness(binary.sum(axis=1)) >= spikiness(binary.sum(axis=0))


def fix_orientation(img: np.ndarray) -> tuple[np.ndarray, int]:
    """Rotate by 0/90/180/270 so text is upright.

    1. The text-line direction (horizontal vs vertical) is measured from the
       ink profile, which is unambiguous.
    2. Of the two rotations consistent with it, Tesseract's orientation
       detector picks one when it is confident; otherwise both are test-read
       and the more legible wins.
    """
    gray = to_gray(img)
    h, w = gray.shape
    scale = 1600 / max(h, w)
    probe = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    probe = cv2.fastNlMeansDenoising(remove_shading(probe), None, h=7)
    cands = (0, 180) if _lines_are_horizontal(probe) else (90, 270)
    rot, conf = -1, 0.0
    try:
        osd = pytesseract.image_to_osd(probe, config="--psm 0")
        rot = int(re.search(r"Rotate: (\d+)", osd).group(1))
        conf = float(re.search(r"Orientation confidence: ([\d.]+)", osd).group(1))
    except (pytesseract.TesseractError, AttributeError):
        pass
    if rot in cands and conf >= 5.0:
        best = rot
    else:
        scores = {r: _read_score(_rotate(probe, r)) for r in cands}
        best = max(scores, key=scores.get)
        # Keep the page as-is unless the flip is clearly better.
        if cands[0] == 0 and best == 180 and scores[180] < scores[0] * 1.1 + 5:
            best = 0
    return _rotate(img, best), best


def estimate_skew(gray: np.ndarray) -> float:
    """Estimate small text skew (degrees) via projection-profile search."""
    h, w = gray.shape
    scale = 1200 / max(h, w)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    binary = cv2.adaptiveThreshold(small, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    if cv2.countNonZero(binary) < 50:
        return 0.0
    center = (binary.shape[1] / 2, binary.shape[0] / 2)

    def sharpness(angle: float) -> float:
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        rot = cv2.warpAffine(binary, m, binary.shape[::-1], flags=cv2.INTER_NEAREST)
        prof = rot.sum(axis=1, dtype=np.float64)
        return float(np.sum(np.diff(prof) ** 2))

    best = max(np.arange(-10, 10.01, 0.5), key=sharpness)
    best = max(np.arange(best - 0.5, best + 0.51, 0.05), key=sharpness)
    return float(best)


def deskew(img: np.ndarray) -> tuple[np.ndarray, float]:
    angle = estimate_skew(to_gray(img))
    if abs(angle) < 0.1:
        return img, 0.0
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    out = cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return out, angle


def _line_heights(gray: np.ndarray) -> np.ndarray:
    """Heights of text lines from the horizontal ink profile of text columns."""
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    h, w = binary.shape
    heights: list[int] = []
    # Narrow vertical strips keep lines separate even if the page is slightly curved.
    strips = max(1, w // 300)
    for i in range(strips):
        col = binary[:, i * w // strips:(i + 1) * w // strips]
        prof = (col > 0).sum(axis=1)
        ink = prof > max(2, col.shape[1] * 0.02)
        run = 0
        for v in np.append(ink, False):
            if v:
                run += 1
            elif run:
                if 3 <= run <= h * 0.08:
                    heights.append(run)
                run = 0
    return np.array(heights)


def estimate_xheight(gray: np.ndarray) -> float | None:
    """Estimate the lowercase x-height of the body text in pixels."""
    lh = _line_heights(gray)
    if len(lh) >= 5:
        # The dense ink band of a line is ~1.3x the x-height (calibrated on common fonts).
        return float(np.median(lh)) / 1.3
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    hs, ws, areas = stats[1:, 3], stats[1:, 2], stats[1:, 4]
    keep = (hs >= 4) & (hs <= gray.shape[0] * 0.1) & (ws <= hs * 4) & (areas >= 8)
    if keep.sum() < 20:
        return None
    return float(np.percentile(hs[keep], 35))


def normalize_scale(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Resize so text x-height is in Tesseract's sweet spot."""
    xh = estimate_xheight(to_gray(img))
    if not xh:
        return img, 1.0
    factor = float(np.clip(TARGET_XHEIGHT / xh, 0.5, 4.0))
    if 0.85 < factor < 1.25:
        return img, 1.0
    interp = cv2.INTER_CUBIC if factor > 1 else cv2.INTER_AREA
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=interp), factor


def remove_shading(gray: np.ndarray) -> np.ndarray:
    """Flatten uneven lighting / shadows by dividing out the background."""
    k = max(15, (min(gray.shape) // 25) | 1)
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (k, k), 0)
    norm = cv2.divide(gray, bg, scale=255)
    return cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX)


def sauvola(gray: np.ndarray, window: int = 31, k: float = 0.2) -> np.ndarray:
    """Sauvola adaptive binarisation (robust for documents)."""
    g = gray.astype(np.float64)
    mean = cv2.boxFilter(g, -1, (window, window))
    sq = cv2.boxFilter(g * g, -1, (window, window))
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    thresh = mean * (1 + k * (std / 128.0 - 1))
    return np.where(g > thresh, 255, 0).astype(np.uint8)


def variants(img: np.ndarray) -> dict[str, np.ndarray]:
    """Produce several cleaned versions; OCR picks whichever reads best."""
    gray = to_gray(img)
    flat = remove_shading(gray)
    den = cv2.fastNlMeansDenoising(flat, None, h=7, templateWindowSize=7, searchWindowSize=21)
    _, otsu = cv2.threshold(den, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    soft = cv2.GaussianBlur(den, (0, 0), 1.5)
    sharp = cv2.addWeighted(den, 1.8, soft, -0.8, 0)
    return {
        "gray": gray,
        "sharp": sharp,
        "flat": den,
        "otsu": otsu,
        "sauvola": sauvola(den),
    }


def make_clean_scan(img: np.ndarray) -> np.ndarray:
    """A pleasant 'scanned' look for display / PDF background."""
    gray = remove_shading(to_gray(img))
    return cv2.fastNlMeansDenoising(gray, None, h=5)
