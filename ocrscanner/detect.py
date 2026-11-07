"""Automatic document detection and perspective correction.

Several independent detectors propose a 4-corner page outline; each
candidate is scored and the best one wins.  If nothing plausible is found
the image is assumed to already be a flat scan and is returned unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

WORK_SIZE = 1000  # longest side used for detection (speed + robustness)
MIN_AREA_FRAC = 0.15  # page must cover at least this much of the photo
MAX_AREA_FRAC = 0.985  # ...and not be the whole frame border


@dataclass
class Detection:
    corners: np.ndarray | None  # 4x2 float32 in original-image coords (tl,tr,br,bl)
    score: float
    method: str

    @property
    def found(self) -> bool:
        return self.corners is not None


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Return corners ordered top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def _resize(img: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = WORK_SIZE / max(h, w)
    if scale >= 1:
        return img.copy(), 1.0
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA), scale


def _quad_from_contour(cnt: np.ndarray) -> np.ndarray | None:
    peri = cv2.arcLength(cnt, True)
    for eps in (0.01, 0.02, 0.03, 0.05, 0.08):
        approx = cv2.approxPolyDP(cnt, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32)
    # Fall back to the convex hull's best-fitting quadrilateral.
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    for eps in (0.02, 0.04, 0.06, 0.1):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    return None


def _edge_support(quad: np.ndarray, grad: np.ndarray) -> float:
    """Fraction of the outline that lies on a strong image edge."""
    h, w = grad.shape
    q = order_corners(quad)
    hits = total = 0
    for i in range(4):
        p0, p1 = q[i], q[(i + 1) % 4]
        for t in np.linspace(0.05, 0.95, 40):
            x, y = p0 + (p1 - p0) * t
            xi, yi = int(round(x)), int(round(y))
            if xi < 2 or yi < 2 or xi >= w - 2 or yi >= h - 2:
                continue  # parts outside the frame are neutral
            total += 1
            if grad[yi - 2: yi + 3, xi - 2: xi + 3].max() > 25:
                hits += 1
    return hits / total if total else 0.0


def _score_quad(quad: np.ndarray, shape: tuple[int, int], gray: np.ndarray, grad: np.ndarray) -> float:
    """Score a candidate page outline in [0, 1]; higher is better."""
    h, w = shape
    area = cv2.contourArea(quad)
    frac = area / float(h * w)
    if frac < MIN_AREA_FRAC or frac > MAX_AREA_FRAC:
        return 0.0
    q = order_corners(quad)
    # Angles should be close to 90 degrees (perspective allows some slack).
    angle_pen = 0.0
    for i in range(4):
        a, b, c = q[i - 1], q[i], q[(i + 1) % 4]
        v1, v2 = a - b, c - b
        angle_pen += abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))
    angle_score = max(0.0, 1.0 - angle_pen / 2.0)
    # Pages are brighter inside than outside (contrast with background).
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, q.astype(np.int32), 255)
    ring = cv2.dilate(mask, np.ones((25, 25), np.uint8)) - mask
    inside = cv2.mean(gray, mask=mask)[0]
    outside = cv2.mean(gray, mask=ring)[0] if cv2.countNonZero(ring) else inside
    contrast = min(1.0, abs(inside - outside) / 60.0)
    margin = 3
    edge_hits = sum(1 for x, y in q if x < margin or y < margin or x > w - margin or y > h - margin)
    border_score = 1.0 - 0.15 * edge_hits
    support = _edge_support(q, grad)
    return float(
        support ** 2
        * (0.5 + 0.5 * angle_score)
        * (0.5 + 0.5 * contrast)
        * (0.7 + 0.3 * frac ** 0.5)
        * border_score
    )


def _candidates(small: np.ndarray) -> list[tuple[np.ndarray, str]]:
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    out: list[tuple[np.ndarray, str]] = []

    def from_binary(binary: np.ndarray, name: str) -> None:
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
            q = _quad_from_contour(cnt)
            if q is not None:
                out.append((q, name))

    # 1) Edge based (works for most photos).
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    med = float(np.median(blur))
    for lo, hi in ((0.66 * med, 1.33 * med), (30, 120), (75, 200)):
        edges = cv2.Canny(blur, lo, hi)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
        from_binary(edges, "edges")

    # 2) Brightness based: paper is usually the brightest large region.
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, k)  # erase text inside page
    _, otsu = cv2.threshold(cv2.GaussianBlur(closed, (7, 7), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    from_binary(otsu, "brightness")

    # 3) Low-saturation + bright (white paper on coloured table).
    s, v = hsv[..., 1], hsv[..., 2]
    paper = ((s < 60) & (v > np.percentile(v, 40))).astype(np.uint8) * 255
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, k)
    paper = cv2.morphologyEx(paper, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    from_binary(paper, "color")
    return out


def detect_document(image: np.ndarray) -> Detection:
    """Find the page outline in a BGR photo."""
    small, scale = _resize(image)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
    grad = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)) / 4
    best = Detection(None, 0.0, "none")
    for quad, name in _candidates(small):
        sc = _score_quad(quad, gray.shape[:2], gray, grad)
        if sc > best.score:
            best = Detection(order_corners(quad) / scale, sc, name)
    if best.score < 0.2:
        return Detection(None, best.score, "none")
    return best


def _line_intersection(l1: np.ndarray, l2: np.ndarray) -> np.ndarray | None:
    (vx1, vy1, x1, y1), (vx2, vy2, x2, y2) = l1, l2
    a = np.array([[vx1, -vx2], [vy1, -vy2]], np.float64)
    if abs(np.linalg.det(a)) < 1e-6:
        return None
    t = np.linalg.solve(a, np.array([x2 - x1, y2 - y1], np.float64))
    return np.array([x1 + t[0] * vx1, y1 + t[0] * vy1], np.float32)


def refine_corners(gray: np.ndarray, corners: np.ndarray, search: int | None = None) -> np.ndarray:
    """Snap each page side to the strongest edge at full resolution.

    Detection runs on a downscaled copy; here we look along the normal of
    every side for the true paper edge, fit a robust line through those
    points and intersect neighbouring sides to get precise corners.
    """
    h, w = gray.shape[:2]
    if search is None:
        search = max(6, int(max(h, w) / WORK_SIZE * 4))
    smooth = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
    c = order_corners(corners)
    center = c.mean(axis=0)
    lines = []
    for i in range(4):
        p0, p1 = c[i], c[(i + 1) % 4]
        d = p1 - p0
        length = float(np.linalg.norm(d))
        if length < 20:
            return corners
        d /= length
        n = np.array([-d[1], d[0]], np.float32)
        if np.dot(center - (p0 + p1) / 2, n) < 0:
            n = -n  # normal points into the page
        offs = np.arange(-search, search + 1, dtype=np.float32)
        pts = []
        for t in np.linspace(0.08, 0.92, 60):
            base = p0 + d * t * length
            xs = base[0] + n[0] * offs
            ys = base[1] + n[1] * offs
            if xs.min() < 1 or ys.min() < 1 or xs.max() > w - 2 or ys.max() > h - 2:
                continue
            prof = cv2.remap(smooth, xs.reshape(1, -1).astype(np.float32), ys.reshape(1, -1).astype(np.float32), cv2.INTER_LINEAR).ravel()
            grad = np.diff(prof)  # paper (inside) brighter than background -> rising profile
            k = int(np.argmax(np.abs(grad)))
            if abs(grad[k]) < 4:
                continue
            o = (offs[k] + offs[k + 1]) / 2
            pts.append(base + n * o)
        if len(pts) < 10:
            lines.append(None)
            continue
        lines.append(cv2.fitLine(np.array(pts, np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel())
    out = c.copy()
    for i in range(4):
        prev_line, cur_line = lines[i - 1], lines[i]
        if prev_line is None or cur_line is None:
            continue
        p = _line_intersection(prev_line, cur_line)
        if p is not None and np.linalg.norm(p - c[i]) < search * 2:
            out[i] = p
    return out


def warp_document(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Perspective-correct the page to a flat, top-down rectangle."""
    tl, tr, br, bl = order_corners(corners)
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    m = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], np.float32), dst)
    out = cv2.warpPerspective(image, m, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # Trim a hair off every side so no dark table edge remains (it confuses OCR layout).
    ix, iy = max(2, int(width * 0.006)), max(2, int(height * 0.006))
    return out[iy: height - iy, ix: width - ix]


def auto_crop(image: np.ndarray) -> tuple[np.ndarray, Detection]:
    """Detect and flatten the document. Returns (page_image, detection)."""
    det = detect_document(image)
    if not det.found:
        return image, det
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    det.corners = refine_corners(gray, det.corners)
    return warp_document(image, det.corners), det
