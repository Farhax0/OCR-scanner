"""Generate realistic synthetic document photos with known ground truth.

Each sample: text rendered on a page -> placed in perspective on a
textured background -> uneven lighting, blur, noise, JPEG compression.
"""

from __future__ import annotations

import random
import textwrap
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIRS = ["/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation",
             "/usr/share/fonts/truetype/freefont"]
FONTS = [
    "DejaVuSans.ttf", "DejaVuSerif.ttf", "LiberationSerif-Regular.ttf",
    "LiberationSans-Regular.ttf", "FreeSerif.ttf", "FreeSans.ttf",
]

SENTENCES = [
    "The quarterly report shows revenue increased by 12.5% compared to last year.",
    "Please sign and return the enclosed agreement no later than March 14, 2026.",
    "Invoice number INV-20481 is due within thirty days of the date of issue.",
    "All employees must complete the safety training before operating machinery.",
    "The committee reviewed 47 applications and selected 9 candidates for interview.",
    "For questions about your account, call customer support at 555-0142.",
    "Total amount payable: $1,284.60 including taxes and shipping charges.",
    "This document is confidential and intended only for the named recipient.",
    "Meeting minutes were approved unanimously at the board session on Friday.",
    "Store the product in a cool, dry place away from direct sunlight.",
    "The warranty covers manufacturing defects for a period of two years.",
    "Our new office is located at 221 Baker Street, Suite 400, London.",
    "Results indicate a strong correlation between sleep quality and focus.",
    "Delivery was delayed because of severe weather conditions in the region.",
    "Section 4.2 describes the procedure for handling customer complaints.",
    "Kindly verify the information below and report any discrepancies promptly.",
]


def find_font(name: str) -> str | None:
    for d in FONT_DIRS:
        p = Path(d, name)
        if p.exists():
            return str(p)
    return None


def make_text(rng: random.Random, n: int = 8) -> str:
    paras = []
    for _ in range(rng.randint(2, 4)):
        paras.append(" ".join(rng.choice(SENTENCES) for _ in range(rng.randint(2, n // 2 + 1))))
    return "\n\n".join(paras)


def render_page(text: str, rng: random.Random, width: int = 1700, font_px: int = 34) -> np.ndarray:
    fonts = [f for f in (find_font(n) for n in FONTS) if f]
    font = ImageFont.truetype(rng.choice(fonts), font_px)
    height = int(width * 1.414)
    img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)
    margin = int(width * 0.08)
    chars = int((width - 2 * margin) / (font_px * 0.52))
    y = margin
    for para in text.split("\n\n"):
        for line in textwrap.wrap(para, chars):
            draw.text((margin, y), line, fill=rng.randint(0, 40), font=font)
            y += int(font_px * 1.5)
        y += int(font_px * 0.8)
    return np.array(img)


def background(h: int, w: int, rng: random.Random) -> np.ndarray:
    base = np.array([rng.randint(30, 140) for _ in range(3)], np.float32)
    noise = cv2.resize(np.random.default_rng(rng.randint(0, 1 << 30)).random((h // 40, w // 40, 3)).astype(np.float32),
                       (w, h), interpolation=cv2.INTER_CUBIC)
    bg = base + (noise - 0.5) * 60
    return np.clip(bg, 0, 255).astype(np.uint8)


def photograph(page: np.ndarray, rng: random.Random, difficulty: str = "medium", width: int = 2400):
    ph, pw = page.shape
    W, H = width, width * 4 // 3
    bg = background(H, W, rng)
    jitter = {"easy": 0.03, "medium": 0.07, "hard": 0.11}[difficulty]
    scale = rng.uniform(0.62, 0.78)
    cw, ch = W * scale, W * scale * ph / pw
    cx, cy = W / 2 + rng.uniform(-W / 30, W / 30), H / 2 + rng.uniform(-W / 30, W / 30)
    base = np.array([[cx - cw / 2, cy - ch / 2], [cx + cw / 2, cy - ch / 2],
                     [cx + cw / 2, cy + ch / 2], [cx - cw / 2, cy + ch / 2]], np.float32)
    dst = base + np.array([[rng.uniform(-1, 1) * W * jitter, rng.uniform(-1, 1) * W * jitter] for _ in range(4)],
                          np.float32)
    src = np.array([[0, 0], [pw, 0], [pw, ph], [0, ph]], np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    page_bgr = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)
    warped = cv2.warpPerspective(page_bgr, m, (W, H), flags=cv2.INTER_CUBIC)
    mask = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), m, (W, H))
    img = np.where(mask[..., None] > 127, warped, bg).astype(np.float32)
    # Uneven lighting + slight warm tint.
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    lx, ly = rng.uniform(0, W), rng.uniform(0, H)
    light = 1.0 - {"easy": 0.15, "medium": 0.3, "hard": 0.45}[difficulty] * (
        np.hypot(xx - lx, yy - ly) / np.hypot(W, H))
    img *= light[..., None]
    img *= np.array([rng.uniform(0.9, 1.0), rng.uniform(0.95, 1.0), 1.0], np.float32)
    sigma = {"easy": 0.6, "medium": 1.0, "hard": 1.4}[difficulty]
    img = cv2.GaussianBlur(img, (0, 0), sigma)
    img += np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, {"easy": 3, "medium": 6, "hard": 9}[difficulty], img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    q = {"easy": 92, "medium": 85, "hard": 75}[difficulty]
    img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
    return img, dst


def sample(seed: int, difficulty: str = "medium", font_px: int = 34, photo_width: int = 2400):
    rng = random.Random(seed)
    text = make_text(rng)
    page = render_page(text, rng, font_px=font_px)
    photo, corners = photograph(page, rng, difficulty, photo_width)
    return photo, corners, text
