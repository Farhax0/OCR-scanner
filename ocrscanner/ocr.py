"""Multi-pass OCR with a line-level ensemble and word re-checking.

Every cleaned variant of the page is OCR'd.  The most trustworthy pass is
used as the base; then, line by line, a more confident reading from another
variant replaces the base line, and lines that the base pass missed
entirely (Tesseract's layout analysis occasionally drops one) are
recovered.  Finally, any word that is still doubtful is re-read at higher
zoom from every variant.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

LOW_CONF = 80.0  # words below this confidence are re-checked / flagged
PAD = 32  # white border added around the page before OCR


def tessdata_dir(lang: str = "eng") -> str | None:
    """Prefer high-accuracy 'tessdata_best' models when installed for every requested language."""
    for c in (os.environ.get("OCR_TESSDATA"), str(Path(__file__).resolve().parent.parent / "models")):
        if c and all(Path(c, f"{lg}.traineddata").exists() for lg in lang.split("+")):
            return c
    return None


def base_config(psm: int = 3, extra: str = "", lang: str = "eng") -> str:
    cfg = f"--oem 1 --psm {psm} -c preserve_interword_spaces=1"
    d = tessdata_dir(lang)
    if d:
        cfg = f'--tessdata-dir "{d}" ' + cfg
    return f"{cfg} {extra}".strip()


@dataclass
class Word:
    text: str
    conf: float
    box: tuple[int, int, int, int]  # x, y, w, h in the (unpadded) page image


@dataclass
class Line:
    words: list[Word]
    block: int
    par: int
    source: str = ""

    @property
    def box(self) -> tuple[int, int, int, int]:
        xs = [w.box[0] for w in self.words]
        ys = [w.box[1] for w in self.words]
        x2 = [w.box[0] + w.box[2] for w in self.words]
        y2 = [w.box[1] + w.box[3] for w in self.words]
        return min(xs), min(ys), max(x2) - min(xs), max(y2) - min(ys)

    @property
    def chars(self) -> int:
        return sum(len(w.text) for w in self.words)

    @property
    def conf(self) -> float:
        n = self.chars
        return sum(len(w.text) * w.conf for w in self.words) / n if n else 0.0

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


@dataclass
class OcrResult:
    lines: list[Line]
    variant: str
    image: np.ndarray
    rechecked: int = 0
    fixed: int = 0
    lines_replaced: int = 0
    lines_recovered: int = 0

    @property
    def words(self) -> list[Word]:
        return [w for ln in self.lines for w in ln.words]

    @property
    def text(self) -> str:
        out: list[str] = []
        prev = None
        for ln in self.lines:
            key = (ln.block, ln.par)
            if prev is not None and key != prev:
                out.append("")
            out.append(ln.text)
            prev = key
        return ("\n".join(out).strip() + "\n") if out else ""

    @property
    def mean_conf(self) -> float:
        ws = self.words
        n = sum(len(w.text) for w in ws)
        return sum(len(w.text) * w.conf for w in ws) / n if n else 0.0

    @property
    def low_conf_words(self) -> list[Word]:
        return [w for w in self.words if w.conf < LOW_CONF]


def _pad(img: np.ndarray) -> np.ndarray:
    return cv2.copyMakeBorder(img, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=255)


def _run(img: np.ndarray, lang: str, name: str = "") -> list[Line]:
    d = pytesseract.image_to_data(_pad(img), lang=lang, config=base_config(3, lang=lang), output_type=Output.DICT)
    lines: dict[tuple[int, int, int], Line] = {}
    order: list[tuple[int, int, int]] = []
    for i, txt in enumerate(d["text"]):
        txt = txt.strip()
        conf = float(d["conf"][i])
        if not txt or conf < 0:
            continue
        key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
        if key not in lines:
            lines[key] = Line([], key[0], key[1], name)
            order.append(key)
        box = (d["left"][i] - PAD, d["top"][i] - PAD, d["width"][i], d["height"][i])
        lines[key].words.append(Word(txt, conf, box))
    return [lines[k] for k in order]


def _quality(lines: list[Line]) -> float:
    """Score an OCR pass: confident, plentiful, low garbage."""
    words = [w for ln in lines for w in ln.words]
    if not words:
        return 0.0
    chars = np.array([len(w.text) for w in words], float)
    conf = np.array([w.conf for w in words], float)
    mean = (chars * conf).sum() / chars.sum()
    low_frac = chars[conf < LOW_CONF].sum() / chars.sum()
    junk = sum(1 for w in words if sum(ch.isalnum() for ch in w.text) == 0 and len(w.text) > 1)
    return float(mean - 25 * low_frac - 2 * junk + min(chars.sum(), 3000) / 300)


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection over the smaller box."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0, min(ay + ah, by + bh) - max(ay, by))
    small = min(aw * ah, bw * bh)
    return iw * ih / small if small else 0.0


def _ensemble(base: OcrResult, others: list[tuple[str, list[Line]]]) -> None:
    for name, lines in others:
        for cand in lines:
            if not cand.words:
                continue
            matches = [i for i, ln in enumerate(base.lines) if _overlap(ln.box, cand.box) > 0.5]
            if not matches:
                # Line missing from the base pass: recover it if it is credible.
                if cand.conf >= 75 and cand.chars >= 3 and sum(c.isalnum() for c in cand.text) >= 3:
                    _insert_line(base, cand)
                    base.lines_recovered += 1
                continue
            if len(matches) != 1:
                continue
            i = matches[0]
            cur = base.lines[i]
            # Same line read better elsewhere (similar length, clearly more confident).
            if (
                cand.conf > cur.conf + 3
                and abs(cand.chars - cur.chars) <= max(2, cur.chars // 10)
                and _overlap(cur.box, cand.box) > 0.8
            ):
                base.lines[i] = Line(cand.words, cur.block, cur.par, name)
                base.lines_replaced += 1


def _insert_line(base: OcrResult, line: Line) -> None:
    """Insert a recovered line after the closest line above it in the same column."""
    x, y, w, h = line.box
    best_i, best_y = -1, -1e9
    for i, ln in enumerate(base.lines):
        lx, ly, lw, lh = ln.box
        if ly + lh / 2 < y + h / 2 and min(x + w, lx + lw) - max(x, lx) > 0 and ly > best_y:
            best_i, best_y = i, ly
    if best_i >= 0:
        ref = base.lines[best_i]
        base.lines.insert(best_i + 1, Line(line.words, ref.block, ref.par, line.source))
    else:
        ref = base.lines[0] if base.lines else None
        base.lines.insert(0, Line(line.words, ref.block if ref else 0, ref.par if ref else 0, line.source))


def _recheck_word(variants: dict[str, np.ndarray], word: Word, lang: str) -> Word:
    """Re-read a doubtful word from every variant at higher zoom; keep the best."""
    x, y, w, h = word.box
    pad = max(4, h // 3)
    best = word
    for img in variants.values():
        crop = img[max(0, y - pad): y + h + pad, max(0, x - pad): x + w + pad]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        crop = cv2.copyMakeBorder(crop, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
        try:
            d = pytesseract.image_to_data(crop, lang=lang, config=base_config(8, lang=lang), output_type=Output.DICT)
        except pytesseract.TesseractError:
            continue
        toks = [(t.strip(), float(c)) for t, c in zip(d["text"], d["conf"]) if t.strip() and float(c) >= 0]
        if len(toks) != 1:
            continue
        txt, conf = toks[0]
        # Must clearly beat the original and keep the same shape: short tokens
        # (digits, initials) must keep their exact length so a neighbour
        # glyph leaking into the crop cannot be glued on.
        max_diff = 0 if len(word.text) <= 3 else max(1, len(word.text) // 4)
        if conf > best.conf + 5 and abs(len(txt) - len(word.text)) <= max_diff:
            best = Word(txt, conf, word.box)
    return best


def recognize(variants: dict[str, np.ndarray], lang: str = "eng", recheck: bool = True) -> OcrResult:
    """OCR every variant, merge them line by line, then re-check weak words."""
    passes = [(name, _run(img, lang, name)) for name, img in variants.items()]
    passes.sort(key=lambda p: _quality(p[1]), reverse=True)
    name, lines = passes[0]
    base = OcrResult(list(lines), name, variants[name])
    good_others = [p for p in passes[1:] if p[1] and _quality(p[1]) > 60]
    _ensemble(base, good_others)
    if recheck:
        for ln in base.lines:
            for i, w in enumerate(ln.words):
                if w.conf < LOW_CONF and base.rechecked < 200:
                    base.rechecked += 1
                    nw = _recheck_word(variants, w, lang)
                    if nw is not w:
                        ln.words[i] = nw
                        base.fixed += 1
    return base


def tesseract_pdf(img: np.ndarray, lang: str = "eng") -> bytes:
    """Tesseract's own PDF renderer (needs pdf.ttf next to the models)."""
    d = tessdata_dir(lang)
    cfg = base_config(3, lang=lang) if d and Path(d, "pdf.ttf").exists() else "--oem 1 --psm 3"
    return pytesseract.image_to_pdf_or_hocr(img, lang=lang, extension="pdf", config=cfg)
