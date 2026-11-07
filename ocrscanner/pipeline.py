"""End-to-end scanning pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import detect, enhance, ocr, pdf, quality


@dataclass
class PageResult:
    text: str
    confidence: float  # 0-100, character-weighted mean word confidence
    needs_review: list[dict]  # words the engine is not sure about
    document_found: bool
    detection_method: str
    corners: list[list[float]] | None
    rotation: int
    skew: float
    variant: str
    lines_replaced: int
    lines_recovered: int
    words_rechecked: int
    words_fixed: int
    quality: dict  # capture-quality grade + warnings
    words: list  # final OCR words with boxes (for the PDF text layer)
    scan_image: np.ndarray  # cleaned, flattened page (for display)
    ocr_image: np.ndarray  # image actually used for OCR
    seconds: float

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 2),
            "needs_review": self.needs_review,
            "document_found": self.document_found,
            "detection_method": self.detection_method,
            "corners": self.corners,
            "rotation": self.rotation,
            "skew": round(self.skew, 2),
            "variant": self.variant,
            "lines_replaced": self.lines_replaced,
            "lines_recovered": self.lines_recovered,
            "words_rechecked": self.words_rechecked,
            "words_fixed": self.words_fixed,
            "quality": self.quality,
            "seconds": round(self.seconds, 2),
        }


@dataclass
class ScanResult:
    pages: list[PageResult] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\f".join(p.text for p in self.pages)

    @property
    def confidence(self) -> float:
        return float(np.mean([p.confidence for p in self.pages])) if self.pages else 0.0

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 2),
            "pages": [p.to_dict() for p in self.pages],
        }

    def searchable_pdf(self, lang: str = "eng") -> bytes:
        """PDF with the scanned pages and an invisible, searchable text layer."""
        out = []
        for p in self.pages:
            if pdf.encodable(p.words):
                out.append(pdf.page_pdf(p.scan_image, p.words))
            else:  # non-Latin script: let Tesseract embed its Unicode font
                out.append(ocr.tesseract_pdf(p.ocr_image, lang))
        return pdf.merge(out)


def scan_image(
    image: np.ndarray,
    lang: str = "eng",
    detect_document: bool = True,
    recheck: bool = True,
) -> PageResult:
    """Scan a single BGR image (photo or flat scan)."""
    t0 = time.time()
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    det = detect.Detection(None, 0.0, "disabled")
    page = image
    if detect_document:
        page, det = detect.auto_crop(image)

    page, rotation = enhance.fix_orientation(page)
    page, skew = enhance.deskew(page)
    page, _ = enhance.normalize_scale(page)

    variants = enhance.variants(page)
    res = ocr.recognize(variants, lang=lang, recheck=recheck)

    chars = sum(len(w.text) for w in res.words)
    low = sum(len(w.text) for w in res.low_conf_words)
    q = quality.assess(enhance.to_gray(page), res.mean_conf, low / chars if chars else 0.0, chars)

    return PageResult(
        text=res.text,
        confidence=res.mean_conf,
        needs_review=[{"text": w.text, "confidence": round(w.conf, 1), "box": list(w.box)} for w in res.low_conf_words],
        document_found=det.found,
        detection_method=det.method,
        corners=det.corners.round(1).tolist() if det.found else None,
        rotation=rotation,
        skew=skew,
        variant=res.variant,
        lines_replaced=res.lines_replaced,
        lines_recovered=res.lines_recovered,
        words_rechecked=res.rechecked,
        words_fixed=res.fixed,
        quality=q,
        words=res.words,
        scan_image=enhance.make_clean_scan(page),
        ocr_image=res.image,
        seconds=time.time() - t0,
    )


def load_images(path: str | Path, dpi: int = 300) -> list[np.ndarray]:
    """Load an image file or every page of a PDF as BGR arrays."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(path))
        pages = []
        for p in pdf:
            pil = p.render(scale=dpi / 72).to_pil().convert("RGB")
            pages.append(cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR))
        return pages
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return [img]


def decode_bytes(data: bytes, filename: str = "") -> list[np.ndarray]:
    if filename.lower().endswith(".pdf") or data[:5] == b"%PDF-":
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(data)
        out = []
        for p in pdf:
            pil = p.render(scale=300 / 72).to_pil().convert("RGB")
            out.append(cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR))
        return out
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unsupported or corrupt image")
    return [img]


def scan_pages(images: list[np.ndarray], **kw) -> ScanResult:
    return ScanResult([scan_image(im, **kw) for im in images])


def scan_file(path: str | Path, **kw) -> ScanResult:
    """Scan an image file or a (multi-page) PDF.

    PDFs are already flat, so document detection is skipped for them.
    """
    is_pdf = Path(path).suffix.lower() == ".pdf"
    if is_pdf:
        kw.setdefault("detect_document", False)
    return scan_pages(load_images(path), **kw)
