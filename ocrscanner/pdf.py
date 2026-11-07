"""Searchable PDF writer.

The page image is embedded and the *verified* OCR text (after ensemble and
re-checking) is laid over it invisibly, word by word, so the PDF can be
searched, selected and copied.  Tesseract's own PDF renderer would re-run
OCR and lose those corrections, so it is only used as a fallback for
scripts the built-in PDF font cannot encode.
"""

from __future__ import annotations

import io
import zlib

import cv2
import numpy as np

A4_WIDTH_IN = 8.27


def _esc(s: bytes) -> bytes:
    return s.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")


def encodable(words) -> bool:
    try:
        for w in words:
            w.text.encode("cp1252")
        return True
    except UnicodeEncodeError:
        return False


def page_pdf(image: np.ndarray, words) -> bytes:
    """Build a one-page PDF: JPEG background + invisible Helvetica text."""
    h, w = image.shape[:2]
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ok, jpg = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, 85])
    jpg = jpg.tobytes()
    dpi = max(72.0, w / A4_WIDTH_IN)
    s = 72.0 / dpi
    pw, ph = w * s, h * s

    ops = [f"q {pw:.2f} 0 0 {ph:.2f} 0 0 cm /Im0 Do Q".encode(), b"BT 3 Tr"]
    for wd in words:
        x, y, bw, bh = wd.box
        if bw <= 0 or bh <= 0:
            continue
        size = max(1.0, bh * s)
        natural = 0.5 * size * len(wd.text)  # approx Helvetica advance
        tz = 100.0 * (bw * s) / natural if natural else 100.0
        baseline = ph - (y + bh) * s + 0.2 * size
        ops.append(
            f"/F1 {size:.2f} Tf {tz:.1f} Tz 1 0 0 1 {x * s:.2f} {baseline:.2f} Tm (".encode()
            + _esc(wd.text.encode("cp1252"))
            + b" ) Tj"
        )
    ops.append(b"ET")
    content = zlib.compress(b"\n".join(ops))

    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {pw:.2f} {ph:.2f}] "
        f"/Resources << /XObject << /Im0 4 0 R >> /Font << /F1 5 0 R >> >> /Contents 6 0 R >>".encode(),
        f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace /DeviceGray "
        f"/BitsPerComponent 8 /Filter /DCTDecode /Length {len(jpg)} >>\nstream\n".encode() + jpg + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        f"<< /Length {len(content)} /Filter /FlateDecode >>\nstream\n".encode() + content + b"\nendstream",
    ]
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for i, o in enumerate(objs, 1):
        offsets.append(out.tell())
        out.write(f"{i} 0 obj\n".encode() + o + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode())
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return out.getvalue()


def merge(pdfs: list[bytes]) -> bytes:
    if len(pdfs) == 1:
        return pdfs[0]
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument.new()
    for d in pdfs:
        doc.import_pages(pdfium.PdfDocument(d))
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
