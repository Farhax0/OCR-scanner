"""Command line interface.

    ocr-scan photo.jpg                      # print text
    ocr-scan scan.pdf -o out.txt --pdf searchable.pdf
    ocr-scan *.jpg --json results.json --save-scan cleaned/
    ocr-scan serve                          # start the web app
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2


def _serve(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="ocr-scan serve")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    import uvicorn

    uvicorn.run("ocrscanner.server:app", host=a.host, port=a.port)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "serve":
        _serve(argv[1:])
        return 0

    from .pipeline import ScanResult, load_images, scan_image

    ap = argparse.ArgumentParser(prog="ocr-scan", description="Free AI document scanner + OCR")
    ap.add_argument("inputs", nargs="+", help="images (jpg/png/tiff/...) or PDFs; all become pages of one document")
    ap.add_argument("-o", "--output", help="write extracted text to this file (default: stdout)")
    ap.add_argument("--pdf", help="write a searchable PDF")
    ap.add_argument("--json", help="write detailed results (confidence, uncertain words, ...) as JSON")
    ap.add_argument("--save-scan", help="directory to save the cleaned, flattened page images")
    ap.add_argument("-l", "--lang", default="eng", help="Tesseract language(s), e.g. eng, hin, eng+fra")
    ap.add_argument("--no-detect", action="store_true", help="skip automatic page detection (input is a flat scan)")
    ap.add_argument("--fast", action="store_true", help="skip word re-checking (faster, slightly less accurate)")
    a = ap.parse_args(argv)

    result = ScanResult()
    for path in a.inputs:
        is_pdf = path.lower().endswith(".pdf")
        for img in load_images(path):
            page = scan_image(img, lang=a.lang, detect_document=not (a.no_detect or is_pdf), recheck=not a.fast)
            result.pages.append(page)
            n = len(result.pages)
            q = page.quality
            print(f"[page {n}] {Path(path).name}: confidence {page.confidence:.1f}%  "
                  f"({q['grade']}, {len(page.needs_review)} words to review, {page.seconds:.1f}s)", file=sys.stderr)
            for w in q["warnings"]:
                print(f"           ! {w}", file=sys.stderr)

    if a.output:
        Path(a.output).write_text(result.text, encoding="utf-8")
    else:
        sys.stdout.write(result.text)
    if a.pdf:
        Path(a.pdf).write_bytes(result.searchable_pdf(a.lang))
    if a.json:
        Path(a.json).write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    if a.save_scan:
        d = Path(a.save_scan)
        d.mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(result.pages, 1):
            cv2.imwrite(str(d / f"page_{i:03d}.png"), p.scan_image)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
