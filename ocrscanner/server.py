"""Web app: upload or photograph documents, get text + searchable PDF.

Run:  ocr-scan serve   (or)   uvicorn ocrscanner.server:app
"""

from __future__ import annotations

import base64
import os
import re
import uuid
from collections import OrderedDict
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.concurrency import run_in_threadpool

from .pipeline import ScanResult, decode_bytes, scan_image

MAX_UPLOAD = int(os.environ.get("OCR_MAX_UPLOAD_MB", "40")) * 1024 * 1024
MAX_PAGES = int(os.environ.get("OCR_MAX_PAGES", "50"))
STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Free OCR Scanner")
_results: OrderedDict[str, tuple[ScanResult, str]] = OrderedDict()  # small in-memory cache for downloads


def _remember(res: ScanResult, lang: str) -> str:
    key = uuid.uuid4().hex
    _results[key] = (res, lang)
    while len(_results) > 20:
        _results.popitem(last=False)
    return key


def _thumb(img) -> str:
    h, w = img.shape[:2]
    s = min(1.0, 1000 / max(h, w))
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/api/scan")
async def scan(
    files: list[UploadFile] = File(...),
    lang: str = Form("eng"),
    detect: bool = Form(True),
    fast: bool = Form(False),
) -> dict:
    if not re.fullmatch(r"[A-Za-z_]{2,20}(\+[A-Za-z_]{2,20}){0,4}", lang):
        raise HTTPException(400, "Invalid language code")
    images = []
    for f in files:
        data = await f.read()
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, f"{f.filename} is too large")
        is_pdf = (f.filename or "").lower().endswith(".pdf") or data[:5] == b"%PDF-"
        try:
            for im in decode_bytes(data, f.filename or ""):
                images.append((im, detect and not is_pdf))
        except ValueError as e:
            raise HTTPException(400, f"{f.filename}: {e}") from e
    if not images:
        raise HTTPException(400, "No pages found")
    if len(images) > MAX_PAGES:
        raise HTTPException(413, f"Too many pages (max {MAX_PAGES})")

    def work() -> ScanResult:
        return ScanResult([scan_image(im, lang=lang, detect_document=d, recheck=not fast) for im, d in images])

    try:
        res = await run_in_threadpool(work)
    except Exception as e:  # tesseract missing language etc.
        raise HTTPException(500, f"OCR failed: {e}") from e
    key = _remember(res, lang)
    out = res.to_dict()
    out["id"] = key
    for p, page in zip(out["pages"], res.pages):
        p["preview"] = _thumb(page.scan_image)
        p["size"] = list(page.ocr_image.shape[:2][::-1])
    return out


@app.get("/api/result/{key}.txt")
def result_txt(key: str) -> PlainTextResponse:
    if key not in _results:
        raise HTTPException(404, "Result expired - scan again")
    return PlainTextResponse(_results[key][0].text, headers={"Content-Disposition": 'attachment; filename="scan.txt"'})


@app.get("/api/result/{key}.pdf")
def result_pdf(key: str) -> Response:
    if key not in _results:
        raise HTTPException(404, "Result expired - scan again")
    res, lang = _results[key]
    return Response(res.searchable_pdf(lang), media_type="application/pdf",
                    headers={"Content-Disposition": 'attachment; filename="scan.pdf"'})
