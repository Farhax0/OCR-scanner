import json

import cv2
from fastapi.testclient import TestClient

from ocrscanner.cli import main
from ocrscanner.metrics import cer
from ocrscanner.server import app


def test_web_api(photo_sample):
    photo, _, truth = photo_sample
    ok, buf = cv2.imencode(".jpg", photo)
    client = TestClient(app)
    assert client.get("/").status_code == 200
    r = client.post("/api/scan", files={"files": ("doc.jpg", buf.tobytes(), "image/jpeg")}, data={"fast": "true"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert cer(truth, data["text"]) < 0.005
    assert data["pages"][0]["preview"].startswith("data:image/jpeg")
    pdf = client.get(f"/api/result/{data['id']}.pdf")
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")
    assert client.get(f"/api/result/{data['id']}.txt").text == data["text"]


def test_web_api_rejects_bad_input():
    client = TestClient(app)
    r = client.post("/api/scan", files={"files": ("x.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 400
    r = client.post("/api/scan", files={"files": ("x.jpg", b"x", "image/jpeg")}, data={"lang": "../etc"})
    assert r.status_code == 400


def test_cli(tmp_path, flat_page):
    img, truth = flat_page
    src = tmp_path / "page.png"
    cv2.imwrite(str(src), img)
    out, pdf, js = tmp_path / "o.txt", tmp_path / "o.pdf", tmp_path / "o.json"
    assert main([str(src), "-o", str(out), "--pdf", str(pdf), "--json", str(js), "--fast"]) == 0
    assert cer(truth, out.read_text()) <= 0.001
    assert pdf.read_bytes().startswith(b"%PDF")
    assert json.loads(js.read_text())["pages"][0]["confidence"] > 90
    # PDF input round-trip: the searchable PDF we produced can be scanned again.
    out2 = tmp_path / "o2.txt"
    assert main([str(pdf), "-o", str(out2), "--fast"]) == 0
    assert cer(truth, out2.read_text()) <= 0.002
