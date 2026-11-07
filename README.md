# OCR-scanner

A **free, private document scanner with OCR**. Photograph a page with your phone, or upload images or PDFs. It finds the page automatically, straightens and cleans it, reads the text, and gives you editable text plus a **searchable PDF**. There are no subscriptions or per-page fees, and your documents never leave your own computer or server.

## 📱 Android app (APK)

**Download:** open the [latest release](../../releases/latest) on your phone and tap **`OCR-Scanner.apk`**. If Android asks, allow *Install unknown apps* for your browser or Files app.

- **Scan document** opens Google's on-device AI document scanner. It finds the page edges, crops, flattens and cleans the page automatically. You can scan many pages, retake pages, or import from the gallery.
- The text is then read **on your phone** by ML Kit's on-device text recognition. It works offline and nothing is uploaded. Languages: English and other Latin-script languages, plus Hindi/Devanagari.
- Words the AI is unsure about are listed so you can check them. You can edit the text, then **copy, share, save as .txt**, or **save a searchable PDF**.
- Needs Android 8.0+. The automatic page scanner needs Google Play services, which almost all phones have. Without it you can still use *Open images from gallery*.

Every push to `main` that changes `android/` builds a new signed APK with GitHub Actions (`.github/workflows/android.yml`) and publishes it as a release. Each new APK installs as an update over the previous one. The source is in `android/`. To build it yourself, run `cd android && ./gradlew assembleRelease` with the Android SDK installed.

> The signing key in `android/keystore/` is public, which is fine for personal use. Before publishing the app on the Play Store, create your own key and pass it through the `OCR_KEYSTORE*` environment variables (repository secrets).

## What it does

| Step | How |
| --- | --- |
| **Automatic page detection** | Three independent detectors (edges, brightness, paper colour) propose page outlines. Each outline is scored by how well its sides sit on real image edges, how rectangular it is and how well it contrasts with the background. The winner's corners are then snapped to the true paper edge at full resolution. Flat scans with no visible background are left untouched. |
| **Perspective correction** | The page is warped to a flat, top-down rectangle. |
| **Orientation** | Sideways and upside-down pages are fixed automatically (0/90/180/270°). |
| **Deskew** | Small tilts are removed with a projection-profile search to 0.05°. |
| **Scale normalisation** | Text is resized to the size the OCR engine reads most accurately. |
| **Cleanup** | Shadow and uneven-lighting removal, denoising, and several binarisations (Otsu, Sauvola) plus a sharpened version. |
| **Ensemble OCR** | Every cleaned version is OCR'd with Tesseract's LSTM network. For each **line**, the most confident reading wins, and lines the engine skipped in one version are recovered from another. |
| **Word re-check** | Any word that is still uncertain is re-read at 2× zoom from every version, and the best reading is kept. |
| **Quality control** | Each page gets a confidence score and a grade (excellent / good / retake recommended). Uncertain words are listed and highlighted on the page image, so nothing doubtful is silently trusted. |
| **Output** | Plain text, JSON with per-word confidence and boxes, cleaned page images, and a searchable PDF whose hidden text layer is the *verified* text. |

## Accuracy

Accuracy is **measured**, not claimed. `benchmarks/run.py` generates realistic phone photos of documents with known text: a random perspective, a textured background, uneven lighting, blur, sensor noise and JPEG compression. It then compares the output with the truth.

Results with the high-accuracy English model, 2400px camera images, 10 photos per level:

| Photo quality | Pages found | Character error rate | Word error rate |
| --- | --- | --- | --- |
| easy | 10/10 | **0.000 %** | 0.00 % |
| medium | 10/10 | **0.000 %** | 0.00 % |
| hard (strong shadows, blur, noise, JPEG q75) | 10/10 | **0.000 %** | 0.00 % |
| stress test: tiny text, blurry 1800px camera (8 photos) | 8/8 | 4.2 % mean. 4 photos perfect, 3 under 1.4 %, and 1 unreadable photo flagged *retake recommended* | 12 % |

Page corners are located to within **~1–2 px** on average, out of 2400 px.

**Be realistic:** an error rate of 0.1 % or lower is achievable, and is what you get, for **printed text in a reasonably sharp photo or scan**. No OCR system can promise it for every input. Handwriting, very blurry or tiny text, heavily damaged paper and unusual fonts will have more errors. That is why the scanner reports its confidence, highlights doubtful words and asks you to retake bad photos rather than quietly returning wrong text.

Reproduce the numbers yourself:

```bash
python benchmarks/run.py --n 10 --difficulty hard -v
```

## Install

You need Python 3.9+ and [Tesseract](https://github.com/tesseract-ocr/tesseract) 5.

On Ubuntu or Debian:

```bash
sudo apt install tesseract-ocr tesseract-ocr-eng
```

On macOS:

```bash
brew install tesseract
```

On Windows, install it from https://github.com/UB-Mannheim/tesseract/wiki.

Then install the scanner:

```bash
pip install -e .
```

Download the most accurate (`tessdata_best`) models. This is recommended:

```bash
./scripts/download_models.sh eng
```

You can add more languages, for example `./scripts/download_models.sh eng hin urd ara`. Models in `./models` are used automatically, and you can point `OCR_TESSDATA` at another folder.

## Use it

### Web app (works from your phone)

```bash
ocr-scan serve --host 0.0.0.0 --port 8000
```

Open `http://<your-computer-ip>:8000` on your phone. Tap **Take photo** and then **Scan**, and you get the text, highlighted uncertain words, *Copy*, *Download .txt* and *Download searchable PDF*. Several photos become pages of one document.

### Command line

```bash
ocr-scan photo.jpg                                   # print the text
ocr-scan page1.jpg page2.jpg -o doc.txt --pdf doc.pdf
ocr-scan scanned.pdf --json details.json             # PDFs too (each page OCR'd)
ocr-scan photo.jpg --save-scan cleaned/              # keep the flattened, cleaned pages
ocr-scan photo.jpg -l eng+hin                        # other languages
ocr-scan photo.jpg --fast                            # skip word re-check (faster)
```

### Python

```python
from ocrscanner import scan_file

result = scan_file("photo.jpg")
print(result.text, result.confidence)
for page in result.pages:
    print(page.quality, page.needs_review)
open("out.pdf", "wb").write(result.searchable_pdf())
```

### Docker (deploy it for free on any server)

```bash
docker build -t ocr-scanner .
```

```bash
docker run -p 8000:8000 ocr-scanner
```

## Tips for the best accuracy

- Use good, even light and avoid shadows from your phone or hand.
- Fill most of the frame with the page, and keep all four corners visible.
- Hold steady and tap to focus. If the result says *retake recommended*, retake the photo.
- Install the `tessdata_best` model for your language (see above).

## Development

```bash
pip install -e ".[dev]"
```

```bash
pytest
```

The tests cover page detection precision, flat scans, rotated pages, end-to-end accuracy (≤ 0.1 % character errors on a realistic photo), the web API, the CLI and the PDF round-trip.

Project layout:

```
ocrscanner/
  detect.py    automatic page detection + perspective correction
  enhance.py   orientation, deskew, scaling, shadow removal, binarisation
  ocr.py       ensemble OCR, line recovery, word re-check
  quality.py   capture-quality grading and warnings
  pdf.py       searchable PDF writer
  pipeline.py  end-to-end scan_image / scan_file
  server.py    web app (FastAPI) + static/index.html
  cli.py       `ocr-scan` command
benchmarks/    synthetic photo generator + accuracy benchmark
tests/
```

## License

MIT
