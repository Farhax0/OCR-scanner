"""Measure end-to-end accuracy on synthetic document photos.

    python benchmarks/run.py --n 10 --difficulty medium
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from synth import sample  # noqa: E402

from ocrscanner import scan_image  # noqa: E402
from ocrscanner.detect import order_corners  # noqa: E402
from ocrscanner.metrics import cer, wer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--difficulty", default="medium", choices=["easy", "medium", "hard"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--font-px", type=int, default=34, help="text size on a 1700px-wide page")
    ap.add_argument("--photo-width", type=int, default=2400, help="camera image width in px")
    ap.add_argument("--save", help="directory to save sample photos")
    ap.add_argument("-v", "--verbose", action="store_true", help="show each OCR mistake")
    args = ap.parse_args()

    cers, wers, corner_err = [], [], []
    for i in range(args.n):
        photo, corners, truth = sample(args.seed + i, args.difficulty, args.font_px, args.photo_width)
        if args.save:
            import cv2

            Path(args.save).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(f"{args.save}/sample_{i}.jpg", photo)
        r = scan_image(photo)
        c, w = cer(truth, r.text), wer(truth, r.text)
        cers.append(c)
        wers.append(w)
        ce = float("nan")
        if r.corners:
            ce = float(np.max(np.linalg.norm(order_corners(np.array(r.corners)) - order_corners(corners), axis=1)))
        corner_err.append(ce)
        if c > 0 and args.verbose:
            import difflib

            for op in difflib.SequenceMatcher(None, truth.split(), r.text.split()).get_opcodes():
                if op[0] != "equal":
                    print("    ", op[0], truth.split()[op[1]:op[2]], "->", r.text.split()[op[3]:op[4]])
        print(f"#{i:02d} detected={r.document_found!s:5} corner_err={ce:6.1f}px "
              f"CER={c * 100:6.3f}%  WER={w * 100:6.2f}%  conf={r.confidence:5.1f} "
              f"variant={r.variant:7} lines+{r.lines_replaced}/{r.lines_recovered} fixed={r.words_fixed}/{r.words_rechecked} {r.seconds:.1f}s {r.quality['grade']} sharp={r.quality['sharpness']}")
    print(f"\n{args.difficulty}: mean CER {np.mean(cers) * 100:.3f}%  "
          f"(accuracy {100 - np.mean(cers) * 100:.3f}%)  mean WER {np.mean(wers) * 100:.2f}%  "
          f"detection {sum(not np.isnan(x) for x in corner_err)}/{args.n}")


if __name__ == "__main__":
    main()
