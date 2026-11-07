"""Automatic capture-quality assessment, so bad photos are flagged, not trusted."""

from __future__ import annotations

import cv2
import numpy as np


def sharpness(gray: np.ndarray) -> float:
    """Edge sharpness of text strokes (higher = sharper), roughly scale-invariant."""
    h, w = gray.shape
    scale = 1500 / max(h, w)
    g = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    strong = mag[mag > np.percentile(mag, 99)]
    lap = cv2.Laplacian(g, cv2.CV_32F)
    return float(np.mean(np.abs(lap[mag > np.percentile(mag, 99)])) / (np.mean(strong) + 1e-6) * 100)


def assess(page_gray: np.ndarray, confidence: float, low_conf_frac: float, text_chars: int) -> dict:
    warnings: list[str] = []
    sharp = sharpness(page_gray)
    # Paper (median) vs. darkest ink; most of a page is blank paper.
    contrast = float(np.median(page_gray) - np.percentile(page_gray, 0.5))
    # Sharpness alone cannot separate "soft but readable" from "unreadable"
    # (they overlap), so blur is only blamed when the OCR also struggled.
    if confidence < 85 and sharp < 7.0:
        warnings.append("Image looks blurry - hold the camera steady and tap to focus, then rescan.")
    if contrast < 60:
        warnings.append("Low contrast - add more light or avoid shadows on the page.")
    if text_chars == 0:
        warnings.append("No text found.")
    if confidence < 85 and text_chars:
        warnings.append("Low OCR confidence - please check the highlighted words or rescan closer.")
    if low_conf_frac > 0.05:
        warnings.append(f"{low_conf_frac:.0%} of the text is uncertain.")
    if confidence >= 92 and low_conf_frac <= 0.01 and not warnings:
        grade = "excellent"
    elif confidence >= 85 and low_conf_frac <= 0.05:
        grade = "good"
    else:
        grade = "retake recommended"
    return {"grade": grade, "sharpness": round(sharp, 1), "contrast": round(contrast, 1), "warnings": warnings}
