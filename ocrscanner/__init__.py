"""Free, local, AI-assisted document scanner with OCR.

Pipeline:  photo/scan -> automatic document detection -> perspective
correction -> orientation & skew fix -> image cleanup -> multi-pass OCR
-> confidence-checked text, searchable PDF.
"""

from .pipeline import ScanResult, scan_image, scan_file

__all__ = ["ScanResult", "scan_image", "scan_file"]
__version__ = "1.0.0"
