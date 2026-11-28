#!/usr/bin/env python3
"""
ocr-advanced.py — Multi-variant, multi-PSM OCR pipeline for omnius.

Implements a full preprocessing + OCR + cross-reference pipeline:
  1. Load image → grayscale → 2x upscale
  2. Generate 7 preprocessing variants (two adaptive windows, OTSU, two fixed
     thresholds, two sharpen kernels, denoise)
  3. Run Tesseract with PSM 4, 6, 11 on each variant (up to 21 passes)
  4. Score results using combined heuristic (confidence * coverage + line bonus)
  5. Optionally extract regions (header/body/footer) with cross-reference
  6. Output as JSON, text, CSV, or write all formats to an output directory

Usage:
  python3 ocr-advanced.py <image_or_dir> [options]

Single image:
  python3 ocr-advanced.py photo.jpg --output json
  python3 ocr-advanced.py scan.png --output-dir ./ocr_out --regions

Batch directory:
  python3 ocr-advanced.py ./images/ --output-dir ./ocr_out --batch

Output (JSON to stdout):
  {
    "text": "best extracted text",
    "confidence": 85.2,
    "variant": "otsu_psm6",
    "lines": 42,
    "all_variants": { ... },
    "regions": { ... }
  }
"""

import sys
import os
import json
import csv
import argparse
import signal
import time
from collections import OrderedDict
from pathlib import Path

def check_deps():
    """Check that required Python packages are available."""
    missing = []
    try:
        import cv2
    except ImportError:
        missing.append("cv2")
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    try:
        import pytesseract
    except ImportError:
        missing.append("pytesseract")
    try:
        from PIL import Image
    except ImportError:
        missing.append("PIL")

    if missing:
        print(json.dumps({
            "error": f"OCR runtime imports became unavailable: {', '.join(missing)}. "
                     "Run POST /v1/ocr/setup and poll GET /v1/ocr/readiness; inference never installs packages.",
            "missing": missing,
        }))
        sys.exit(1)

check_deps()

import cv2
import numpy as np
import pytesseract
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Image preprocessing variants
# ---------------------------------------------------------------------------

def to_grayscale(img):
    """Convert BGR to grayscale if needed."""
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def upscale_2x(gray):
    """2x bicubic upscale for better OCR character recognition."""
    h, w = gray.shape
    return cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)


def variant_adaptive_wide(gray):
    """Adaptive Gaussian threshold — wide window (31px), handles gradual lighting."""
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )


def variant_adaptive_fine(gray):
    """Adaptive Gaussian threshold — fine window (11px), catches small text detail."""
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )


def variant_otsu(gray):
    """OTSU threshold — optimal global threshold for bimodal images."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def variant_fixed_140(gray):
    """Fixed threshold 140 — standard cutoff for dark text on light paper."""
    _, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
    return binary


def variant_fixed_150(gray):
    """Fixed threshold 150 — slightly brighter cutoff for lighter scans."""
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return binary


def variant_sharpen_laplacian_otsu(gray):
    """Laplacian sharpen + OTSU — aggressive edge enhancement."""
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def variant_sharpen_unsharp_otsu(gray):
    """Unsharp mask sharpen + OTSU — gentler enhancement, better for photos."""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def variant_denoise_otsu(gray):
    """Denoise + OTSU — removes JPEG artifacts and photo noise."""
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


ALL_VARIANTS = {
    "adaptive_wide":   variant_adaptive_wide,
    "adaptive_fine":   variant_adaptive_fine,
    "otsu":            variant_otsu,
    "fixed_140":       variant_fixed_140,
    "fixed_150":       variant_fixed_150,
    "sharpen_lap":     variant_sharpen_laplacian_otsu,
    "sharpen_unsharp": variant_sharpen_unsharp_otsu,
    "denoise":         variant_denoise_otsu,
}

PSM_MODES = {
    4: "single_block",
    6: "default",
    11: "sparse",
}

# Advanced OCR is a bounded recovery pipeline, not a blind Cartesian product.
# A 327x333 crop previously paid for 24 variants/PSMs and two Tesseract child
# processes per attempt. Small inputs now begin with two high-yield attempts;
# larger/low-evidence images expand only while budget remains.
SMALL_CROP_AREA_PX = 512_000
MEDIUM_IMAGE_AREA_PX = 2_000_000
MAX_PIPELINE_DEADLINE_MS = 80_000
MAX_TESSERACT_ATTEMPT_SECONDS = 12.0
# Small conditioned crops normally complete in well under a second. A
# six-second cap still permits a busy Tesseract child, while keeping a failed
# recovery pair from consuming most of the REST deadline.
SMALL_CROP_TESSERACT_ATTEMPT_SECONDS = 6.0
MIN_TESSERACT_ATTEMPT_SECONDS = 0.25
MIN_ACCEPTED_CONFIDENCE = 50.0
MIN_SUBSTANTIVE_TEXT_CHARS = 12
HIGH_VOLUME_GARBAGE_CHARS = 64
HIGH_VOLUME_GARBAGE_CONFIDENCE = 35.0
TERMINAL_SYMBOL_GARBAGE_CHARS = 24
TERMINAL_SYMBOL_GARBAGE_ALNUM_RATIO = 0.20
ACTIVE_DEADLINE = None


class OcrPipelineTimeout(RuntimeError):
    pass


class OcrPipelineCancelled(RuntimeError):
    pass


def cancellation_signal_handler(signum, _frame):
    raise OcrPipelineCancelled(f"received signal {signum}")


class OcrDeadline:
    def __init__(self, deadline_ms):
        self.deadline_ms = max(1_000, min(int(deadline_ms), MAX_PIPELINE_DEADLINE_MS))
        self.started = time.monotonic()

    def remaining_seconds(self):
        return self.deadline_ms / 1000.0 - (time.monotonic() - self.started)

    def check(self, stage):
        if self.remaining_seconds() <= 0:
            raise OcrPipelineTimeout(f"OCR deadline exceeded during {stage}")

    def tesseract_timeout_seconds(self, attempt_cap_seconds=MAX_TESSERACT_ATTEMPT_SECONDS):
        self.check("Tesseract scheduling")
        return max(
            MIN_TESSERACT_ATTEMPT_SECONDS,
            min(attempt_cap_seconds, self.remaining_seconds()),
        )


def diagnostic(code, message, deadline, stage, attempts_completed=0, attempts_planned=0):
    return {
        "schema": "omnius.ocr-diagnostic.v1",
        "code": code,
        "message": message,
        "stage": stage,
        "deadline_ms": deadline.deadline_ms,
        "attempts_completed": attempts_completed,
        "attempts_planned": attempts_planned,
    }


# ---------------------------------------------------------------------------
# OCR execution
# ---------------------------------------------------------------------------

def text_from_tesseract_data(data):
    """Reconstruct line breaks from one TSV pass; do not spawn Tesseract twice."""
    lines = OrderedDict()
    texts = data.get("text", [])
    for index, raw_text in enumerate(texts):
        text = str(raw_text or "").strip()
        if not text:
            continue
        key = tuple(
            int(data.get(field, [0] * len(texts))[index] or 0)
            for field in ("block_num", "par_num", "line_num")
        )
        lines.setdefault(key, []).append(text)
    return "\n".join(" ".join(words) for words in lines.values()).strip()


def run_tesseract(binary_img, deadline, language="eng", psm=6,
                  attempt_cap_seconds=MAX_TESSERACT_ATTEMPT_SECONDS):
    """Run one bounded Tesseract TSV pass and derive text plus confidence."""
    deadline.check("Tesseract")
    pil_img = Image.fromarray(binary_img)
    config = f"--psm {psm}"
    data = pytesseract.image_to_data(
        pil_img,
        lang=language,
        config=config,
        output_type=pytesseract.Output.DICT,
        timeout=deadline.tesseract_timeout_seconds(attempt_cap_seconds),
    )
    text = text_from_tesseract_data(data)
    confs = []
    for value in data.get("conf", []):
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            continue
        if confidence >= 0:
            confs.append(confidence)
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    line_count = len([line for line in text.split("\n") if line.strip()])
    return text, avg_conf, line_count


def assess_ocr_evidence(text, confidence, line_count):
    """Classify OCR output before it can become agent-visible evidence."""
    normalized = str(text or "").strip()
    chars = len(normalized)
    if chars == 0:
        return {
            "state": "low_information",
            "accepted": False,
            "reason": "no_readable_text",
            "chars": 0,
            "confidence": round(float(confidence), 1),
            "lines": int(line_count),
        }
    alnum_ratio = sum(character.isalnum() for character in normalized) / max(1, chars)
    if chars < MIN_SUBSTANTIVE_TEXT_CHARS and confidence < MIN_ACCEPTED_CONFIDENCE:
        return {
            "state": "low_information",
            "accepted": False,
            "reason": "insufficient_low_confidence_text",
            "chars": chars,
            "confidence": round(float(confidence), 1),
            "lines": int(line_count),
            "alnum_ratio": round(alnum_ratio, 3),
        }
    if (
        (chars >= HIGH_VOLUME_GARBAGE_CHARS and confidence < HIGH_VOLUME_GARBAGE_CONFIDENCE)
        or confidence < MIN_ACCEPTED_CONFIDENCE
        or alnum_ratio < 0.45
    ):
        reason = (
            "high_volume_low_confidence_text"
            if chars >= HIGH_VOLUME_GARBAGE_CHARS and confidence < HIGH_VOLUME_GARBAGE_CONFIDENCE
            else "low_confidence_or_symbol_heavy_text"
        )
        return {
            "state": "rejected",
            "accepted": False,
            "reason": reason,
            "chars": chars,
            "confidence": round(float(confidence), 1),
            "lines": int(line_count),
            "alnum_ratio": round(alnum_ratio, 3),
        }
    return {
        "state": "accepted",
        "accepted": True,
        "reason": "confidence_and_text_quality_met",
        "chars": chars,
        "confidence": round(float(confidence), 1),
        "lines": int(line_count),
        "alnum_ratio": round(alnum_ratio, 3),
    }


def is_terminal_small_crop_rejection(evidence):
    """Whether a first small-crop pass proves another recovery pass is futile.

    Blanks and plausible alphanumeric low-confidence text remain recoverable
    through the second variant. A large very-low-confidence transcript or a
    distinctly symbol-heavy stream cannot become safe evidence through another
    PSM6 pass, and should not make callers wait for one.
    """
    if evidence.get("state") != "rejected":
        return False
    if evidence.get("reason") == "high_volume_low_confidence_text":
        return True
    return (
        evidence.get("chars", 0) >= TERMINAL_SYMBOL_GARBAGE_CHARS
        and evidence.get("alnum_ratio", 1.0) < TERMINAL_SYMBOL_GARBAGE_ALNUM_RATIO
    )


def compute_score(text, confidence, line_count, evidence=None):
    """Combined scoring heuristic:
      - confidence * sqrt(char_count)  — rewards quality and coverage
      - + line_count * 10              — bonus for structured output (more lines = better parse)
    The agent discovered that line-count is a strong proxy for successful parsing
    on structured documents like invoices and forms."""
    quality = evidence or assess_ocr_evidence(text, confidence, line_count)
    char_count = len(text)
    if not quality["accepted"] or char_count == 0:
        return 0
    return confidence * (char_count ** 0.5) + line_count * 10


def extract_region(gray, y_start_pct, y_end_pct, x_start_pct=0, x_end_pct=100):
    """Extract a region from the image by percentage coordinates."""
    h, w = gray.shape
    y1 = int(h * y_start_pct / 100)
    y2 = int(h * y_end_pct / 100)
    x1 = int(w * x_start_pct / 100)
    x2 = int(w * x_end_pct / 100)
    return gray[y1:y2, x1:x2]


def extract_pixel_region(gray, x, y, w, h):
    """Extract a region by pixel coordinates."""
    return gray[y:y+h, x:x+w]


def build_ocr_plan(image_area_px, single_psm=None):
    """Return the smallest credible variant/PSM plan for the effective image."""
    psm_modes = [single_psm] if single_psm else (
        [6] if image_area_px <= SMALL_CROP_AREA_PX
        else [6, 11] if image_area_px <= MEDIUM_IMAGE_AREA_PX
        else [6, 11, 4]
    )
    variants = (
        ["otsu", "adaptive_fine"] if image_area_px <= SMALL_CROP_AREA_PX
        else ["otsu", "adaptive_fine", "denoise", "sharpen_unsharp"]
        if image_area_px <= MEDIUM_IMAGE_AREA_PX
        else list(ALL_VARIANTS.keys())
    )
    # OTSU/PSM6 is deliberately first: a legible result ends the search
    # instead of paying for redundant variants that cannot improve the answer.
    return [(variant, psm) for psm in psm_modes for variant in variants]


def has_sufficient_evidence(text, confidence, line_count):
    quality = assess_ocr_evidence(text, confidence, line_count)
    return quality["accepted"] and confidence >= 70.0 and line_count >= 1


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_txt(text, output_path):
    """Write plain text output."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)


def write_csv(text, output_path):
    """Write CSV with line numbers."""
    lines = [l for l in text.split("\n") if l.strip()]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Line_Number", "Extracted_Text"])
        for i, line in enumerate(lines, 1):
            writer.writerow([i, line])


def write_pdf(text, output_path):
    """Write searchable PDF using reportlab (if available)."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas as pdf_canvas
        from reportlab.lib.units import inch
    except ImportError:
        return False

    lines = [l for l in text.split("\n") if l.strip()]
    c = pdf_canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    y = height - 1 * inch

    for line in lines:
        if y < 1 * inch:
            c.showPage()
            y = height - 1 * inch
        c.drawString(1 * inch, y, line)
        y -= 14

    c.save()
    return True


def write_all_outputs(text, base_name, output_dir):
    """Write TXT + CSV + PDF to output directory."""
    os.makedirs(output_dir, exist_ok=True)

    txt_path = os.path.join(output_dir, f"{base_name}.txt")
    write_txt(text, txt_path)

    csv_path = os.path.join(output_dir, f"{base_name}.csv")
    write_csv(text, csv_path)

    pdf_path = os.path.join(output_dir, f"{base_name}.pdf")
    pdf_ok = write_pdf(text, pdf_path)

    return {
        "txt": txt_path,
        "csv": csv_path,
        "pdf": pdf_path if pdf_ok else None,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_variant_plan(gray, plan, deadline, language, debug_dir=None, debug_prefix="full",
                     attempt_cap_seconds=MAX_TESSERACT_ATTEMPT_SECONDS,
                     stop_on_terminal_rejection=False):
    """Run a bounded plan, stopping early once a legible result is proven."""
    all_results = {}
    ocr_errors = []
    best_key = None
    best_score = -1
    early_exit = False
    terminal_rejection = False

    for attempt_index, (vname, psm) in enumerate(plan):
        deadline.check("preprocessing")
        key = f"{vname}_psm{psm}"
        try:
            binary = ALL_VARIANTS[vname](gray)
        except Exception as error:
            ocr_errors.append(f"{key}: preprocessing failed: {error}")
            continue
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}_{vname}.png"), binary)
        try:
            text, confidence, line_count = run_tesseract(
                binary, deadline, language, psm, attempt_cap_seconds
            )
        except OcrPipelineTimeout:
            raise
        except OcrPipelineCancelled:
            raise
        except Exception as error:
            ocr_errors.append(f"{key}: {error}")
            continue
        char_count = len(text)
        evidence = assess_ocr_evidence(text, confidence, line_count)
        score = compute_score(text, confidence, line_count, evidence)
        all_results[key] = {
            "text": text,
            "chars": char_count,
            "lines": line_count,
            "confidence": round(confidence, 1),
            "score": round(score, 1),
            "evidence": evidence,
        }
        if evidence["accepted"] and score > best_score:
            best_score = score
            best_key = key
        if has_sufficient_evidence(text, confidence, line_count):
            early_exit = True
            break
        if (
            stop_on_terminal_rejection
            and attempt_index == 0
            and is_terminal_small_crop_rejection(evidence)
        ):
            terminal_rejection = True
            break

    return all_results, ocr_errors, best_key, early_exit, terminal_rejection


def run_pipeline(image_path, deadline, language="eng", do_regions=False, debug_dir=None,
                 single_psm=None, pixel_region=None, output_dir=None):
    """Run bounded OCR. The effective crop controls plan size, not the full frame."""
    deadline.check("image load")
    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"Could not load image: {image_path}"}

    h_orig, w_orig = img.shape[:2]
    gray = to_grayscale(img)
    if pixel_region:
        rx, ry, rw, rh = pixel_region
        if rx >= w_orig or ry >= h_orig:
            return {"error": "Requested OCR region lies outside the image"}
        gray = extract_pixel_region(gray, rx, ry, rw, rh)
    if gray.size == 0:
        return {"error": "Requested OCR region is empty"}

    effective_area = int(gray.shape[0] * gray.shape[1])
    deadline.check("image conditioning")
    gray_2x = upscale_2x(gray)
    plan = build_ocr_plan(effective_area, single_psm)
    is_small_crop = effective_area <= SMALL_CROP_AREA_PX
    attempt_cap_seconds = (
        SMALL_CROP_TESSERACT_ATTEMPT_SECONDS
        if is_small_crop
        else MAX_TESSERACT_ATTEMPT_SECONDS
    )
    attempts_completed = 0
    try:
        all_results, ocr_errors, best_key, early_exit, terminal_rejection = run_variant_plan(
            gray_2x,
            plan,
            deadline,
            language,
            debug_dir,
            attempt_cap_seconds=attempt_cap_seconds,
            stop_on_terminal_rejection=is_small_crop,
        )
        attempts_completed = len(all_results)
        if not best_key:
            if not all_results:
                detail = ocr_errors[-1] if ocr_errors else "no preprocessing variant completed"
                return {"error": f"Tesseract failed for every bounded OCR attempt: {detail}"}
            rejected = [item for item in all_results.values() if item["evidence"]["state"] == "rejected"]
            if rejected:
                worst = max(rejected, key=lambda item: (item["chars"], -item["confidence"]))
                evidence = worst["evidence"]
                message = (
                    "OCR text was suppressed because it did not meet evidence-quality requirements "
                    f"({evidence['reason']}; {evidence['chars']} chars at {evidence['confidence']}% confidence)."
                )
                return {
                    "error": message,
                    "diagnostic": {
                        **diagnostic("ocr_evidence_rejected", message, deadline, "evidence_quality", attempts_completed, len(plan)),
                        "evidence": evidence,
                        "terminal_small_crop_rejection": terminal_rejection,
                        "attempt_cap_seconds": attempt_cap_seconds,
                    },
                }
            # Empty or tiny non-substantive detections are valid observations:
            # do not invent text and do not report them as a pipeline error.
            result = {
                "text": "",
                "confidence": 0.0,
                "variant": "none",
                "chars": 0,
                "lines": 0,
                "score": 0.0,
                "image_size": f"{w_orig}x{h_orig}",
                "variants_tested": len(all_results),
                "all_variants": {},
                "quality": {
                    "schema": "omnius.ocr-evidence.v1",
                    "state": "low_information",
                    "accepted": False,
                    "reason": "no_accepted_readable_text",
                    "low_information_variants": len(all_results),
                },
                "diagnostic": diagnostic(
                    "ocr_low_information",
                    "OCR produced no accepted readable text; the result is low-information rather than evidence.",
                    deadline,
                    "evidence_quality",
                    attempts_completed,
                    len(plan),
                ),
                "strategy": {
                    "effective_area_px": effective_area,
                    "attempts_planned": len(plan),
                    "attempts_completed": attempts_completed,
                    "early_exit": False,
                    "terminal_small_crop_rejection": terminal_rejection,
                    "attempt_cap_seconds": attempt_cap_seconds,
                    "deadline_ms": deadline.deadline_ms,
                },
            }
            return result

        best = all_results[best_key]
        accepted_results = {
            key: value for key, value in all_results.items()
            if value["evidence"]["state"] == "accepted"
        }
        result = {
            "text": best["text"],
            "confidence": best["confidence"],
            "variant": best_key,
            "chars": best["chars"],
            "lines": best["lines"],
            "score": best["score"],
            "image_size": f"{w_orig}x{h_orig}",
            "variants_tested": len(all_results),
            # Never leak rejected raw OCR as alternate evidence to agents.
            "all_variants": accepted_results,
            "quality": {
                "schema": "omnius.ocr-evidence.v1",
                **best["evidence"],
                "rejected_variants": sum(
                    1 for item in all_results.values() if item["evidence"]["state"] == "rejected"
                ),
                "low_information_variants": sum(
                    1 for item in all_results.values() if item["evidence"]["state"] == "low_information"
                ),
            },
            "strategy": {
                "effective_area_px": effective_area,
                "attempts_planned": len(plan),
                "attempts_completed": attempts_completed,
                "early_exit": early_exit,
                "terminal_small_crop_rejection": terminal_rejection,
                "attempt_cap_seconds": attempt_cap_seconds,
                "deadline_ms": deadline.deadline_ms,
            },
        }

        if do_regions:
            regions = {}
            region_defs = {"header": (0, 35), "body": (30, 80), "footer": (75, 100)}
            for rname, (y_start, y_end) in region_defs.items():
                deadline.check(f"region {rname}")
                region_gray = extract_region(gray_2x, y_start, y_end)
                # Region requests use at most two high-yield attempts. The
                # main result already provides full-frame coverage.
                region_area = region_gray.shape[0] * region_gray.shape[1]
                region_plan = build_ocr_plan(region_area, single_psm)[:2]
                region_is_small = region_area <= SMALL_CROP_AREA_PX
                region_results, _errors, region_best_key, _early_exit, _terminal_rejection = run_variant_plan(
                    region_gray,
                    region_plan,
                    deadline,
                    language,
                    debug_dir,
                    f"region_{rname}",
                    attempt_cap_seconds=(
                        SMALL_CROP_TESSERACT_ATTEMPT_SECONDS
                        if region_is_small
                        else MAX_TESSERACT_ATTEMPT_SECONDS
                    ),
                    stop_on_terminal_rejection=region_is_small,
                )
                regions[rname] = region_results[region_best_key]["text"] if region_best_key else ""
            result["regions"] = regions

        if debug_dir:
            result["debug_dir"] = debug_dir
        if output_dir:
            base_name = Path(image_path).stem
            result["output_files"] = write_all_outputs(best["text"], base_name, output_dir)
        return result
    except OcrPipelineTimeout as error:
        return {
            "error": str(error),
            "diagnostic": diagnostic("ocr_timeout", str(error), deadline, "pipeline", attempts_completed, len(plan)),
        }
    except OcrPipelineCancelled as error:
        return {
            "error": "Advanced OCR cancelled",
            "diagnostic": diagnostic("ocr_cancelled", str(error), deadline, "pipeline", attempts_completed, len(plan)),
        }


def run_batch(images_dir, deadline, language="eng", do_regions=False, debug_dir=None,
              output_dir=None):
    """Process all images in a directory."""
    images_dir = os.path.abspath(images_dir)
    if not os.path.isdir(images_dir):
        return {"error": f"Not a directory: {images_dir}"}

    out_dir = output_dir or os.path.join(images_dir, "ocr_out")
    os.makedirs(out_dir, exist_ok=True)

    batch_results = {}
    image_files = sorted(
        f for f in os.listdir(images_dir)
        if Path(f).suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_files:
        return {"error": f"No image files found in {images_dir}"}

    for img_file in image_files:
        try:
            deadline.check("batch scheduling")
        except OcrPipelineTimeout as error:
            return {
                "error": str(error),
                "diagnostic": diagnostic("ocr_timeout", str(error), deadline, "batch", len(batch_results), len(image_files)),
            }
        img_path = os.path.join(images_dir, img_file)
        img_debug = os.path.join(debug_dir, Path(img_file).stem) if debug_dir else None
        result = run_pipeline(
            img_path,
            deadline,
            language=language,
            do_regions=do_regions,
            debug_dir=img_debug,
            output_dir=out_dir,
        )
        # Compact per-image result (omit all_variants for batch summary)
        batch_results[img_file] = {
            "text": result.get("text", ""),
            "confidence": result.get("confidence", 0),
            "variant": result.get("variant", ""),
            "chars": result.get("chars", 0),
            "lines": result.get("lines", 0),
            "output_files": result.get("output_files"),
            "error": result.get("error"),
        }
        if result.get("diagnostic", {}).get("code") in {"ocr_timeout", "ocr_cancelled"}:
            return {
                "error": result["error"],
                "diagnostic": result["diagnostic"],
                "batch": True,
                "images_processed": len(batch_results),
                "results": batch_results,
            }

    # Write summary
    summary_path = os.path.join(out_dir, "OCR_PROCESSING_SUMMARY.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# OCR Processing Summary Report\n\n")
        f.write(f"**Source:** `{images_dir}`\n\n")
        f.write("## Processed Documents\n\n")
        f.write("| Document | Lines | Chars | Confidence | Variant |\n")
        f.write("|----------|-------|-------|------------|----------|\n")
        for img, data in batch_results.items():
            if data.get("error"):
                f.write(f"| {img} | ERROR | - | - | {data['error']} |\n")
            else:
                f.write(
                    f"| {img} | {data['lines']} | {data['chars']} "
                    f"| {data['confidence']}% | {data['variant']} |\n"
                )

    return {
        "batch": True,
        "images_processed": len(batch_results),
        "output_dir": out_dir,
        "summary": summary_path,
        "results": batch_results,
    }


def main():
    global ACTIVE_DEADLINE
    parser = argparse.ArgumentParser(
        description="Advanced multi-variant OCR pipeline for omnius"
    )
    parser.add_argument(
        "image",
        help="Path to image file, or directory for --batch mode",
    )
    parser.add_argument("--language", "-l", default="eng",
                        help="OCR language (default: eng)")
    parser.add_argument("--regions", action="store_true",
                        help="Also OCR header/body/footer regions")
    parser.add_argument("--debug-dir",
                        help="Save preprocessed images to this directory")
    parser.add_argument("--psm", type=int, choices=[4, 6, 11],
                        help="Use single PSM mode instead of all 3")
    parser.add_argument("--region",
                        help="Crop region before OCR: x,y,w,h in pixels")
    parser.add_argument("--output", choices=["json", "text"], default="json",
                        help="Stdout output format (default: json)")
    parser.add_argument("--output-dir",
                        help="Write TXT + CSV + PDF outputs to this directory")
    parser.add_argument("--batch", action="store_true",
                        help="Process all images in a directory")
    parser.add_argument("--deadline-ms", type=int, default=MAX_PIPELINE_DEADLINE_MS,
                        help="Bound total OCR work; clamped to the managed maximum")

    args = parser.parse_args()
    deadline = OcrDeadline(args.deadline_ms)
    ACTIVE_DEADLINE = deadline
    signal.signal(signal.SIGTERM, cancellation_signal_handler)
    signal.signal(signal.SIGINT, cancellation_signal_handler)

    # Batch mode
    if args.batch or os.path.isdir(args.image):
        result = run_batch(
            args.image,
            deadline,
            language=args.language,
            do_regions=args.regions,
            debug_dir=args.debug_dir,
            output_dir=args.output_dir,
        )
        if args.output == "text":
            if "error" in result:
                print(f"ERROR: {result['error']}", file=sys.stderr)
                sys.exit(1)
            print(f"Processed {result['images_processed']} images → {result['output_dir']}")
        else:
            print(json.dumps(result, indent=2))
        sys.exit(1 if "error" in result else 0)

    # Single image mode
    if not os.path.isfile(args.image):
        print(json.dumps({"error": f"File not found: {args.image}"}))
        sys.exit(1)

    pixel_region = None
    if args.region:
        try:
            pixel_region = tuple(int(x) for x in args.region.split(","))
            if len(pixel_region) != 4:
                raise ValueError
        except ValueError:
            print(json.dumps({"error": "Region must be x,y,w,h (4 integers)"}))
            sys.exit(1)

    result = run_pipeline(
        args.image,
        deadline,
        language=args.language,
        do_regions=args.regions,
        debug_dir=args.debug_dir,
        single_psm=args.psm,
        pixel_region=pixel_region,
        output_dir=args.output_dir,
    )

    if args.output == "text":
        if "error" in result:
            print(f"ERROR: {result['error']}", file=sys.stderr)
            sys.exit(1)
        print(result["text"])
    else:
        print(json.dumps(result, indent=2))
        if "error" in result:
            sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except OcrPipelineTimeout as error:
        deadline = ACTIVE_DEADLINE or OcrDeadline(MAX_PIPELINE_DEADLINE_MS)
        print(json.dumps({
            "error": str(error),
            "diagnostic": diagnostic("ocr_timeout", str(error), deadline, "entrypoint"),
        }))
        sys.exit(124)
    except OcrPipelineCancelled as error:
        deadline = ACTIVE_DEADLINE or OcrDeadline(MAX_PIPELINE_DEADLINE_MS)
        print(json.dumps({
            "error": "Advanced OCR cancelled",
            "diagnostic": diagnostic("ocr_cancelled", str(error), deadline, "entrypoint"),
        }))
        sys.exit(130)
