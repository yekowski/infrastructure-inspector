---
name: inspector
description: Perform PostGIS logging, PDF ticket generation, OpenCV image analysis, and map visualization for geospatial infrastructure inspections using secure CLI helper scripts.
---

# Geospatial Inspector Skill Manual

This manual documents the executable workflows and capabilities of the **VisionInspect AI** platform.

To eliminate prompt-injection vulnerabilities and prevent hallucinated measurements or inline code execution, agents must NOT evaluate raw strings, invent measurement numbers, or write custom drawing scripts. Instead, agents must extract parameters and invoke the pre-compiled CLI helper tools provided in this workspace.

---

## 1. Core Platform Workflows

### A. Skill: `analyze_concrete_defect`
* **Trigger**: When analyzing an uploaded concrete photograph.
* **Procedure**:
  1. Initialize the YOLOv8 instance segmentation model loading the custom weights at `models/cracks_spalling_v1.pt`.
  2. Pass the input image through the model with `retina_masks=True` to extract high-resolution segmentation masks using `result.masks.data.cpu().numpy()`.
  3. Convert the detected mask to standard OpenCV `uint8` format by multiplying by `255`.
  4. Apply `cv2.distanceTransform(mask, cv2.DIST_L2, 5)` to the converted mask to compute the Euclidean distance to the nearest background pixel.
  5. Apply `skimage.morphology.skeletonize()` to the binary mask to extract a 1-pixel medial axis. Extract the distance transform values strictly along this skeleton line. Sort the extracted values, drop the top 5% as outliers, and take the 95th percentile value. Multiply this 95th-percentile radius by `2.0` to calculate the final crack pixel width.
  6. Apply morphological open/close cleaning and contours extraction.

### B. Skill: `calibrate_gsd_scale`
* **Trigger**: When determining the physical scale factor (Ground Sampling Distance in mm/px).
* **Procedure** (strict priority hierarchy):
  1. **Priority 1 — Reference Marker (absolute ground truth):** If `--reference-marker-width-mm` is provided, the GSD is calculated directly from the user-drawn calibration line pixel distance divided into the known physical width. EXIF extraction is bypassed entirely.
  2. **Priority 2 — True EXIF Hardware Dimensions:** If no reference marker is provided, attempt to parse `FocalPlaneXResolution` (tag 41486), `FocalPlaneResolutionUnit` (tag 41488), `FocalLength` (tag 37386), and `ExifImageWidth` (tag 40962). Derive the true sensor width in mm from `ExifImageWidth / FocalPlaneXResolution`, converting units based on `FocalPlaneResolutionUnit` (2=inches, 3=cm, 4=mm). Calculate GSD as `sensor_width_mm / (FocalLength × ExifImageWidth)`. Wrap all EXIF parsing in defensive `try/except` blocks and cast to floats.
  3. **Priority 3 — Uncalibrated Fallback:** If EXIF hardware data is missing, incomplete, or parsing fails, fall back defensively to a default macro scale of `0.1` mm/px.
  4. Save status label as `"Calibrated"`, `"EXIF Calibrated"`, or `"Uncalibrated (Default GSD)"`.

### C. Skill: `enforce_hitl_safety_gate`
* **Trigger**: When finalizing inspection metrics.
* **Procedure**:
  1. Retrieve `confidence_pct` from the Vision Skill output payload.
  2. Safety Gate Check:
     * **If `confidence_pct` >= 75%**: Proceed with standard severity routing (Minor, Moderate, Severe) and allow PDF/PostGIS automated dispatch.
     * **If `confidence_pct` < 75%**: IMMEDIATELY HALT automated ticket generation. Override the severity status to `"Requires Manual Review"` (mapped internally as `"REQUIRES_MANUAL_REVIEW"` or `"Requires Manual Review"`), block automatic PDF compilation, and queue the record for manual inspector validation/override.

---

## 2. Command Line Interface (CLI) Execution Signatures

### A. Computer Vision Image Analysis (`analyze_image.py`)
Run the vision detection and measurement pipeline:
```bash
python3 .agents/skills/inspector/scripts/analyze_image.py \
  --image-path "<path_to_image>" \
  [--output-dir "<output_directory>"] \
  [--reference-marker-width-mm <float>]
```

**Expected JSON Output Schema**:
```json
{
  "image_name": "sample.jpg",
  "annotated_path": "/path/to/annotated_sample.jpg",
  "crack_type": "Radial Floor Crack",
  "crack_width_mm": 0.45,
  "max_width_mm": 0.45,
  "measured_pixel_width": 4.5,
  "uncertainty_mm": 0.05,
  "confidence_pct": 92.4,
  "severity": "Moderate",
  "priority": "Medium",
  "maintenance_action": "Apply flexible epoxy/polyurethane sealant",
  "calibration_status": "EXIF Calibrated",
  "status": "Requires PDF ticket",
  "lon": -122.4194,
  "lat": 37.7749
}
```

### B. PostGIS Database Logging (`log_inspection.py`)
Record inspection metrics into database tables:
```bash
python3 .agents/skills/inspector/scripts/log_inspection.py \
  --ticket-id "<ticket_id>" \
  --inspector-name "<inspector_name>" \
  --status "<status>" \
  --lon <lon> \
  --lat <lat> \
  --notes "<notes>"
```

### C. PDF Ticket Generation (`generate_ticket.py`)
Compile official ReportLab PDF inspection tickets:
```bash
python3 .agents/skills/inspector/scripts/generate_ticket.py \
  --output-path "<output_path>" \
  --ticket-id "<ticket_id>" \
  --inspector-name "<inspector_name>" \
  --status "<status>" \
  --lon <lon> \
  --lat <lat> \
  --notes "<notes>" \
  [--crack-type "<crack_type>"] \
  [--crack-width "<crack_width>"] \
  [--uncertainty "<uncertainty>"] \
  [--confidence "<confidence>"] \
  [--priority "<priority>"] \
  [--maintenance-action "<maintenance_action>"]
```

### D. Map Visualization (`generate_map.py`)
Update the interactive HTML Folium mapping visualization:
```bash
python3 .agents/skills/inspector/scripts/generate_map.py
```

---

## 3. Regression Testing & E2E Validation
Before any changes to the vision pipeline are finalized, you must verify correctness using the automated E2E gate script:
```bash
python3 run_evals.py
```
- **Tolerance**: Asserts synthetic test measurements are within `0.1px` of expected dimensions.
- **Safety**: Validates security routing payloads for SQL injection and prompt injection blocks.
