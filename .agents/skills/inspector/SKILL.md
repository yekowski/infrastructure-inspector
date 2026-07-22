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
  1. Initialize the YOLOv8 instance segmentation model loading the custom weights at `models/crack_seg_best.pt`.
  2. Pass the input image through the model to extract the segmentation polygon masks using `result.masks.data.cpu().numpy()`.
  3. Convert the detected mask to standard OpenCV `uint8` format by multiplying by `255`.
  4. Apply `cv2.distanceTransform(mask, cv2.DIST_L2, 5)` to the converted mask to get the distance to the nearest background pixel.
  5. Compute the diameter/thickness of the crack using the corrected math: multiply `np.max(dist_transform)` by `2.0` to calculate the maximum pixel width.
  6. Apply morphological open/close cleaning and contours extraction.

### B. Skill: `calibrate_gsd_scale`
* **Trigger**: When determining the physical scale factor (Ground Sampling Distance in mm/px).
* **Procedure**:
  1. Open the image file and attempt to extract EXIF tags: `FocalLength` (tag 37386), `ExifImageWidth` (tag 40962 / `img.width`), and `SubjectDistance` (tag 37382).
  2. Defensive Casting: Wrap EXIF parsing in try/except blocks and cast all extracted parameters to floats.
  3. Photogrammetry GSD Formula: If valid parameters are present, calculate GSD using:
     $$\text{scale\_mm\_per\_px} = \frac{\text{SubjectDistance} \times 36.0}{\text{FocalLength} \times \text{ExifImageWidth}}$$
     *(Assuming full-frame 36mm sensor width. Convert SubjectDistance from meters to millimeters by multiplying by 1000 if necessary).*
  4. Fallback Rule: If EXIF is missing, incomplete, or returns `SubjectDistance = 0`, fallback defensively to a default macro scale of `0.1` mm/px.
  5. Save status label as `"EXIF Calibrated"` or `"Uncalibrated (Default GSD)"`.

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
