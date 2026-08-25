---
name: inspector
description: Perform PostGIS logging, PDF ticket generation, OpenCV image and video analysis, spatial canvas registration, and map visualization for geospatial infrastructure inspections using secure CLI helper scripts.
---

# Geospatial Inspector Skill Manual

This manual documents the executable workflows and capabilities of the **VisionInspect AI** platform.

To eliminate prompt-injection vulnerabilities and prevent hallucinated measurements or inline code execution, agents must NOT evaluate raw strings, invent measurement numbers, or write custom drawing scripts. Instead, agents must extract parameters and invoke the pre-compiled CLI helper tools provided in this workspace. All configuration parameters must be injected via environment variables, and all runtime errors and metrics must be emitted using structured JSON logging.

---

## 1. Core Architectural & Logging Standards
* **Configuration**: The application configuration is injected dynamically via environment variables (e.g., `PERCENTILE_CUTOFF`, `WIDTH_MULTIPLIER`, `DEFAULT_GSD`) parsed by a centralized configuration utility (`config.py`).
* **Structured Observability**: Standard print logging (`print()`) is prohibited for debugging or diagnostics. All workflow executions, warnings, and errors must be output as structured JSON logs via Python's built-in `logging` module.

---

## 2. Core Platform Workflows

### A. Skill: `analyze_concrete_defect`
* **Trigger**: When analyzing an uploaded concrete photograph.
* **Procedure**:
  1. Initialize the YOLOv8 instance segmentation model loading the custom weights at `models/cracks_spalling_v1.pt`.
  2. Pass the input image through the model with `retina_masks=True` to extract high-resolution segmentation masks using `result.masks.data.cpu().numpy()`.
  3. Convert the detected mask to standard OpenCV `uint8` format by multiplying by `255`.
  4. Apply `cv2.distanceTransform(mask, cv2.DIST_L2, 5)` to the converted mask to compute the Euclidean distance to the nearest background pixel. Protect against mathematical anomalies by replacing NaNs/infinities using `np.nan_to_num()`.
  5. Apply OpenCV's native thinning algorithm `cv2.ximgproc.thinning(mask)` to extract a 1-pixel medial axis, eliminating third-party dependency bloat. Extract the distance transform values along this thinned skeleton line. Sort the values, drop the top 5% as outliers (configured by `PERCENTILE_CUTOFF`), and calculate the final pixel width using `WIDTH_MULTIPLIER` applied to the target percentile.
  6. Apply morphological open/close cleaning and contours extraction.

### B. Skill: `calibrate_gsd_scale`
* **Trigger**: When determining the physical scale factor (Ground Sampling Distance in mm/px).
* **Procedure** (strict priority hierarchy):
  1. **Priority 1 — Reference Marker (absolute ground truth):** If `--reference-marker-width-mm` is provided, the GSD is calculated directly from the user-drawn calibration line pixel distance divided into the known physical width. EXIF extraction is bypassed entirely.
  2. **Priority 2 — True EXIF Hardware Dimensions:** If no reference marker is provided, attempt to parse `FocalPlaneXResolution` (tag 41486), `FocalPlaneResolutionUnit` (tag 41488), `FocalLength` (tag 37386), and `ExifImageWidth` (tag 40962). Derive the true sensor width in mm from `ExifImageWidth / FocalPlaneXResolution`, converting units based on `FocalPlaneResolutionUnit` (2=inches, 3=cm, 4=mm). Calculate GSD as `sensor_width_mm / (FocalLength × ExifImageWidth)`. Wrap all EXIF parsing in defensive `try/except` blocks and cast to floats.
  3. **Priority 3 — Uncalibrated Fallback:** If EXIF hardware data is missing, incomplete, or parsing fails, fall back defensively to the default GSD (configured via `DEFAULT_GSD`, defaulting to `0.1` mm/px).
  4. Save status label as `"Calibrated"`, `"EXIF Calibrated"`, or `"Uncalibrated (Default GSD)"`.

### C. Skill: `enforce_hitl_safety_gate`
* **Trigger**: When finalizing inspection metrics.
* **Procedure**:
  1. Retrieve `confidence_pct` from the Vision Skill output payload.
  2. Safety Gate Check:
     * **If `confidence_pct` >= 75%**: Proceed with standard severity routing (Minor, Moderate, Severe) and allow PDF/PostGIS automated dispatch.
     * **If `confidence_pct` < 75%**: IMMEDIATELY HALT automated ticket generation. Override the severity status to `"Requires Manual Review"` (mapped internally as `"REQUIRES_MANUAL_REVIEW"` or `"Requires Manual Review"`), block automatic PDF compilation, and queue the record for manual inspector validation/override.

---

## 3. Video Processing & Spatial Canvas Mapping

### D. Skill: `sample_video_frames`
* **Trigger**: When a video file (`.mp4`, `.mov`, `.avi`) is uploaded for structural inspection.
* **Procedure**:
  1. Open the video using `cv2.VideoCapture(video_path)`.
  2. Read the native framerate via `cap.get(cv2.CAP_PROP_FPS)`.
  3. Calculate the frame interval: `interval = int(native_fps / target_fps)` where `target_fps` is configurable (default: `3` FPS).
  4. Yield frames one-at-a-time using a generator pattern. Do NOT accumulate frames in a list. Release each frame after processing.
  5. For each sampled frame, emit a structured JSON log entry containing `frame_index`, `timestamp_sec`, and processing status.
  6. Release the `VideoCapture` handle on completion or exception (`cap.release()` in a `finally` block).

### E. Skill: `register_frame_to_canvas`
* **Trigger**: For each sampled video frame, after YOLO segmentation produces a binary defect mask.
* **Procedure (Dual-Path Processing)**:
  1. **Semantic Path**: Pass the frame through the existing `analyze_concrete_defect` workflow (YOLOv8 → binary mask extraction → Distance Transform). Retain the binary mask output.
  2. **Spatial Path**:
     a. Convert the frame to grayscale (`cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)`).
     b. Detect keypoints and compute descriptors using SIFT (`cv2.SIFT_create()`) as the primary detector. SIFT is mandated over ORB because concrete surfaces are often low-texture and uniform — SIFT's gradient-histogram descriptors are significantly more robust for these surfaces than ORB's binary BRIEF descriptors. Since frames are sampled at low FPS (3–5), accuracy is prioritized over real-time performance.
     c. Match descriptors against the previous frame's descriptors using `cv2.BFMatcher(cv2.NORM_L2)`. Apply Lowe's ratio test (threshold: `0.75`) to filter ambiguous matches.
  3. **Registration & Validation**:
     a. Extract matched keypoint coordinates from both frames.
     b. Compute the homography matrix: `H, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)`.
     c. **Validate Matrix Properties**:
        - Require `np.sum(inliers) >= MIN_HOMOGRAPHY_INLIERS` (default: `10`).
        - Require `0.1 < abs(np.linalg.det(H)) < MAX_HOMOGRAPHY_DET_VARIANCE` (default: `10.0`).
        - **Reject Perspective Skew**: Require perspective terms $|H_{2,0}| \le \text{MAX\_PERSPECTIVE\_SKEW}$ and $|H_{2,1}| \le \text{MAX\_PERSPECTIVE\_SKEW}$ (default: `0.001`). If any check fails, log a warning and skip the frame.
  4. **Dynamic Canvas Expansion & Warp**:
     a. Multiply $H_{\text{frame}}$ with $H_{\text{cumulative}}$ to get the candidate transformation.
     b. Transform the 4 boundary corners of the incoming frame to find the candidate bounding box on the canvas.
     c. If the candidate bounding box extends beyond current canvas dimensions, dynamically expand the canvas using `cv2.copyMakeBorder(canvas, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)` and update $H_{\text{cumulative}}$ with a 2D translation matrix $T(\Delta x, \Delta y)$.
     d. Warp the binary defect mask: `warped_mask = cv2.warpPerspective(mask, H_cumulative, (canvas_w, canvas_h))`.
  5. **Canvas Fusion**:
     a. Fuse the warped mask into the persistent Global Canvas using `cv2.bitwise_or(canvas, warped_mask)`.
     b. The canvas is append-only. Do NOT clear or reinitialize it between frames.

### F. Skill: `measure_fused_canvas`
* **Trigger**: After all sampled frames have been registered and fused onto the Global Canvas.
* **Procedure**:
  1. Run `cv2.connectedComponentsWithStats(canvas)` on the fused Global Canvas to identify individual defect regions.
  2. For each connected component, extract: `area_px`, `bounding_box`, `centroid_x`, `centroid_y`.
  3. Apply the existing `calibrate_gsd_scale` workflow to convert pixel measurements to physical units (mm/mm²).
  4. **Crack Physical Length Calculation**:
     a. Skeletonize the component binary mask using `cv2.ximgproc.thinning(component_mask)`.
     b. Build an adjacency graph of the 1-pixel skeleton. Identify node degrees: endpoints (degree $= 1$) and bifurcation branch nodes (degree $> 2$).
     c. **Spur-Pruning**: Iteratively prune dead-end branch paths connected to an endpoint whose path length is less than `MIN_SPUR_LENGTH_PX` (default: `10` pixels), preserving the main topological spine.
     d. Traverse the pruned skeleton graph. For each 8-connected neighbor step: assign a weight of $1.0\text{px}$ for orthogonal steps ($\Delta x + \Delta y = 1$) and $\sqrt{2} \approx 1.414\text{px}$ for diagonal steps ($\Delta x = 1, \Delta y = 1$).
     e. Multiply total pruned skeleton pixel length by GSD to compute `crack_length_mm` and `crack_length_cm`.
  5. **Crack Orientation Classification**:
     a. Extract points along the pruned skeleton or outer contour.
     b. Compute Principal Component Analysis (PCA) or fit a minimum area ellipse (`cv2.fitEllipse()`) / rectangle (`cv2.minAreaRect()`) to find the major axis angle $\theta \in [-90^\circ, 90^\circ]$ relative to the horizontal axis.
     c. Classify orientation string tag:
        - `"Horizontal"` ($|\theta| \le 15^\circ$)
        - `"Vertical"` ($75^\circ \le |\theta| \le 90^\circ$)
        - `"Diagonal"` ($15^\circ < |\theta| < 75^\circ$)
  6. Apply the existing `enforce_hitl_safety_gate` to the aggregated confidence (mean confidence across contributing frames).
  7. Output the fused defect measurements using the **Strict Output Schema**, augmented with `crack_length_mm`, `crack_length_cm`, `orientation`, `source: "video"`, and `frame_count`.

---

## 4. Command Line Interface (CLI) Execution Signatures

### A. Computer Vision Image Analysis (`analyze_image.py`)
Run the vision detection and measurement pipeline:
```bash
python3 .agents/skills/inspector/scripts/analyze_image.py \
  --image-path "<path_to_image>" \
  [--output-dir "<output_directory>"] \
  [--reference-marker-width-mm <float>]
```

### B. Video Inspection Pipeline (`video/process_video.py`)
Run the dual-path spatial canvas video inspection pipeline:
```bash
python3 video/process_video.py \
  --video-path "<path_to_video>" \
  [--target-fps <int>] \
  [--conf <float>] \
  [--gsd <float>]
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
  "crack_length_mm": 145.2,
  "crack_length_cm": 14.52,
  "orientation": "Diagonal",
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

## 5. Deployment & Containerization Standards
To ensure robust security, minimal image size, and environment parity, all containerized deployments of the VisionInspect AI pipeline must adhere to the following standards:
* **Base Image**: Utilize a standard, slim official Python base image (e.g., `python:3.11-slim`). Avoid heavy default images or bare alpine versions, as C-library compatibility (glibc) is critical for native OpenCV performance.
* **Security (Non-Root)**: Containers must never execute as the `root` superuser. The Dockerfile must explicitly create a dedicated system group and user (e.g., `appuser`) with limited system privileges, and use the `USER` instruction to run the container runtime process.
* **Context Exclusion**: An explicit `.dockerignore` file must exclude version control files (`.git`), runtime caches (`__pycache__`), environment configurations (`.env`), database connection strings, local test images, and unnecessary project markdown documentation.
* **Execution Entrypoint**: The container entrypoint must expose the primary CLI wrapper `analyze_image.py` as its default entrypoint (`ENTRYPOINT ["python3", ".agents/skills/inspector/scripts/analyze_image.py"]`), enabling execution arguments to be passed directly at runtime.

---

## 6. Regression Testing & E2E Validation
Before any changes to the vision pipeline are finalized, you must verify correctness using the automated E2E gate script:
```bash
python3 run_evals.py
```
- **Tolerance**: Asserts synthetic test measurements are within `0.1px` of expected dimensions.
- **Safety**: Validates security routing payloads for SQL injection and prompt injection blocks.
