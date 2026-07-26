# VisionInspect AI: Intelligent Infrastructure Inspection Platform

VisionInspect AI is an enterprise-grade civil engineering pipeline designed for structural health monitoring, spatial analysis, and automated maintenance dispatching. It ingests inspection photographs, segments concrete defects, computes sub-millimeter crack measurements, and manages maintenance tickets with a Human-in-the-Loop review system.

---

## 🚀 Core Platform Capabilities

### 1. YOLOv8 Instance Segmentation (`/vision`)
- Uses domain-specific YOLOv8 segmentation weights (`models/cracks_spalling_v1.pt`) to detect cracks and concrete spalling.
- Extracts polygon masks using PyTorch CPU arrays (`masks.data.cpu().numpy()`) and converts them to standard OpenCV binary masks.

### 2. OpenCV Distance Transform Math (`/calibration`)
- Feeds binary segmentations into the OpenCV Distance Transform (`cv2.distanceTransform`) to locate the center of mass of the defect.
- **Diameter Math Fix**: Multiplies the output radius of the Distance Transform by `2.0` to calculate the exact physical crack mouth width (diameter) in pixels.

### 3. EXIF Photogrammetry Scale Calibration (`/calibration`)
- Parses EXIF metadata (`FocalLength`, `ExifImageWidth`, `SubjectDistance`) from inspection photographs.
- Applies standard camera photogrammetry formula assuming a full-frame 36mm sensor width to compute the Ground Sampling Distance (GSD):
  $$\text{scale\_mm\_per\_px} = \frac{\text{SubjectDistance} \times 36.0}{\text{FocalLength} \times \text{ExifImageWidth}}$$
- **Defensive Fallback**: Defaults to a hardcoded macro scale of `0.1` mm/px if EXIF metadata is missing, incomplete, or returns `SubjectDistance = 0`.

### 4. 75% Human-in-the-Loop (HITL) Safety Gate (`/pipeline`)
- Implements strict thresholding on model prediction confidence:
  - **Confidence $\ge$ 75%**: Automatically routes severity and priority, and enables automated ticket PDF compilation.
  - **Confidence $<$ 75%**: Halts the automated pipeline, overrides severity to `"Requires Manual Review"`, and flags the defect for human technician approval.

---

## 📁 Modular Directory Structure

```
infrastructure-inspector/
├── vision/                   # Model loading, YOLOv8 inference, mask extraction
├── calibration/              # EXIF parsing, camera lookup, and GSD scale math
├── pipeline/                 # Orchestrator connecting vision, calibration, and QC
├── database/                 # Future PostGIS storage and temporal tracking
├── ui/                       # Future dashboard and manual review pages
├── AGENTS.md                 # Workspace master architectural guidelines
├── app.py                    # Streamlit Dashboard UI & review interface
├── run_evals.py              # Automated test runner and quality control gate
├── evals.json                # Security payload and routing test cases
└── .agents/                  # Customizations and skills registry
```

---

## 🛠️ Execution & Usage

### 1. Run the Vision Pipeline
To process an image, extract metrics, and generate an annotated output:
```bash
python3 .agents/skills/inspector/scripts/analyze_image.py \
  --image-path "sample_test_crack.jpg" \
  --output-dir "runs/"
```

### 2. Run Automated Regression Evals
Verify computer vision precision (within `0.1px` tolerance) and security assertions:
```bash
python3 run_evals.py
```

### 3. Start Streamlit Dashboard UI
```bash
streamlit run app.py
```

---

## 🔮 Phase 2 Roadmap
- **Multi-Class Training**: Extend segmentation models to detect concrete spalling and exposed rebar.
- **Graph Morphological Thinning**: Implement skeletonization to measure crack lengths and identify branching/bifurcation nodes.
- **Temporal Growth tracking**: Link new inspections with historical data using PostGIS spatial joins.
