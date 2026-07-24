import os
import sys
import cv2
import json
import numpy as np

from vision.segmentation import run_dl_instance_segmentation
from calibration.scale import get_exif_gps, get_calibration_scale, measure_crack_width

def calculate_engineering_metrics(image_width_px, measured_pixel_width, spalling_pixel_area, confidence_pct, gsd, cal_status, reference_marker_width_mm=None):
    """
    Calculates physical sub-millimeter crack width, uncertainty, spalling area, and severity classification.
    Safely handles intact zero-width/area cases.
    Enforces the 75% HITL Safety Gate.
    """
    calibration_status = cal_status
    
    # 75% HITL Safety Gate Check
    is_below_confidence_gate = (confidence_pct < 75.0)
    
    crack_width_mm = round(measured_pixel_width * gsd, 2)
    spalling_area_mm2 = round(spalling_pixel_area * (gsd ** 2), 2)
    uncertainty_mm = round(max(0.02, crack_width_mm * 0.10), 2)
    
    # Standard classification defaults
    severity = "None"
    priority = "None"
    maintenance_action = "No structural defect detected. Surface is intact."
    bgr_color = (94, 197, 34)  # Green BGR
    
    # 1. Crack classification
    if crack_width_mm > 0.0:
        if crack_width_mm < 0.3:
            severity = "Minor"
            priority = "Low"
            maintenance_action = "Monitor during routine maintenance cycle"
            bgr_color = (94, 197, 34)  # Green BGR
        elif crack_width_mm <= 1.0:
            severity = "Moderate"
            priority = "Medium"
            maintenance_action = "Apply flexible epoxy/polyurethane sealant"
            bgr_color = (8, 179, 234)  # Yellow BGR
        else:
            severity = "Severe"
            priority = "High"
            maintenance_action = "Immediate structural assessment & load restriction"
            bgr_color = (68, 68, 239)  # Red BGR
            
    # 2. Spalling classification (can escalate severity/priority)
    if spalling_area_mm2 > 0.0:
        sp_severity = "Moderate"
        sp_priority = "Medium"
        sp_action = "Patch spalled area with structural repair mortar"
        sp_color = (8, 179, 234)  # Yellow BGR
        
        if spalling_area_mm2 >= 100.0:
            sp_severity = "Severe"
            sp_priority = "High"
            sp_action = "Immediate repair of concrete spalling & reinforcement inspect"
            sp_color = (68, 68, 239)  # Red BGR
            
        severity_order = {"None": 0, "Minor": 1, "Requires Manual Review": 2, "Moderate": 3, "Severe": 4}
        current_rank = severity_order.get(severity, 0)
        sp_rank = severity_order.get(sp_severity, 0)
        
        if sp_rank > current_rank:
            severity = sp_severity
            priority = sp_priority
            maintenance_action = sp_action
            bgr_color = sp_color
        elif sp_rank == current_rank and crack_width_mm > 0.0:
            maintenance_action = f"{maintenance_action} & patch spalled area"
        
    # Enforce HITL Safety Gate Override for low confidence
    if is_below_confidence_gate:
        severity = "Requires Manual Review"
        priority = "None"
        maintenance_action = "Halted due to low model confidence. Route to manual review."
        
    return {
        "crack_width_mm": crack_width_mm,
        "measured_pixel_width": measured_pixel_width,
        "spalling_area_mm2": spalling_area_mm2,
        "spalling_pixel_area": spalling_pixel_area,
        "uncertainty_mm": uncertainty_mm,
        "confidence_pct": confidence_pct,
        "severity": severity,
        "priority": priority,
        "maintenance_action": maintenance_action,
        "bgr_color": bgr_color,
        "calibration_status": calibration_status
    }

def generate_annotated_image(image_path, output_annotated_path, valid_contours, metrics):
    """
    Annotates the input image by drawing crisp 2px solid contour lines directly over AI-segmented defect mask
    and rendering a single top-left HUD summary banner block.
    """
    try:
        img_cv = cv2.imread(image_path)
        if img_cv is None:
            return False
            
        bgr_color = metrics["bgr_color"]
        
        # Draw 2-pixel solid lines over defect contours
        if valid_contours:
            cv2.drawContours(img_cv, valid_contours, -1, bgr_color, 2)
            
        # Render single top-left HUD summary banner block
        overlay = img_cv.copy()
        hud_w, hud_h = 580, 40
        cv2.rectangle(overlay, (10, 10), (10 + hud_w, 10 + hud_h), (20, 20, 20), -1)
        alpha = 0.75
        cv2.addWeighted(overlay, alpha, img_cv, 1 - alpha, 0, img_cv)
        
        # Border around HUD
        cv2.rectangle(img_cv, (10, 10), (10 + hud_w, 10 + hud_h), (180, 180, 180), 1)
        
        # Build dynamic HUD Text displaying crack and spalling measurements
        hud_parts = []
        if metrics.get("crack_width_mm", 0.0) > 0.0:
            hud_parts.append(f"Crack: {metrics['crack_width_mm']}mm")
        if metrics.get("spalling_area_mm2", 0.0) > 0.0:
            hud_parts.append(f"Spalling: {metrics['spalling_area_mm2']}mm2")
        if not hud_parts:
            hud_parts.append("Intact")
            
        hud_text = f"{' | '.join(hud_parts)} | {metrics['calibration_status']} | Severity: {metrics['severity']}"
        
        cv2.putText(
            img_cv,
            hud_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        
        os.makedirs(os.path.dirname(os.path.abspath(output_annotated_path)), exist_ok=True)
        cv2.imwrite(output_annotated_path, img_cv)
        return True
    except Exception as e:
        print(f"[WARNING] Failed to generate annotated image: {e}", file=sys.stderr)
        return False

def run_pipeline(image_path, output_dir=None, reference_marker_width_mm=None, conf=0.25, gsd=None):
    """
    Modular execution orchestrator connecting ingestion, segmentation, scale calibration,
    and output annotation. Returns output JSON payload.
    """
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        raise ValueError(f"Could not decode image at '{image_path}'")
        
    image_height_px, image_width_px = image_cv.shape[:2]
    
    # 1. GPS EXIF Location
    lon, lat = get_exif_gps(image_path)
    
    # 2. YOLO Instance Segmentation (Multi-class with custom confidence)
    masks_dict, confidence_pct = run_dl_instance_segmentation(image_path, image_cv, conf=conf)
    
    # 3. Calibration GSD Scale Calculation (EXIF-based photogrammetry or User-Provided GSD)
    if gsd is not None:
        calc_gsd = float(gsd)
        cal_status = "Calibrated"
    else:
        calc_gsd, cal_status = get_calibration_scale(image_path, image_width_px, reference_marker_width_mm)
    
    # 4. OpenCV Distance Transform Geometry Measurement for Cracks
    measured_pixel_width, crack_contours = measure_crack_width(masks_dict['crack'])
    
    # 4b. OpenCV Measurement for Spalling Area
    spalling_mask = masks_dict['spalling']
    spalling_pixel_area = float(cv2.countNonZero(spalling_mask))
    spalling_contours, _ = cv2.findContours(spalling_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 5. Engineering Metric Calculations (Zero-width intact safe, 75% Safety Gate check)
    metrics = calculate_engineering_metrics(
        image_width_px, measured_pixel_width, spalling_pixel_area, confidence_pct, calc_gsd, cal_status, reference_marker_width_mm
    )
    
    # 6. Save Clean Annotated Image with 2px Contour Overlays & HUD Banner
    out_dir = output_dir if output_dir else os.path.dirname(os.path.abspath(image_path))
    annotated_filename = "annotated_" + os.path.basename(image_path)
    if not annotated_filename.endswith(".jpg") and not annotated_filename.endswith(".png"):
        annotated_filename += ".jpg"
    annotated_path = os.path.join(out_dir, annotated_filename)
    
    # Combine contours of crack and spalling for drawing
    valid_contours = list(crack_contours) + list(spalling_contours)
    generate_annotated_image(image_path, annotated_path, valid_contours, metrics)
    
    # 7. Status Determination
    if metrics["severity"] in ["Severe", "Moderate"]:
        status = "Requires PDF ticket"
    elif metrics["severity"] == "Requires Manual Review":
        status = "Requires Manual Review"
    else:
        status = "Log only"
        
    # Map classes to their user-facing labels dynamically
    detected_classes = []
    if measured_pixel_width > 0:
        detected_classes.append("Crack")
    if spalling_pixel_area > 0:
        detected_classes.append("Spalling")
        
    if detected_classes:
        crack_type = " & ".join(detected_classes)
    else:
        crack_type = "None (Intact)"
        
    # 8. Strict Output Schema JSON Payload
    json_data = {
        "image_name": os.path.basename(image_path),
        "annotated_path": annotated_path,
        "crack_type": crack_type,
        "crack_width_mm": metrics["crack_width_mm"],
        "max_width_mm": metrics["crack_width_mm"],
        "measured_pixel_width": metrics["measured_pixel_width"],
        "spalling_area_mm2": metrics["spalling_area_mm2"],
        "spalling_pixel_area": metrics["spalling_pixel_area"],
        "uncertainty_mm": metrics["uncertainty_mm"],
        "confidence_pct": metrics["confidence_pct"],
        "severity": metrics["severity"],
        "priority": metrics["priority"],
        "maintenance_action": metrics["maintenance_action"],
        "calibration_status": metrics["calibration_status"],
        "status": status,
        "lon": lon,
        "lat": lat
    }
    
    payload_summary = (
        f"VISION PAYLOAD: Image {os.path.basename(image_path)} analyzed. "
        f"Crack width: {metrics['crack_width_mm']}mm. "
        f"Status: {status}. "
        f"Coordinates: {lon}, {lat}."
    )
    
    return json_data, payload_summary
