import os
import sys
import cv2
import numpy as np
from PIL import Image, ExifTags
from config import PERCENTILE_CUTOFF, WIDTH_MULTIPLIER, DEFAULT_GSD, MIN_SPUR_LENGTH_PX, get_logger

logger = get_logger(__name__)

def get_exif_gps(image_path):
    """
    Extracts GPS coordinates (lon, lat) from image EXIF metadata.
    Falls back to (-122.4194, 37.7749) if EXIF info is missing or invalid.
    """
    default_coords = (-122.4194, 37.7749)
    try:
        img = Image.open(image_path)
        exif_data = img.getexif()
        if not exif_data:
            return default_coords
            
        gps_info = exif_data.get_ifd(34853)
        if not gps_info:
            return default_coords
            
        def to_degrees(value):
            d = float(value[0])
            m = float(value[1])
            s = float(value[2])
            return d + (m / 60.0) + (s / 3600.0)
            
        lat_ref = gps_info.get(1)
        lat_val = gps_info.get(2)
        lon_ref = gps_info.get(3)
        lon_val = gps_info.get(4)
        
        if lat_val and lon_val:
            lat = to_degrees(lat_val)
            if lat_ref == 'S':
                lat = -lat
            lon = to_degrees(lon_val)
            if lon_ref == 'W':
                lon = -lon
            return round(lon, 4), round(lat, 4)
    except Exception as e:
        logger.warning(
            "EXIF GPS extraction failed, using default coordinates",
            exc_info=True,
            extra={"default_coords": default_coords, "image_path": image_path}
        )
        
    return default_coords

def extract_exif_metadata(image_path):
    """
    Attempts to extract FocalLength, FocalPlaneXResolution, FocalPlaneResolutionUnit,
    and ExifImageWidth from image EXIF data to derive true sensor dimensions.
    """
    focal_length = None
    exif_width = None
    focal_plane_x_res = None
    focal_plane_res_unit = None
    
    try:
        with Image.open(image_path) as img:
            exif_width = float(img.width)
            
            exif_data = img._getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    try:
                        if tag_id == 37386:  # FocalLength
                            focal_length = float(value)
                        elif tag_id == 40962:  # ExifImageWidth
                            exif_width = float(value)
                        elif tag_id == 41486:  # FocalPlaneXResolution
                            focal_plane_x_res = float(value)
                        elif tag_id == 41488:  # FocalPlaneResolutionUnit
                            focal_plane_res_unit = int(value)
                    except (TypeError, ValueError):
                        continue
    except Exception as e:
        logger.warning(
            "EXIF metadata extraction failed",
            exc_info=True,
            extra={"image_path": image_path}
        )
        
    return focal_length, exif_width, focal_plane_x_res, focal_plane_res_unit

def calculate_gsd(physical_width_mm, pixel_distance):
    """
    Calculates dynamic GSD scale directly from the physical marker width (mm) and the drawn line pixel distance.
    """
    if pixel_distance <= 0.0:
        raise ValueError("Pixel distance must be greater than zero.")
    return physical_width_mm / pixel_distance

def get_calibration_scale(image_path, image_width_px, reference_marker_width_mm=None):
    """
    Calculates GSD scale using a strict priority hierarchy:
      1. Reference marker (absolute ground truth, bypasses EXIF entirely)
      2. True EXIF hardware sensor dimensions (FocalPlaneXResolution + FocalLength)
      3. Uncalibrated fallback (0.1 mm/px)
    """
    focal_length, exif_width, focal_plane_x_res, focal_plane_res_unit = extract_exif_metadata(image_path)
    
    if not exif_width or exif_width <= 0:
        exif_width = float(image_width_px)
    
    if (focal_length and focal_length > 0.0 and
        focal_plane_x_res and focal_plane_x_res > 0.0 and
        exif_width > 0.0):
        
        try:
            if focal_plane_res_unit == 3:  # centimeters
                sensor_width_mm = (exif_width / focal_plane_x_res) * 10.0
            elif focal_plane_res_unit == 4:  # millimeters
                sensor_width_mm = exif_width / focal_plane_x_res
            else:  # default to inches (unit 2)
                sensor_width_mm = (exif_width / focal_plane_x_res) * 25.4
            
            if sensor_width_mm > 0.0:
                gsd = sensor_width_mm / (focal_length * exif_width)
                return gsd, "EXIF Calibrated"
        except (ZeroDivisionError, TypeError, ValueError):
            pass
    
    return DEFAULT_GSD, "Uncalibrated (Default GSD)"

def measure_crack_width(mask_binary):
    """
    Measures crack pixel width using morphological thinning and
    percentile distance transform sampling along the medial axis.
    Returns 0.0, [] if no foreground pixels exist in the mask.
    """
    if not isinstance(mask_binary, np.ndarray):
        logger.error("Input mask must be a numpy ndarray", extra={"input_type": str(type(mask_binary))})
        return 0.0, []

    if np.issubdtype(mask_binary.dtype, np.floating):
        mask_binary = np.nan_to_num(mask_binary, nan=0.0, posinf=0.0, neginf=0.0)
        
    if mask_binary.dtype != np.uint8:
        mask_binary = (mask_binary > 0).astype(np.uint8) * 255
    else:
        _, mask_binary = cv2.threshold(mask_binary, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    if cv2.countNonZero(mask_binary) == 0:
        logger.info("Empty mask passed to measure_crack_width, returning 0.0")
        return 0.0, []
    
    dist_transform = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)
    dist_transform = np.nan_to_num(dist_transform, nan=0.0, posinf=0.0, neginf=0.0)
    
    skeleton = cv2.ximgproc.thinning(mask_binary)
    
    skeleton_distances = dist_transform[skeleton > 0]
    
    if skeleton_distances.size == 0:
        logger.info("No skeleton points found after thinning")
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return 0.0, contours
    
    sorted_distances = np.sort(skeleton_distances)
    cutoff_index = int(len(sorted_distances) * PERCENTILE_CUTOFF)
    if cutoff_index < 1:
        cutoff_index = len(sorted_distances)
    trimmed = sorted_distances[:cutoff_index]
    target_radius = float(trimmed[-1]) if len(trimmed) > 0 else 0.0
    
    pixel_width = round(target_radius * WIDTH_MULTIPLIER, 2)
    
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    logger.info(
        "Crack width measured successfully",
        extra={
            "pixel_width": pixel_width,
            "target_radius": target_radius,
            "percentile_cutoff": PERCENTILE_CUTOFF,
            "width_multiplier": WIDTH_MULTIPLIER,
            "skeleton_points_count": len(skeleton_distances)
        }
    )
    return pixel_width, contours

def measure_crack_length(mask_binary, gsd=DEFAULT_GSD):
    """
    Measures physical crack length using topological graph extraction,
    iterative spur-pruning (removing terminal branches < MIN_SPUR_LENGTH_PX),
    and 8-connected neighbor traversal with diagonal sqrt(2) weighting.

    Returns: (length_mm, length_cm, pruned_skeleton)
    """
    if not isinstance(mask_binary, np.ndarray) or cv2.countNonZero(mask_binary) == 0:
        return 0.0, 0.0, np.zeros_like(mask_binary, dtype=np.uint8) if isinstance(mask_binary, np.ndarray) else np.array([])

    if mask_binary.dtype != np.uint8:
        mask_binary = (mask_binary > 0).astype(np.uint8) * 255

    # 1. Skeletonize using native OpenCV thinning
    skeleton = cv2.ximgproc.thinning(mask_binary)
    if cv2.countNonZero(skeleton) == 0:
        return 0.0, 0.0, skeleton

    skel_binary = (skeleton > 0).astype(np.uint8)
    
    # 2. Topological Spur-Pruning
    min_spur_len = MIN_SPUR_LENGTH_PX
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.float32)

    pruned = skel_binary.copy()
    
    changed = True
    iterations = 0
    max_iterations = 20
    
    while changed and iterations < max_iterations:
        changed = False
        iterations += 1
        neighbor_sum = cv2.filter2D(pruned.astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT)
        # Endpoints have center=10 and exactly 1 active neighbor -> neighbor_sum == 11
        endpoints = np.argwhere((pruned == 1) & (np.abs(neighbor_sum - 11.0) < 1e-3))

        for ep_y, ep_x in endpoints:
            if pruned[ep_y, ep_x] == 0:
                continue

            # Trace branch starting from endpoint
            branch_pixels = [(ep_y, ep_x)]
            curr_y, curr_x = ep_y, ep_x

            while True:
                ymin, ymax = max(0, curr_y - 1), min(pruned.shape[0], curr_y + 2)
                xmin, xmax = max(0, curr_x - 1), min(pruned.shape[1], curr_x + 2)
                
                neighbors = []
                for ny in range(ymin, ymax):
                    for nx in range(xmin, xmax):
                        if (ny != curr_y or nx != curr_x) and pruned[ny, nx] == 1:
                            neighbors.append((ny, nx))

                unvisited = [p for p in neighbors if p not in branch_pixels]

                if len(unvisited) == 1:
                    next_y, next_x = unvisited[0]
                    next_sum = neighbor_sum[next_y, next_x] - 10
                    branch_pixels.append((next_y, next_x))
                    if next_sum > 2:
                        # Reached a bifurcation branch node
                        break
                    curr_y, curr_x = next_y, next_x
                else:
                    break

            if len(branch_pixels) < min_spur_len:
                for py, px in branch_pixels:
                    p_neighbors = neighbor_sum[py, px] - 10
                    if p_neighbors <= 2 or (py, px) == (ep_y, ep_x):
                        pruned[py, px] = 0
                changed = True

    # 3. Traverse Pruned Skeleton with Diagonal sqrt(2) Weighting
    pruned_pts = np.argwhere(pruned == 1)
    if len(pruned_pts) == 0:
        pruned = skel_binary
        pruned_pts = np.argwhere(pruned == 1)

    total_pixel_len = 0.0
    visited_edges = set()

    for py, px in pruned_pts:
        for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            ny, nx = py + dy, px + dx
            if 0 <= ny < pruned.shape[0] and 0 <= nx < pruned.shape[1] and pruned[ny, nx] == 1:
                edge = tuple(sorted([(py, px), (ny, nx)]))
                if edge not in visited_edges:
                    visited_edges.add(edge)
                    step_dist = np.sqrt(dy**2 + dx**2)
                    total_pixel_len += step_dist

    length_mm = round(total_pixel_len * gsd, 2)
    length_cm = round(length_mm / 10.0, 2)

    logger.info(
        "Crack length measured with spur-pruning",
        extra={
            "length_mm": length_mm,
            "length_cm": length_cm,
            "total_pixel_len": round(total_pixel_len, 2),
            "min_spur_length_px": MIN_SPUR_LENGTH_PX
        }
    )

    return length_mm, length_cm, (pruned * 255).astype(np.uint8)

def measure_crack_orientation(mask_binary):
    """
    Computes crack principal axis orientation using PCA on skeleton/contour coordinates.
    Classifies orientation as 'Horizontal' (|theta| <= 15°),
    'Vertical' (75° <= |theta| <= 90°), or 'Diagonal' (15° < |theta| < 75°).

    Returns: (orientation_label, angle_degrees)
    """
    if not isinstance(mask_binary, np.ndarray) or cv2.countNonZero(mask_binary) == 0:
        return "None", 0.0

    if mask_binary.dtype != np.uint8:
        mask_binary = (mask_binary > 0).astype(np.uint8) * 255

    pts = np.argwhere(mask_binary > 0)
    if len(pts) < 5:
        return "Horizontal", 0.0

    # Swap (y, x) to (x, y)
    pts_xy = pts[:, [1, 0]].astype(np.float64)

    # Perform PCA using covariance matrix
    mean = np.mean(pts_xy, axis=0)
    pts_centered = pts_xy - mean
    cov = np.cov(pts_centered, rowvar=False)

    evals, evecs = np.linalg.eigh(cov)

    primary_vec = evecs[:, np.argmax(evals)]  # (v_x, v_y)
    vx, vy = primary_vec[0], primary_vec[1]

    angle_rad = np.arctan2(vy, vx)
    angle_deg = round(float(np.degrees(angle_rad)), 1)

    if angle_deg > 90.0:
        angle_deg -= 180.0
    elif angle_deg < -90.0:
        angle_deg += 180.0

    abs_angle = abs(angle_deg)

    if abs_angle <= 15.0:
        orientation = "Horizontal"
    elif 75.0 <= abs_angle <= 90.0:
        orientation = "Vertical"
    else:
        orientation = "Diagonal"

    logger.info(
        "Crack orientation calculated",
        extra={
            "orientation": orientation,
            "angle_deg": angle_deg
        }
    )

    return orientation, angle_deg
