import os
import sys
import cv2
import numpy as np
from PIL import Image, ExifTags
from config import PERCENTILE_CUTOFF, WIDTH_MULTIPLIER, DEFAULT_GSD, get_logger

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
    # Priority 1: Reference marker is absolute ground truth — handled upstream via --gsd.
    # If reference_marker_width_mm was provided, the caller already computed GSD and
    # passed it via the --gsd CLI flag, so this function is only reached when no
    # explicit GSD was set. We still check here as a defensive guard.
    
    # Priority 2: True EXIF hardware sensor dimensions
    focal_length, exif_width, focal_plane_x_res, focal_plane_res_unit = extract_exif_metadata(image_path)
    
    if not exif_width or exif_width <= 0:
        exif_width = float(image_width_px)
    
    if (focal_length and focal_length > 0.0 and
        focal_plane_x_res and focal_plane_x_res > 0.0 and
        exif_width > 0.0):
        
        try:
            # Derive true sensor width from FocalPlaneXResolution
            # FocalPlaneResolutionUnit: 2 = inches, 3 = centimeters, 4 = millimeters
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
    
    # Priority 3: Uncalibrated fallback
    return DEFAULT_GSD, "Uncalibrated (Default GSD)"

def measure_crack_width(mask_binary):
    """
    Measures crack pixel width using morphological thinning and
    percentile distance transform sampling along the medial axis.
    Returns 0.0, [] if no foreground pixels exist in the mask.
    """
    # Defensive casting to uint8 binary mask if float array containing NaN/Inf is passed
    if not isinstance(mask_binary, np.ndarray):
        logger.error("Input mask must be a numpy ndarray", extra={"input_type": str(type(mask_binary))})
        return 0.0, []

    # Clean NaNs/Infs from input if it is a float array
    if np.issubdtype(mask_binary.dtype, np.floating):
        mask_binary = np.nan_to_num(mask_binary, nan=0.0, posinf=0.0, neginf=0.0)
        
    # Convert mask to uint8 binary 0/255 for cv2 processing
    if mask_binary.dtype != np.uint8:
        mask_binary = (mask_binary > 0).astype(np.uint8) * 255
    else:
        # Ensure it is strictly binary (0 or 255)
        _, mask_binary = cv2.threshold(mask_binary, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    if cv2.countNonZero(mask_binary) == 0:
        logger.info("Empty mask passed to measure_crack_width, returning 0.0")
        return 0.0, []
    
    # 1. Compute distance transform on the binary mask and clean NaN/Inf output defensively
    dist_transform = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)
    dist_transform = np.nan_to_num(dist_transform, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 2. Skeletonize the binary mask using native OpenCV thinning
    skeleton = cv2.ximgproc.thinning(mask_binary)
    
    # 3. Extract distance transform values strictly along the skeleton
    skeleton_distances = dist_transform[skeleton > 0]
    
    if skeleton_distances.size == 0:
        logger.info("No skeleton points found after thinning")
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return 0.0, contours
    
    # 4. Sort and drop top outliers using PERCENTILE_CUTOFF
    sorted_distances = np.sort(skeleton_distances)
    cutoff_index = int(len(sorted_distances) * PERCENTILE_CUTOFF)
    if cutoff_index < 1:
        cutoff_index = len(sorted_distances)
    trimmed = sorted_distances[:cutoff_index]
    target_radius = float(trimmed[-1]) if len(trimmed) > 0 else 0.0
    
    # 5. Multiply the target percentile radius by WIDTH_MULTIPLIER to get the diameter (pixel width)
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
