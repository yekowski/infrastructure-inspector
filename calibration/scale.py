import os
import sys
import cv2
import numpy as np
from PIL import Image, ExifTags

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
        print(f"[WARNING] EXIF GPS extraction failed: {e}. Using default coordinates.", file=sys.stderr)
        
    return default_coords

def extract_exif_metadata(image_path):
    """
    Attempts to extract FocalLength, ExifImageWidth, and SubjectDistance from image EXIF data.
    """
    focal_length = None
    exif_width = None
    subject_distance = None
    
    try:
        with Image.open(image_path) as img:
            exif_width = float(img.width)
            
            exif_data = img._getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag_name = ExifTags.TAGS.get(tag_id, None)
                    if tag_name == 'FocalLength':
                        focal_length = float(value)
                    elif tag_name == 'SubjectDistance':
                        subject_distance = float(value)
                    elif tag_name == 'ExifImageWidth':
                        exif_width = float(value)
                    if tag_id == 37386:
                        focal_length = float(value)
                    elif tag_id == 37382:
                        subject_distance = float(value)
                    elif tag_id == 40962:
                        exif_width = float(value)
    except Exception as e:
        print(f"[WARNING] EXIF metadata extraction failed: {e}", file=sys.stderr)
        
    return focal_length, exif_width, subject_distance

def calculate_gsd(physical_width_mm, pixel_distance):
    """
    Calculates dynamic GSD scale directly from the physical marker width (mm) and the drawn line pixel distance.
    """
    if pixel_distance <= 0.0:
        raise ValueError("Pixel distance must be greater than zero.")
    return physical_width_mm / pixel_distance

def get_calibration_scale(image_path, image_width_px, reference_marker_width_mm=None):
    """
    Calculates GSD scale using the photogrammetry GSD formula:
    scale_mm_per_px = (SubjectDistance * 36.0) / (FocalLength * ExifImageWidth)
    Falls back defensively to default GSD (0.1 mm/px) if EXIF info is missing.
    """
    focal_length, exif_width, subject_distance = extract_exif_metadata(image_path)
    
    if not exif_width or exif_width <= 0:
        exif_width = float(image_width_px)
        
    if (focal_length and focal_length > 0.0 and 
        subject_distance and subject_distance > 0.0 and 
        exif_width > 0.0):
        
        # Convert SubjectDistance from meters to millimeters if stored in meters
        subject_distance_mm = subject_distance
        if subject_distance < 100.0:
            subject_distance_mm = subject_distance * 1000.0
            
        gsd = (subject_distance_mm * 36.0) / (focal_length * exif_width)
        calibration_status = "EXIF Calibrated"
    else:
        gsd = 0.1
        calibration_status = "Uncalibrated (Default GSD)"
        
    return gsd, calibration_status

def measure_crack_width(mask_binary):
    """
    Applies cv2.distanceTransform and calculates the maximum pixel width of the crack
    by multiplying the distance transform radius value by 2.0 (diameter math).
    Returns 0.0, [] if no mask binary exists.
    """
    if cv2.countNonZero(mask_binary) == 0:
        return 0.0, []
        
    dist_transform = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)
    max_pixel_width = float(np.max(dist_transform) * 2.0) if dist_transform.size > 0 else 0.0
    
    if max_pixel_width <= 0.0:
        max_pixel_width = 0.0
        
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return round(max_pixel_width, 2), contours
