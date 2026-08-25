#!/usr/bin/env python3
"""
Visual Debugging Tool for Topological Spur-Pruning & PCA Crack Orientation.

Validates the mathematical correctness of:
  - Raw unpruned skeleton extraction
  - Iterative topological spur-pruning (MIN_SPUR_LENGTH_PX = 10)
  - 8-connected neighbor traversal with diagonal sqrt(2) weighting
  - PCA principal axis orientation angle calculation

Outputs:
  - Terminal comparison of raw vs. pruned skeleton length
  - debug_skeleton.png visual artifact showing mask, raw skeleton, pruned skeleton, and PCA angle axis.
"""

import os
import sys
import cv2
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from calibration.scale import measure_crack_length, measure_crack_orientation, DEFAULT_GSD


def generate_synthetic_y_crack(size=(500, 500)):
    """
    Generates a 500x500 uint8 binary mask containing a curved, Y-shaped crack
    with multiple short dead-end branches (spurs < 10px long).
    """
    img = np.zeros(size, dtype=np.uint8)

    # 1. Main curved stem: from (100, 80) to bifurcation node (280, 240)
    pts_main = []
    for t in np.linspace(0, 1, 100):
        x = int(100 + 180 * t + 12 * np.sin(t * np.pi * 3))
        y = int(80 + 160 * t + 8 * np.cos(t * np.pi * 2))
        pts_main.append((x, y))

    for i in range(len(pts_main) - 1):
        cv2.line(img, pts_main[i], pts_main[i + 1], 255, 3)

    # 2. Right branch of Y: from (280, 240) to (420, 400)
    pts_right = []
    for t in np.linspace(0, 1, 80):
        x = int(280 + 140 * t + 8 * np.sin(t * np.pi * 2))
        y = int(240 + 160 * t)
        pts_right.append((x, y))

    for i in range(len(pts_right) - 1):
        cv2.line(img, pts_right[i], pts_right[i + 1], 255, 3)

    # 3. Left branch of Y: from (280, 240) to (140, 420)
    pts_left = []
    for t in np.linspace(0, 1, 80):
        x = int(280 - 140 * t - 8 * np.cos(t * np.pi * 2))
        y = int(240 + 180 * t)
        pts_left.append((x, y))

    for i in range(len(pts_left) - 1):
        cv2.line(img, pts_left[i], pts_left[i + 1], 255, 3)

    # Thin the main structure first to get a baseline 1px skeleton, then attach 1px micro-spurs
    skel_base = cv2.ximgproc.thinning(img)
    skel_with_spurs = skel_base.copy()

    # Find skeleton points to attach perpendicular micro-spurs (length 5 to 7 pixels)
    skel_pts = np.argwhere(skel_base > 0)
    step = len(skel_pts) // 10
    
    for idx in range(step, len(skel_pts) - step, step):
        py, px = skel_pts[idx]
        # Alternate perpendicular vectors
        direction = 1 if idx % (2 * step) == 0 else -1
        for len_idx in range(1, 7):
            sy = py + direction * len_idx
            sx = px + direction * len_idx
            if 0 <= sy < size[0] and 0 <= sx < size[1]:
                skel_with_spurs[sy, sx] = 255

    # Dilate slightly to form realistic concrete defect mask with micro-spurs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_dilated = cv2.dilate(skel_with_spurs, kernel, iterations=1)

    return mask_dilated


def calculate_raw_skeleton_length(skel_binary, gsd=DEFAULT_GSD):
    """Calculates total pixel length of raw unpruned skeleton with diagonal sqrt(2) weighting."""
    pts = np.argwhere(skel_binary > 0)
    visited_edges = set()
    total_pixel_len = 0.0

    for py, px in pts:
        for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            ny, nx = py + dy, px + dx
            if 0 <= ny < skel_binary.shape[0] and 0 <= nx < skel_binary.shape[1] and skel_binary[ny, nx] > 0:
                edge = tuple(sorted([(py, px), (ny, nx)]))
                if edge not in visited_edges:
                    visited_edges.add(edge)
                    step_dist = np.sqrt(dy**2 + dx**2)
                    total_pixel_len += step_dist

    length_mm = round(total_pixel_len * gsd, 2)
    return total_pixel_len, length_mm


def main():
    # 1. Generate Synthetic Mask with micro-spurs
    mask = generate_synthetic_y_crack()

    # 2. Extract Raw Unpruned Skeleton
    raw_skeleton = cv2.ximgproc.thinning(mask)
    raw_pixel_len, raw_len_mm = calculate_raw_skeleton_length(raw_skeleton, DEFAULT_GSD)

    # 3. Run Scale Calibration Functions (Pruned Length & PCA Orientation)
    pruned_len_mm, pruned_len_cm, pruned_skeleton = measure_crack_length(mask, gsd=DEFAULT_GSD)
    orientation, angle_deg = measure_crack_orientation(mask)

    # Calculate pruned pixel length
    pruned_pixel_len, _ = calculate_raw_skeleton_length(pruned_skeleton, DEFAULT_GSD)

    # 4. Print Terminal Metrics
    print("=" * 65)
    print("           TOPOLOGICAL SPUR-PRUNING & PCA DIAGNOSTIC")
    print("=" * 65)
    print(f" Raw Unpruned Skeleton Length : {raw_pixel_len:.2f} px ({raw_len_mm:.2f} mm)")
    print(f" Pruned Skeleton Spine Length : {pruned_pixel_len:.2f} px ({pruned_len_mm:.2f} mm)")
    print(f" Snipped Spur Length Diff     : {raw_pixel_len - pruned_pixel_len:.2f} px ({(raw_len_mm - pruned_len_mm):.2f} mm)")
    print(f" Reduction Percentage         : {((raw_pixel_len - pruned_pixel_len) / raw_pixel_len * 100):.1f}%")
    print("-" * 65)
    print(f" PCA Principal Axis Angle     : {angle_deg}°")
    print(f" Classified Orientation Tag   : {orientation}")
    print("=" * 65)

    # 5. Create Visual Debug Image
    h, w = mask.shape
    vis_img = np.zeros((h, w, 3), dtype=np.uint8)
    vis_img[mask > 0] = (80, 80, 80)

    # Raw unpruned skeleton in RED (0, 0, 255)
    vis_img[raw_skeleton > 0] = (0, 0, 255)

    # Final pruned skeleton in GREEN (0, 255, 0) overlaid on top
    vis_img[pruned_skeleton > 0] = (0, 255, 0)

    # Draw PCA Angle Axis in BLUE (255, 0, 0)
    pts = np.argwhere(mask > 0)
    cy, cx = np.mean(pts, axis=0)
    cx, cy = int(cx), int(cy)

    angle_rad = np.radians(angle_deg)
    axis_len = 150
    p1 = (int(cx - axis_len * np.cos(angle_rad)), int(cy - axis_len * np.sin(angle_rad)))
    p2 = (int(cx + axis_len * np.cos(angle_rad)), int(cy + axis_len * np.sin(angle_rad)))
    cv2.line(vis_img, p1, p2, (255, 0, 0), 2, cv2.LINE_AA)

    # Yellow dot at centroid
    cv2.circle(vis_img, (cx, cy), 6, (0, 255, 255), -1)

    # Legend Overlay
    cv2.rectangle(vis_img, (10, 10), (340, 110), (20, 20, 20), -1)
    cv2.rectangle(vis_img, (10, 10), (340, 110), (180, 180, 180), 1)

    cv2.putText(vis_img, "Gray: Defect Mask", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    cv2.putText(vis_img, "Red: Snipped Micro-Spurs", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    cv2.putText(vis_img, "Green: Pruned Skeleton Spine", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    cv2.putText(vis_img, f"Blue Line: PCA Axis ({angle_deg}deg - {orientation})", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 100), 1)

    output_filename = "debug_skeleton.png"
    cv2.imwrite(output_filename, vis_img)
    print(f"\n[DEBUG] Visual artifact saved to disk: {output_filename}")


if __name__ == "__main__":
    main()
