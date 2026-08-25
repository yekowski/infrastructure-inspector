#!/usr/bin/env python3
"""
Video Homography MVP — Spatial Canvas Camera Tracking Proof-of-Concept.

Proves the SIFT-based image registration math works on raw video frames
before integrating the YOLO semantic path. Strictly follows the architectural
constraints defined in AGENTS.md and SKILL.md:

  - Streaming Memory Discipline (frame-by-frame generator, no list accumulation)
  - SIFT as primary keypoint detector (mandated for low-texture concrete)
  - Homography validation with configurable thresholds via config.py
  - Structured JSON logging via config.get_logger()

Usage:
    python3 video/homography_mvp.py --video-path <path_to_video> [--target-fps 3]
"""

import os
import sys
import argparse
import json
import numpy as np
import cv2

# Add project root to sys.path for config imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import (
    get_logger,
    MIN_HOMOGRAPHY_INLIERS,
    MAX_HOMOGRAPHY_DET_VARIANCE,
    MAX_PERSPECTIVE_SKEW,
)

logger = get_logger("video.homography_mvp")


# ── Frame Sampling Generator (Streaming Memory Discipline) ──────────────────

def sample_frames(video_path, target_fps=3):
    """
    Generator that yields (frame_index, timestamp_sec, frame_bgr) tuples
    sampled at target_fps from the source video.

    Strictly follows the Frame Iterator Pattern: frames are yielded one-at-a-time.
    No frame accumulation into lists or arrays.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Failed to open video file", extra={"video_path": video_path})
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    try:
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if native_fps <= 0:
            logger.warning("Could not read native FPS, defaulting to 30",
                           extra={"video_path": video_path})
            native_fps = 30.0

        interval = max(1, int(native_fps / target_fps))

        logger.info("Video opened for sampling", extra={
            "video_path": video_path,
            "native_fps": round(native_fps, 2),
            "target_fps": target_fps,
            "frame_interval": interval,
            "total_frames": total_frames
        })

        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % interval == 0:
                timestamp_sec = round(frame_index / native_fps, 3)
                yield frame_index, timestamp_sec, frame

            frame_index += 1

    finally:
        cap.release()
        logger.info("VideoCapture released", extra={"video_path": video_path})


# ── SIFT Spatial Path ────────────────────────────────────────────────────────

def extract_sift_features(frame_bgr):
    """
    Spatial Path: convert to grayscale, extract SIFT keypoints and descriptors.
    SIFT is mandated over ORB for low-texture concrete surfaces.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_descriptors(desc_prev, desc_curr):
    """
    Match SIFT descriptors between consecutive frames using BFMatcher with L2 norm.
    Applies Lowe's ratio test (threshold: 0.75) to filter ambiguous matches.
    Returns list of good DMatch objects.
    """
    if desc_prev is None or desc_curr is None:
        return []

    if len(desc_prev) < 2 or len(desc_curr) < 2:
        return []

    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = bf.knnMatch(desc_curr, desc_prev, k=2)

    good_matches = []
    for match_pair in raw_matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    return good_matches


# ── Homography Computation & Validation ──────────────────────────────────────

def compute_and_validate_homography(kp_curr, kp_prev, good_matches):
    """
    Compute homography from current frame to previous frame's coordinate space.
    Validates against configurable thresholds:
      - MIN_HOMOGRAPHY_INLIERS (default: 10)
      - MAX_HOMOGRAPHY_DET_VARIANCE (default: 10.0, check: 0.1 < |det(H)| < max)
      - MAX_PERSPECTIVE_SKEW (default: 0.001, check: |H[2,0]| <= max and |H[2,1]| <= max)

    Returns (H, num_inliers, det_H, is_valid) or (None, 0, 0.0, False) on failure.
    """
    if len(good_matches) < 4:
        return None, len(good_matches), 0.0, False

    src_pts = np.float32([kp_curr[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_prev[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None or inlier_mask is None:
        return None, 0, 0.0, False

    num_inliers = int(np.sum(inlier_mask))
    det_H = float(np.linalg.det(H))

    # Validation gates (AGENTS.md §6 — Video Pipeline Code Constraints)
    inlier_ok = num_inliers >= MIN_HOMOGRAPHY_INLIERS
    det_ok = 0.1 < abs(det_H) < MAX_HOMOGRAPHY_DET_VARIANCE
    perspective_ok = abs(H[2, 0]) <= MAX_PERSPECTIVE_SKEW and abs(H[2, 1]) <= MAX_PERSPECTIVE_SKEW
    is_valid = inlier_ok and det_ok and perspective_ok

    return H, num_inliers, det_H, is_valid


# ── Main Pipeline ────────────────────────────────────────────────────────────

def run_homography_mvp(video_path, target_fps=3):
    """
    Processes a video file, sampling at target_fps, computing SIFT homography
    between consecutive frames, and outputting structured JSON logs per frame.

    Returns a summary dict with counts of registered/skipped frames.
    """
    results = []
    prev_keypoints = None
    prev_descriptors = None

    frames_processed = 0
    frames_registered = 0
    frames_skipped = 0

    for frame_index, timestamp_sec, frame in sample_frames(video_path, target_fps):
        frames_processed += 1

        # Spatial Path: SIFT feature extraction
        kp_curr, desc_curr = extract_sift_features(frame)
        num_keypoints = len(kp_curr)

        # First frame: establish baseline, no registration possible
        if prev_descriptors is None:
            frame_result = {
                "frame_index": frame_index,
                "timestamp_sec": timestamp_sec,
                "status": "baseline",
                "keypoints_detected": num_keypoints,
                "matches": 0,
                "inliers": 0,
                "det_H": 0.0,
                "homography_valid": False,
                "reason": "First frame — establishing spatial baseline"
            }
            results.append(frame_result)
            logger.info("Frame processed", extra=frame_result)

            prev_keypoints = kp_curr
            prev_descriptors = desc_curr
            continue

        # Match against previous frame
        good_matches = match_descriptors(prev_descriptors, desc_curr)
        num_matches = len(good_matches)

        # Compute and validate homography
        H, num_inliers, det_H, is_valid = compute_and_validate_homography(
            kp_curr, prev_keypoints, good_matches
        )

        if is_valid:
            frames_registered += 1
            frame_result = {
                "frame_index": frame_index,
                "timestamp_sec": timestamp_sec,
                "status": "registered",
                "keypoints_detected": num_keypoints,
                "matches": num_matches,
                "inliers": num_inliers,
                "det_H": round(det_H, 6),
                "homography_valid": True
            }
        else:
            frames_skipped += 1
            # Determine skip reason
            reasons = []
            if num_matches < 4:
                reasons.append(f"insufficient matches ({num_matches} < 4)")
            elif num_inliers < MIN_HOMOGRAPHY_INLIERS:
                reasons.append(f"inliers below threshold ({num_inliers} < {MIN_HOMOGRAPHY_INLIERS})")
            if H is not None and not (0.1 < abs(det_H) < MAX_HOMOGRAPHY_DET_VARIANCE):
                reasons.append(f"degenerate det(H)={det_H:.4f} outside (0.1, {MAX_HOMOGRAPHY_DET_VARIANCE})")

            frame_result = {
                "frame_index": frame_index,
                "timestamp_sec": timestamp_sec,
                "status": "skipped",
                "keypoints_detected": num_keypoints,
                "matches": num_matches,
                "inliers": num_inliers,
                "det_H": round(det_H, 6),
                "homography_valid": False,
                "reason": "; ".join(reasons) if reasons else "unknown validation failure"
            }

        results.append(frame_result)
        logger.info("Frame processed", extra=frame_result)

        # Update previous frame reference (always advance, even on skip)
        prev_keypoints = kp_curr
        prev_descriptors = desc_curr

    summary = {
        "video_path": video_path,
        "target_fps": target_fps,
        "frames_sampled": frames_processed,
        "frames_registered": frames_registered,
        "frames_skipped": frames_skipped,
        "registration_rate_pct": round(
            (frames_registered / max(1, frames_processed - 1)) * 100, 1
        ),
        "thresholds": {
            "MIN_HOMOGRAPHY_INLIERS": MIN_HOMOGRAPHY_INLIERS,
            "MAX_HOMOGRAPHY_DET_VARIANCE": MAX_HOMOGRAPHY_DET_VARIANCE
        }
    }

    logger.info("Homography MVP complete", extra=summary)
    return summary, results


# ── CLI Entrypoint ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Video Homography MVP — SIFT spatial registration proof-of-concept.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--video-path", required=True, help="Path to input video file")
    parser.add_argument("--target-fps", type=int, default=3,
                        help="Target sampling rate in FPS")
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        logger.error("Video file not found", extra={"video_path": args.video_path})
        sys.exit(1)

    summary, results = run_homography_mvp(args.video_path, args.target_fps)

    # Output summary to stdout for downstream consumption
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
