#!/usr/bin/env python3
"""
Lightweight CLI Entrypoint for the VisionInspect AI Platform.
Loads modular vision/calibration/pipeline packages and executes the orchestrator.
"""

import os
import sys
import argparse
import json

# Add project root workspace directory to sys.path to resolve modular imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the modular run_pipeline function
try:
    from pipeline.orchestrator import run_pipeline
except ImportError as e:
    print(f"Error: Failed to import pipeline orchestrator: {e}", file=sys.stderr)
    sys.exit(3)

def main():
    parser = argparse.ArgumentParser(
        description="VisionInspect AI modular execution entrypoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--image-path", required=True, help="Path to structural image file")
    parser.add_argument("--output-dir", default="", help="Directory to output annotated image")
    parser.add_argument("--reference-marker-width-mm", type=float, default=None, help="Known reference marker width in mm for physical calibration")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for YOLO segmentation model")
    parser.add_argument("--gsd", type=float, default=None, help="Explicit GSD value to use for scale calibration")
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image path '{args.image_path}' does not exist.", file=sys.stderr)
        sys.exit(1)
        
    try:
        output_dir = args.output_dir if args.output_dir else None
        json_data, payload_summary = run_pipeline(
            args.image_path,
            output_dir=output_dir,
            reference_marker_width_mm=args.reference_marker_width_mm,
            conf=args.conf,
            gsd=args.gsd
        )
        
        # Output standard schema outputs expected by downstream services
        print(f"JSON PAYLOAD: {json.dumps(json_data)}")
        print(payload_summary)
    except Exception as e:
        print(f"Error executing pipeline: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(2)

if __name__ == "__main__":
    main()
