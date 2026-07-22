#!/usr/bin/env python3
"""
End-to-End Integration Test for the Geospatial Inspector Agent.
1. Regenerates the EXIF image mock_crack_exif.jpg with coordinates (Lat 45.5 N, Lon 75.25 W).
2. Runs analyze_image.py to extract defect measurements and GPS coordinates.
3. Parses the VISION PAYLOAD output.
4. Executes BOTH log_inspection.py (to log the inspection in PostGIS)
   and generate_ticket.py (to render the PDF ticket).
5. Queries the PostGIS database to confirm the insertion and checks the workspace for the PDF.
"""

import os
import sys
import subprocess
import re
import json
from PIL import Image, ImageDraw

WORKSPACE_DIR = "/Users/yekowski/agy2-projects/infrastructure-inspector"
ANALYZE_SCRIPT = os.path.join(WORKSPACE_DIR, ".agents/skills/inspector/scripts/analyze_image.py")
LOG_SCRIPT = os.path.join(WORKSPACE_DIR, ".agents/skills/inspector/scripts/log_inspection.py")
TICKET_SCRIPT = os.path.join(WORKSPACE_DIR, ".agents/skills/inspector/scripts/generate_ticket.py")

def recreate_mock_image():
    print("Step 1: Regenerating mock image with EXIF coordinates Lat 45.5 N, Lon 75.25 W...")
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)
    # Draw simulated crack
    draw.line([(400, 100), (410, 500)], fill='black', width=8)
    
    exif = img.getexif()
    gps_ifd = exif.get_ifd(34853)
    gps_ifd[1] = 'N'
    gps_ifd[2] = (45.0, 30.0, 0.0) 
    gps_ifd[3] = 'W'
    gps_ifd[4] = (75.0, 15.0, 0.0)
    
    image_path = os.path.join(WORKSPACE_DIR, "mock_crack_exif.jpg")
    img.save(image_path, "JPEG", exif=exif)
    print(f" -> Mock image saved at: {image_path}")
    return image_path

def run_image_analysis(image_path):
    print("\nStep 2: Running analyze_image.py against the image...")
    cmd = ["python3", ANALYZE_SCRIPT, "--image-path", image_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if result.returncode != 0:
        print(f"Error during image analysis: {result.stderr}", file=sys.stderr)
        sys.exit(1)
        
    print(f" -> Raw Output:\n{result.stdout.strip()}")
    
    # Extract the VISION PAYLOAD string
    payload_line = None
    for line in result.stdout.splitlines():
        if "VISION PAYLOAD:" in line:
            payload_line = line
            break
            
    if not payload_line:
        print("Error: Could not find VISION PAYLOAD line in output.", file=sys.stderr)
        sys.exit(1)
        
    print(f" -> Captured Payload: '{payload_line}'")
    return payload_line

def simulate_agent_routing(payload_string):
    print("\nStep 3: Simulating agent decision and routing logic...")
    print(" -> Parsing payload fields...")
    
    # Regex parse:
    # VISION PAYLOAD: Image {name} analyzed. Crack width: {width}mm. Status: {status}. Coordinates: {lon}, {lat}.
    pattern = r"VISION PAYLOAD:\s*Image\s*(?P<name>[\w\.\-]+)\s*analyzed\.\s*Crack width:\s*(?P<width>[\d\.]+)mm\.\s*Status:\s*(?P<status>[^.]+)\.\s*Coordinates:\s*(?P<lon>[\-\d\.]+),\s*(?P<lat>[\-\d\.]+)\."
    match = re.search(pattern, payload_string)
    
    if not match:
        print("Error: Payload string format does not match regex pattern.", file=sys.stderr)
        sys.exit(1)
        
    data = match.groupdict()
    print("Parsed Data:")
    print(json.dumps(data, indent=2))
    
    lon = float(data["lon"])
    lat = float(data["lat"])
    width = float(data["width"])
    status = data["status"]
    
    # Agent Action Selection based on rules
    print(f" -> Evaluation: Status is '{status}', Crack width = {width}mm")
    
    actions = []
    # Always log to database
    actions.append("log_database")
    
    if status == "Requires PDF ticket" or width > 2.0:
        print(" -> Action Trigger: Crack width exceeds 2.0mm. Agent routes to BOTH Log and PDF generation.")
        actions.append("generate_pdf")
    else:
        print(" -> Action Trigger: Crack width <= 2.0mm. Agent routes to Log only.")
        
    return actions, {
        "ticket_id": "TK-CV-01",
        "inspector_name": "CV Pipeline",
        "status": status,
        "lon": lon,
        "lat": lat,
        "notes": f"Vision analysis completed for image {data['name']}. Detected crack width: {width}mm."
    }

def execute_agent_actions(actions, params):
    print("\nStep 4: Executing routed tools as the agent...")
    
    env = os.environ.copy()
    env["DB_CONN_STR"] = "postgresql://postgres:supersecret@localhost:5433/inspector_db"
    
    if "log_database" in actions:
        print(" -> Triggering database logging tool (log_inspection.py)...")
        cmd = [
            "python3", LOG_SCRIPT,
            "--ticket-id", params["ticket_id"],
            "--inspector-name", params["inspector_name"],
            "--status", params["status"],
            "--lon", str(params["lon"]),
            "--lat", str(params["lat"]),
            "--notes", params["notes"]
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        if result.returncode == 0:
            print(f"    [OK] DB Log Successful: {result.stdout.strip()}")
        else:
            print(f"    [FAIL] DB Log Failed: {result.stderr.strip()}", file=sys.stderr)
            sys.exit(1)
            
    if "generate_pdf" in actions:
        pdf_path = os.path.join(WORKSPACE_DIR, f"ticket_{params['ticket_id']}.pdf")
        print(f" -> Triggering PDF ticket generator tool (generate_ticket.py)...")
        cmd = [
            "python3", TICKET_SCRIPT,
            "--output-path", pdf_path,
            "--ticket-id", params["ticket_id"],
            "--inspector-name", params["inspector_name"],
            "--status", params["status"],
            "--lon", str(params["lon"]),
            "--lat", str(params["lat"]),
            "--notes", params["notes"]
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        if result.returncode == 0:
            print(f"    [OK] PDF Ticket Generation Successful: {result.stdout.strip()}")
            print(f"    [OK] PDF File verified exists: {os.path.exists(pdf_path)}")
        else:
            print(f"    [FAIL] PDF Generation Failed: {result.stderr.strip()}", file=sys.stderr)
            sys.exit(1)

def verify_live_database_records():
    print("\nStep 5: Querying live PostGIS database to verify coordinates (-75.25, 45.5)...")
    cmd = [
        "docker", "exec", "postgis_inspector",
        "psql", "-U", "postgres", "-d", "inspector_db",
        "-c", "SELECT ticket_id, inspector_name, status, ST_AsText(geom) FROM inspection_logs WHERE ticket_id = 'TK-CV-01';"
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        print(f"Database Query Output:\n{result.stdout.strip()}")
        # Check if coordinates match
        if "-75.25 45.5" in result.stdout:
            print(" -> [VERIFIED] Live coordinates (-75.25, 45.5) successfully written to database record!")
            return True
        else:
            print(" -> [FAIL] DB Record coordinates do not match expected (-75.25, 45.5)!")
            return False
    else:
        print(f" -> [FAIL] Database verification query failed: {result.stderr.strip()}", file=sys.stderr)
        return False

def main():
    print("==================================================")
    print("Live End-to-End Agentic Integration Test")
    print("==================================================\n")
    
    # 1. Setup Image
    image_path = recreate_mock_image()
    
    # 2. Run Image Preprocessing
    payload = run_image_analysis(image_path)
    
    # 3. Simulate Agent Decisions
    actions, params = simulate_agent_routing(payload)
    
    # 4. Trigger Tools
    execute_agent_actions(actions, params)
    
    # 5. Verify Database Records
    db_verified = verify_live_database_records()
    
    # Clean up mock image
    try:
        os.remove(image_path)
    except Exception:
        pass
        
    print("\n==================================================")
    print(f"E2E Integration Status: {'PASS' if db_verified else 'FAIL'}")
    print("==================================================")

if __name__ == "__main__":
    main()
