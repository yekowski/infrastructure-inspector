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
    
    # Extract the JSON payload line
    json_line = None
    for line in result.stdout.splitlines():
        if line.startswith("JSON PAYLOAD:"):
            json_line = line.replace("JSON PAYLOAD:", "").strip()
            break
            
    if not json_line:
        print("Error: Could not find JSON PAYLOAD line in output.", file=sys.stderr)
        sys.exit(1)
        
    print(f" -> Captured Payload: '{json_line}'")
    return json_line

def simulate_agent_routing(json_payload_str):
    print("\nStep 3: Simulating agent decision and routing logic...")
    print(" -> Parsing payload fields...")
    
    try:
        json_data = json.loads(json_payload_str)
    except Exception as e:
        print(f"Error parsing JSON payload: {e}", file=sys.stderr)
        sys.exit(1)
        
    print("Parsed Data:")
    print(json.dumps(json_data, indent=2))
    
    lon = float(json_data["lon"])
    lat = float(json_data["lat"])
    width = float(json_data["crack_width_mm"])
    status = json_data["status"]
    severity = json_data["severity"]
    spalling_area = json_data["spalling_area_mm2"]
    crack_type = json_data.get("crack_type", "Radial Floor Crack")
    maintenance_action = json_data.get("maintenance_action")
    uncertainty = json_data.get("uncertainty_mm")
    
    print(f" -> Evaluation: Status is '{status}', Severity is '{severity}', Crack width = {width}mm, Spalling area = {spalling_area}mm2")
    
    actions = []
    # Always log to database
    actions.append("log_database")
    
    if status == "Requires PDF ticket" or width > 2.0:
        print(" -> Action Trigger: Severity triggers PDF ticket. Agent routes to BOTH Log and PDF generation.")
        actions.append("generate_pdf")
    else:
        print(" -> Action Trigger: Severity does not warrant PDF. Agent routes to Log only.")
        
    if "Spalling" in crack_type and "Crack" not in crack_type:
        notes = f"Spalling Area: {spalling_area} mm². Action: {maintenance_action}"
    elif "Crack" in crack_type and "Spalling" not in crack_type:
        notes = f"COD Width: {width}mm ±{uncertainty}mm. Action: {maintenance_action}"
    else:
        notes = f"COD Width: {width}mm ±{uncertainty}mm. Spalling Area: {spalling_area} mm². Action: {maintenance_action}"
        
    return actions, {
        "ticket_id": "TK-CV-01",
        "inspector_name": "CV Pipeline",
        "status": status,
        "severity": severity,
        "lon": lon,
        "lat": lat,
        "notes": notes,
        "crack_type": crack_type,
        "crack_width": f"{width}mm",
        "uncertainty": f"±{uncertainty}mm",
        "confidence": f"{json_data.get('confidence_pct', 92.4)}%",
        "priority": json_data.get("priority", "High"),
        "maintenance_action": maintenance_action,
        "spalling_area": f"{spalling_area} mm²"
    }

def execute_agent_actions(actions, params):
    print("\nStep 4: Executing routed tools as the agent...")
    
    env = os.environ.copy()
    env["DB_CONN_STR"] = "postgresql://postgres.iqilvtabnquonnoylprh:tudhak-0roCto-poxcef@aws-0-eu-central-1.pooler.supabase.com:6543/postgres?sslmode=require"
    
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
            "--status", params["severity"],
            "--lon", str(params["lon"]),
            "--lat", str(params["lat"]),
            "--notes", params["notes"],
            "--crack-type", params.get("crack_type", "Radial Floor Crack"),
            "--crack-width", params.get("crack_width", "0.0mm"),
            "--uncertainty", params.get("uncertainty", "±0.02mm"),
            "--confidence", params.get("confidence", "92.4%"),
            "--priority", params.get("priority", "High"),
            "--maintenance-action", params.get("maintenance_action", "Immediate structural evaluation"),
            "--spalling-area", params.get("spalling_area", "0.0 mm²")
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
    db_conn_str = "postgresql://postgres.iqilvtabnquonnoylprh:tudhak-0roCto-poxcef@aws-0-eu-central-1.pooler.supabase.com:6543/postgres?sslmode=require"
    try:
        import psycopg2
        with psycopg2.connect(db_conn_str) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT ticket_id, inspector_name, status, ST_AsText(geom) FROM inspection_logs WHERE ticket_id = 'TK-CV-01';")
                row = cursor.fetchone()
                if row:
                    print(f"Database Query Output: {row}")
                    if "-75.25 45.5" in row[3]:
                        print(" -> [VERIFIED] Live coordinates (-75.25, 45.5) successfully written to database record!")
                        return True
                    else:
                        print(" -> [FAIL] DB Record coordinates do not match expected (-75.25, 45.5)!")
                        return False
                else:
                    print(" -> [FAIL] No record found in Supabase for ticket 'TK-CV-01'!")
                    return False
    except Exception as e:
        print(f" -> [FAIL] Database verification query failed: {e}", file=sys.stderr)
        return False

def main():
    print("==================================================")
    print("Live End-to-End Agentic Integration Test")
    print("==================================================\n")
    
    # Pre-test Database Cleanup: Delete existing TK-CV-01 from Supabase
    db_conn_str = "postgresql://postgres.iqilvtabnquonnoylprh:tudhak-0roCto-poxcef@aws-0-eu-central-1.pooler.supabase.com:6543/postgres?sslmode=require"
    try:
        import psycopg2
        with psycopg2.connect(db_conn_str) as conn:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM inspection_logs WHERE ticket_id = 'TK-CV-01';")
                conn.commit()
                print(" -> [CLEANUP] Existing database record for 'TK-CV-01' deleted successfully.")
    except Exception as e:
        print(f" -> [CLEANUP] Database cleanup warning: {e}", file=sys.stderr)
        
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
