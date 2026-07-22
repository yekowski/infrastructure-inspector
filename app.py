import os
import sys
import subprocess
import json
import re
import streamlit as st
import streamlit.components.v1 as components

# 1. Page Config for wide layout and theme styling
st.set_page_config(
    page_title="Geospatial Infrastructure Inspector Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

WORKSPACE_DIR = "/Users/yekowski/agy2-projects/infrastructure-inspector"
ANALYZE_SCRIPT = os.path.join(WORKSPACE_DIR, ".agents/skills/inspector/scripts/analyze_image.py")
LOG_SCRIPT = os.path.join(WORKSPACE_DIR, ".agents/skills/inspector/scripts/log_inspection.py")
TICKET_SCRIPT = os.path.join(WORKSPACE_DIR, ".agents/skills/inspector/scripts/generate_ticket.py")
MAP_SCRIPT = os.path.join(WORKSPACE_DIR, ".agents/skills/inspector/scripts/generate_map.py")
MAP_FILE = os.path.join(WORKSPACE_DIR, "inspections_map.html")
ENV_FILE = os.path.join(WORKSPACE_DIR, ".env")

def load_env_vars():
    """
    Loads environment variables from local .env file.
    """
    env = os.environ.copy()
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, val = line.strip().split('=', 1)
                    env[key.strip()] = val.strip().strip('"').strip("'")
    return env

# Expose page header
st.markdown("""
    <div style="background-color:#1A365D;padding:15px;border-radius:10px;margin-bottom:20px;">
        <h1 style="color:white;margin:0;font-family:Arial,sans-serif;">Geospatial Infrastructure Inspector</h1>
        <p style="color:#E2E8F0;margin:5px 0 0 0;">Human-in-the-Loop (HITL) Workflow, Quality Control, and PostGIS Mapping</p>
    </div>
""", unsafe_allow_html=True)

# Define Tabs
tab1, tab2 = st.tabs(["📸 Inspection Intake (HITL)", "🗺️ Geospatial Dashboard"])

with tab1:
    st.header("Civil Engineering Inspection Photo Analysis & HITL Verification")
    st.write("Upload structural photographs of concrete or masonry to compute sub-millimeter Crack Opening Displacement (COD), uncertainty bounds, EXIF GPS coordinates, and overlay annotations.")
    
    # Optional calibration input
    ref_marker_mm = st.number_input("Optional Physical Reference Marker Width (mm)", min_value=0.0, max_value=200.0, value=0.0, step=1.0, help="Specify if a reference calibration target is present in frame.")
    
    uploaded_file = st.file_uploader("Upload Inspection Photograph", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Save temporary image file
        temp_dir = "/tmp/inspector_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        with st.spinner("Executing Deep Learning Instance Segmentation (YOLOv8-seg), OpenCV Skeletonization, Distance Transform, and Annotating Image..."):
            cmd = ["python3", ANALYZE_SCRIPT, "--image-path", temp_path, "--output-dir", temp_dir]
            if ref_marker_mm > 0.0:
                cmd.extend(["--reference-marker-width-mm", str(ref_marker_mm)])
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
        if res.returncode != 0:
            st.error("Analysis Script failed to run!")
            st.code(res.stderr)
        else:
            stdout_str = res.stdout.strip()
            
            # Extract JSON Payload line
            json_data = {}
            for line in stdout_str.splitlines():
                if line.startswith("JSON PAYLOAD:"):
                    try:
                        json_data = json.loads(line.replace("JSON PAYLOAD:", "").strip())
                    except Exception as e:
                        st.warning(f"Could not parse JSON payload: {e}")
                        
            # Side-by-side image display
            st.subheader("Visual Inspection Comparison")
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(uploaded_file, caption=f"Original Photograph: {uploaded_file.name}", use_container_width=True)
                
            with col2:
                annotated_path = json_data.get("annotated_path", os.path.join(temp_dir, "annotated_" + uploaded_file.name))
                if os.path.exists(annotated_path):
                    st.image(annotated_path, caption="Annotated Image: Defect Bounding Box Overlay", use_container_width=True)
                else:
                    st.warning("Annotated image file not found.")
                    
            st.divider()
            st.subheader("Civil Engineering Structural Metrics & Quality Control")
            
            if json_data:
                confidence = float(json_data.get("confidence_pct", 0.0))
                
                # Check Quality Control Gate 1: Confidence Thresholding (< 75%)
                if confidence < 75.0:
                    st.error(f"⚠️ Low Confidence Detection ({confidence}% < 75.0%). Automated workflow HALTED! Status set to: Requires Manual Review.")
                    json_data["severity"] = "Requires Manual Review"
                    json_data["status"] = "Requires Manual Review"
                else:
                    st.success(f"✔ Quality Control Passed: High Confidence Detection ({confidence}% >= 75.0%).")
                    
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Defect Type", json_data.get("crack_type", "Radial Floor Crack"))
                m2.metric("Estimated Width", f"{json_data.get('crack_width_mm')} mm", f"±{json_data.get('uncertainty_mm')} mm")
                m3.metric("Detection Confidence", f"{confidence}%")
                m4.metric("Severity Level", json_data.get("severity"), delta_color="inverse" if json_data.get("severity")=="Severe" else "normal")
                
                m5, m6, m7 = st.columns(3)
                m5.metric("Scale Calibration", json_data.get("calibration_status", "Uncalibrated (Default GSD)"))
                m6.metric("Action Priority", json_data.get("priority", "High"))
                m7.metric("GPS Location (Lon, Lat)", f"{json_data.get('lon')}, {json_data.get('lat')}")
                
                st.info(f"**Recommended Maintenance Action**: {json_data.get('maintenance_action')}")
                
                st.divider()
                st.subheader("Human-in-the-Loop (HITL) Verification Gate")
                st.write("Review the automated payload above. Choose to **Approve & Generate Ticket** to commit to PostGIS database and compile PDF work order, or **Reject / Manual Override** to supply custom human inspector inputs.")
                
                ticket_id = f"TK-INT-{uploaded_file.name.split('.')[0][:10]}"
                lon = json_data.get("lon")
                lat = json_data.get("lat")
                status = json_data.get("status")
                width = json_data.get("crack_width_mm")
                severity = json_data.get("severity")
                notes = f"COD Width: {width}mm ±{json_data.get('uncertainty_mm')}mm. Action: {json_data.get('maintenance_action')}"
                
                env = load_env_vars()
                
                hitl_col1, hitl_col2 = st.columns(2)
                
                with hitl_col1:
                    # Disable approve button if low confidence
                    disable_approve = (confidence < 75.0)
                    if st.button("✔ Approve & Generate Ticket", disabled=disable_approve, type="primary", use_container_width=True):
                        # 1. Log DB
                        with st.spinner("Logging to PostGIS database..."):
                            db_cmd = [
                                "python3", LOG_SCRIPT,
                                "--ticket-id", ticket_id,
                                "--inspector-name", "Streamlit Civil Intake",
                                "--status", status,
                                "--lon", str(lon),
                                "--lat", str(lat),
                                "--notes", notes
                            ]
                            db_res = subprocess.run(db_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
                            
                        if db_res.returncode == 0:
                            st.success(f"✔ Database insertion complete! {db_res.stdout.strip()}")
                        else:
                            st.error(f"✘ Database insertion failed! {db_res.stderr.strip()}")
                            
                        # 2. PDF Ticket
                        with st.spinner("Generating ReportLab PDF work order ticket..."):
                            pdf_path = os.path.join(WORKSPACE_DIR, f"ticket_{ticket_id}.pdf")
                            pdf_cmd = [
                                "python3", TICKET_SCRIPT,
                                "--output-path", pdf_path,
                                "--ticket-id", ticket_id,
                                "--inspector-name", "Streamlit Civil Intake",
                                "--status", status,
                                "--lon", str(lon),
                                "--lat", str(lat),
                                "--notes", notes,
                                "--crack-type", json_data.get("crack_type", "Radial Floor Crack"),
                                "--crack-width", f"{width}mm",
                                "--uncertainty", f"±{json_data.get('uncertainty_mm')}mm",
                                "--confidence", f"{confidence}%",
                                "--priority", json_data.get("priority", "High"),
                                "--maintenance-action", json_data.get("maintenance_action", "Structural Evaluation")
                            ]
                            pdf_res = subprocess.run(pdf_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
                            
                        if pdf_res.returncode == 0:
                            st.success(f"✔ PDF ticket successfully compiled! File path: `{pdf_path}`")
                            if os.path.exists(pdf_path):
                                with open(pdf_path, "rb") as pdf_file:
                                    st.download_button(
                                        label="Download PDF Ticket File",
                                        data=pdf_file.read(),
                                        file_name=f"ticket_{ticket_id}.pdf",
                                        mime="application/pdf"
                                    )
                        else:
                            st.error(f"✘ PDF generation failed! {pdf_res.stderr.strip()}")
                            
                with hitl_col2:
                    if st.button("✖ Reject / Manual Override", use_container_width=True):
                        st.session_state["show_override_form"] = True
                        
                if st.session_state.get("show_override_form", False):
                    st.warning("Manual Override Mode Active. Enter human inspector values below:")
                    with st.form("manual_override_form"):
                        override_inspector = st.text_input("Human Inspector Name", value="Inspector Jane Doe")
                        override_width = st.number_input("Override Crack Width (mm)", min_value=0.01, max_value=50.0, value=float(width) if width else 0.5)
                        override_status = st.selectbox("Override Status", ["Completed", "Requires PDF ticket", "Requires Manual Review", "Log only"])
                        override_notes = st.text_area("Human Technician Notes", value="Manual override applied after visual verification.")
                        
                        submit_override = st.form_submit_button("Submit Manual Override Log")
                        
                        if submit_override:
                            with st.spinner("Logging manual override record to PostGIS database..."):
                                db_cmd = [
                                    "python3", LOG_SCRIPT,
                                    "--ticket-id", f"{ticket_id}-OVR",
                                    "--inspector-name", override_inspector,
                                    "--status", override_status,
                                    "--lon", str(lon),
                                    "--lat", str(lat),
                                    "--notes", f"{override_notes} (overridden: true, width: {override_width}mm)"
                                ]
                                db_res = subprocess.run(db_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
                                
                            if db_res.returncode == 0:
                                st.success(f"✔ Manual override record successfully logged to PostGIS! (overridden: true)")
                                st.session_state["show_override_form"] = False
                            else:
                                st.error(f"✘ Failed to log manual override record: {db_res.stderr.strip()}")
            else:
                st.code(stdout_str)

with tab2:
    st.header("Geospatial Inspection Map Dashboard")
    st.write("Visual distribution of structural inspections queried from the PostGIS database. Marker icons denote status severity (Red = Severe defect, Green = Completed, Blue = Log only).")
    
    env = load_env_vars()
    
    if not os.path.exists(MAP_FILE):
        with st.spinner("Map file not found. Generating initial inspections map..."):
            subprocess.run(["python3", MAP_SCRIPT], env=env)
            
    if st.button("Refresh Map from Database"):
        with st.spinner("Updating map markers with latest records..."):
            subprocess.run(["python3", MAP_SCRIPT], env=env)
            st.success("Map refreshed successfully!")
            
    if os.path.exists(MAP_FILE):
        with open(MAP_FILE, 'r') as f:
            html_data = f.read()
        components.html(html_data, height=650, scrolling=True)
    else:
        st.warning("Mapping visualization could not be generated. Please check your database settings.")
