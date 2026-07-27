import os
import sys
import subprocess
import json
import re
from datetime import datetime
import math
import base64
import io
from PIL import Image
import streamlit as st

# ── CCv2 inline drawable-line component ─────────────────────────────────────
# Users click two points on the inspection image to define a calibration line.
# The JS sends { x1, y1, x2, y2, pixelLength } back via setStateValue.

_DRAW_LINE_HTML = """\
<canvas id="draw-canvas" style="cursor:crosshair;display:block;border-radius:6px;"></canvas>
<p id="hint" style="margin:6px 0 0 0;font-size:13px;color:var(--st-text-color,#ccc);font-family:sans-serif;">
  Click two points on the reference marker to define the calibration line.
</p>
"""

_DRAW_LINE_CSS = """\
:host { display:block; }
"""

_DRAW_LINE_JS = """\
export default function(component) {
  const { parentElement, data, setStateValue } = component;
  const canvas = parentElement.querySelector("#draw-canvas");
  const hint   = parentElement.querySelector("#hint");
  if (!canvas || !hint) return;

  const img = new Image();
  img.onload = () => {
    canvas.width  = data.canvasWidth;
    canvas.height = data.canvasHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0, data.canvasWidth, data.canvasHeight);

    // Restore previous points if any
    let clicks = data.clicks || [];
    if (clicks.length >= 2) drawLine(ctx, clicks, img, data);

    canvas.onclick = (e) => {
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      if (clicks.length < 2) {
        clicks.push({ x, y });
      } else {
        clicks = [{ x, y }];
      }

      // Redraw
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, data.canvasWidth, data.canvasHeight);

      // Draw dots
      for (const pt of clicks) {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 5, 0, Math.PI * 2);
        ctx.fillStyle = "#FF3333";
        ctx.fill();
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      if (clicks.length === 2) {
        drawLine(ctx, clicks, img, data);
        const dx = clicks[1].x - clicks[0].x;
        const dy = clicks[1].y - clicks[0].y;
        const pixelLength = Math.sqrt(dx * dx + dy * dy);
        hint.textContent = "Line set (" + pixelLength.toFixed(1) + " px on canvas). Recalibrate by clicking two new points.";
        setStateValue("line", {
          x1: clicks[0].x, y1: clicks[0].y,
          x2: clicks[1].x, y2: clicks[1].y,
          pixelLength: pixelLength
        });
      } else {
        hint.textContent = "Click the second point to complete the line.";
      }
    };
  };
  img.src = data.imageSrc;

  function drawLine(ctx, pts, img, data) {
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    ctx.lineTo(pts[1].x, pts[1].y);
    ctx.strokeStyle = "#FF3333";
    ctx.lineWidth = 2.5;
    ctx.setLineDash([6, 4]);
    ctx.stroke();
    ctx.setLineDash([]);
    for (const pt of pts) {
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 5, 0, Math.PI * 2);
      ctx.fillStyle = "#FF3333";
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }
}
"""

_draw_line_component = st.components.v2.component(
    "draw_calibration_line",
    html=_DRAW_LINE_HTML,
    css=_DRAW_LINE_CSS,
    js=_DRAW_LINE_JS,
)

# 1. Page Config for wide layout and theme styling
st.set_page_config(
    page_title="Geospatial Infrastructure Inspector Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
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
            
        # --- Image Pre-processing: Downsample to avoid OOM on Streamlit Cloud (1GB limit) ---
        MAX_DIMENSION = 1600
        img_preprocess = Image.open(temp_path)
        orig_w, orig_h = img_preprocess.size
        
        if max(orig_w, orig_h) > MAX_DIMENSION:
            # Preserve EXIF metadata (GPS coordinates are critical for the pipeline)
            exif_data = img_preprocess.info.get("exif", None)
            
            # Calculate new dimensions preserving aspect ratio
            if orig_w >= orig_h:
                new_w = MAX_DIMENSION
                new_h = int(orig_h * (MAX_DIMENSION / orig_w))
            else:
                new_h = MAX_DIMENSION
                new_w = int(orig_w * (MAX_DIMENSION / orig_h))
            
            img_preprocess = img_preprocess.resize((new_w, new_h), Image.LANCZOS)
            
            # Save back with EXIF preserved
            save_kwargs = {"format": "JPEG", "quality": 90}
            if exif_data:
                save_kwargs["exif"] = exif_data
            img_preprocess.save(temp_path, **save_kwargs)
            
            st.caption(f"📐 Image downsampled: {orig_w}×{orig_h} → {new_w}×{new_h} px (max {MAX_DIMENSION}px to stay within memory limits)")
        
        img_preprocess.close()
        # --- End Pre-processing ---
            
        run_analysis = False
        computed_gsd = None
        
        if ref_marker_mm > 0.0:
            # Calibration mode active: show canvas
            st.markdown("### :material/straighten: Interactive scale calibration")
            st.markdown("Click **two points** on the reference marker to define a calibration line.")
            
            img = Image.open(temp_path)
            orig_w, orig_h = img.size
            
            display_width = 800
            scale_factor = display_width / orig_w
            display_height = int(orig_h * scale_factor)
            
            # Encode image as base64 data URL for the canvas background
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            image_src = f"data:image/jpeg;base64,{b64}"
            
            # Read any previously-set line from session state
            CANVAS_KEY = "calibration_canvas"
            canvas_state = st.session_state.get(CANVAS_KEY, {})
            prev_line = canvas_state.get("line", None)
            
            _draw_line_component(
                key=CANVAS_KEY,
                data={
                    "imageSrc": image_src,
                    "canvasWidth": display_width,
                    "canvasHeight": display_height,
                    "clicks": [],
                },
                height=display_height + 30,
                on_line_change=lambda: None,
            )
            
            # Read the line measurement back from session state
            canvas_state = st.session_state.get(CANVAS_KEY, {})
            line_data = canvas_state.get("line", None)
            
            if line_data is not None and line_data.get("pixelLength", 0) > 0:
                pixel_distance_canvas = line_data["pixelLength"]
                pixel_distance_orig = pixel_distance_canvas / scale_factor
                computed_gsd = ref_marker_mm / pixel_distance_orig
                st.success(f"✔ Calibration line set: {pixel_distance_canvas:.1f} px on canvas ({pixel_distance_orig:.1f} px original) = {ref_marker_mm} mm. Calculated GSD: {computed_gsd:.4f} mm/px.")
                run_analysis = True
            else:
                st.info("Awaiting calibration line. Click two points across the reference marker on the image above to proceed.")
        else:
            run_analysis = True
            
        if run_analysis:
            with st.spinner("Executing Deep Learning Instance Segmentation (YOLOv8-seg), OpenCV Skeletonization, Distance Transform, and Annotating Image..."):
                cmd = [sys.executable, ANALYZE_SCRIPT, "--image-path", temp_path, "--output-dir", temp_dir]
                if ref_marker_mm > 0.0:
                    cmd.extend(["--reference-marker-width-mm", str(ref_marker_mm)])
                if computed_gsd is not None:
                    cmd.extend(["--gsd", f"{computed_gsd:.6f}"])
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
                    st.image(uploaded_file, caption=f"Original Photograph: {uploaded_file.name}")
                    
                with col2:
                    annotated_path = json_data.get("annotated_path", os.path.join(temp_dir, "annotated_" + uploaded_file.name))
                    if os.path.exists(annotated_path):
                        st.image(annotated_path, caption="Annotated Image: Defect Bounding Box Overlay")
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
                    
                    m5, m6, m7, m8 = st.columns(4)
                    m5.metric("Scale Calibration", json_data.get("calibration_status", "Uncalibrated (Default GSD)"))
                    m6.metric("Action Priority", json_data.get("priority", "High"))
                    m7.metric("GPS Location (Lon, Lat)", f"{json_data.get('lon')}, {json_data.get('lat')}")
                    if json_data.get("spalling_area_mm2", 0.0) > 0.0:
                        m8.metric("Spalling Area", f"{json_data.get('spalling_area_mm2')} mm²")
                    else:
                        m8.metric("Spalling Area", "N/A")
                    
                    st.info(f"**Recommended Maintenance Action**: {json_data.get('maintenance_action')}")
                    
                    st.divider()
                    st.subheader("Human-in-the-Loop (HITL) Verification Gate")
                    st.write("Review the automated payload above. Choose to **Approve & Generate Ticket** to commit to PostGIS database and compile PDF work order, or **Reject / Manual Override** to supply custom human inspector inputs.")
                    
                    ticket_id = f"TK-INT-{uploaded_file.name.split('.')[0][:10]}-{datetime.now().strftime('%y%m%d-%H%M')}"
                    lon = json_data.get("lon")
                    lat = json_data.get("lat")
                    status = json_data.get("status")
                    width = json_data.get("crack_width_mm")
                    severity = json_data.get("severity")
                    spalling_area = json_data.get("spalling_area_mm2")
                    crack_type = json_data.get("crack_type", "Radial Floor Crack")
                    maintenance_action = json_data.get("maintenance_action")
                    uncertainty = json_data.get("uncertainty_mm")
                    
                    if "Spalling" in crack_type and "Crack" not in crack_type:
                        notes = f"Spalling Area: {spalling_area} mm². Action: {maintenance_action}"
                    elif "Crack" in crack_type and "Spalling" not in crack_type:
                        notes = f"COD Width: {width}mm ±{uncertainty}mm. Action: {maintenance_action}"
                    else:
                        notes = f"COD Width: {width}mm ±{uncertainty}mm. Spalling Area: {spalling_area} mm². Action: {maintenance_action}"
                    
                    env = load_env_vars()
                    
                    hitl_col1, hitl_col2 = st.columns(2)
                    
                    with hitl_col1:
                        # Disable approve button if low confidence
                        disable_approve = (confidence < 75.0)
                        if st.button("✔ Approve & Generate Ticket", disabled=disable_approve, type="primary", use_container_width=True):
                            # 1. Log DB
                            with st.spinner("Logging to PostGIS database..."):
                                db_cmd = [
                                    sys.executable, LOG_SCRIPT,
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
                                err_msg = db_res.stderr.strip()
                                if "could not translate host name" in err_msg or "connection timeout" in err_msg or "timeout" in err_msg or "could not connect to server" in err_msg or "nodename nor servname provided" in err_msg:
                                    st.error("⚠️ **Database Connection Failed**: Could not connect to Supabase Cloud PostgreSQL. The host may be unreachable or the connection timed out. Please verify that the database is active (not paused) and check your internet connection.")
                                else:
                                    st.error(f"✘ Database insertion failed: {err_msg}")
                                
                            # 2. PDF Ticket
                            with st.spinner("Generating ReportLab PDF work order ticket..."):
                                pdf_path = os.path.join(WORKSPACE_DIR, f"ticket_{ticket_id}.pdf")
                                pdf_cmd = [
                                    sys.executable, TICKET_SCRIPT,
                                    "--output-path", pdf_path,
                                    "--ticket-id", ticket_id,
                                    "--inspector-name", "Streamlit Civil Intake",
                                    "--status", severity,
                                    "--lon", str(lon),
                                    "--lat", str(lat),
                                    "--notes", notes,
                                    "--crack-type", crack_type,
                                    "--crack-width", f"{width}mm",
                                    "--uncertainty", f"±{uncertainty}mm",
                                    "--confidence", f"{confidence}%",
                                    "--priority", json_data.get("priority", "High"),
                                    "--maintenance-action", maintenance_action,
                                    "--spalling-area", f"{spalling_area} mm²"
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
                                        sys.executable, LOG_SCRIPT,
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
                                    err_msg = db_res.stderr.strip()
                                    if "could not translate host name" in err_msg or "connection timeout" in err_msg or "timeout" in err_msg or "could not connect to server" in err_msg or "nodename nor servname provided" in err_msg:
                                        st.error("⚠️ **Database Connection Failed**: Could not connect to Supabase Cloud PostgreSQL. The host may be unreachable or the connection timed out. Please verify that the database is active (not paused) and check your internet connection.")
                                    else:
                                        st.error(f"✘ Failed to log manual override record: {err_msg}")
                else:
                    st.code(stdout_str)

with tab2:
    st.header("Geospatial Inspection Map Dashboard")
    st.write("Visual distribution of structural inspections queried from the PostGIS database. Marker icons denote status severity (Red = Severe defect, Green = Completed, Blue = Log only).")
    
    env = load_env_vars()
    
    if not os.path.exists(MAP_FILE):
        with st.spinner("Map file not found. Generating initial inspections map..."):
            res = subprocess.run([sys.executable, MAP_SCRIPT], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
            if res.returncode != 0:
                st.error("⚠️ **Database Connection Failed**: Map generation failed because the Supabase Cloud PostgreSQL database is unreachable.")
            
    if st.button("Refresh Map from Database"):
        with st.spinner("Updating map markers with latest records..."):
            res = subprocess.run([sys.executable, MAP_SCRIPT], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
            if res.returncode == 0:
                st.success("Map refreshed successfully!")
            else:
                st.error("⚠️ **Database Connection Failed**: Map refresh failed because the Supabase Cloud PostgreSQL database is unreachable.")
    if os.path.exists(MAP_FILE):
        with open(MAP_FILE, 'r') as f:
            html_data = f.read()
        st.html(f'<div style="height:650px;overflow:auto;">{html_data}</div>', unsafe_allow_javascript=True)
    else:
        st.warning("Mapping visualization could not be generated. Please check your database settings.")
