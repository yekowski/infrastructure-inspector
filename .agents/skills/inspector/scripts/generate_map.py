#!/usr/bin/env python3
"""
Executable script for querying PostGIS inspection logs and generating an interactive HTML map.
Uses folium to plot locations with color-coded status markers.
"""

import os
import sys
import folium

def get_db_connection():
    """
    Establishes connection to PostGIS database using env variables.
    """
    db_conn_str = os.environ.get("DB_CONN_STR")
    if not db_conn_str:
        host = os.environ.get("DB_HOST")
        user = os.environ.get("DB_USER")
        password = os.environ.get("DB_PASSWORD")
        dbname = os.environ.get("DB_NAME")
        if all([host, user, password, dbname]):
            db_conn_str = f"postgresql://{user}:{password}@{host}/{dbname}"
            
    if not db_conn_str:
        return None
        
    try:
        import psycopg2
        return psycopg2.connect(db_conn_str)
    except Exception as e:
        print(f"[WARNING] Database connection failed: {e}", file=sys.stderr)
        return None

def query_inspections():
    """
    Queries inspection records from the database, extracting coordinates.
    """
    has_config = os.environ.get("DB_CONN_STR") or (
        os.environ.get("DB_HOST") and 
        os.environ.get("DB_USER") and 
        os.environ.get("DB_PASSWORD") and 
        os.environ.get("DB_NAME")
    )
    
    conn = get_db_connection()
    if not conn:
        if has_config and os.environ.get("DB_CONN_STR") != "mock_connection":
            print("Error: Database connection failed while trying to refresh map.", file=sys.stderr)
            sys.exit(3)
        print("[WARNING] Running in offline/mock database mode.", file=sys.stderr)
        return []
        
    query = """
        SELECT ticket_id, inspector_name, status, technician_notes, ST_X(geom) as lon, ST_Y(geom) as lat
        FROM inspection_logs;
    """
    try:
        with conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                columns = [col[0] for col in cursor.description]
                rows = cursor.fetchall()
                return [dict(zip(columns, row)) for row in rows]
    except Exception as e:
        print(f"Error querying database: {e}", file=sys.stderr)
        return []
    finally:
        conn.close()

def generate_html_map(records, output_path):
    """
    Generates Folium map and adds color-coded markers.
    """
    default_lat, default_lon = 37.7749, -122.4194
    
    # Calculate map center based on average coordinates if records exist
    if records:
        valid_coords = [(r["lat"], r["lon"]) for r in records if r["lat"] is not None and r["lon"] is not None]
        if valid_coords:
            center_lat = sum(c[0] for c in valid_coords) / len(valid_coords)
            center_lon = sum(c[1] for c in valid_coords) / len(valid_coords)
        else:
            center_lat, center_lon = default_lat, default_lon
    else:
        center_lat, center_lon = default_lat, default_lon
        
    # Initialize Folium Map (Note: Coordinates are passed as [latitude, longitude])
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="OpenStreetMap")
    
    if not records:
        # Add a default placeholder marker if the DB is empty
        folium.Marker(
            location=[default_lat, default_lon],
            popup="No active inspections found. System active.",
            tooltip="System Center",
            icon=folium.Icon(color="gray", icon="info-sign")
        ).add_to(m)
        print("Database is empty. Created a map with a system placeholder marker.")
    else:
        for r in records:
            lat, lon = r["lat"], r["lon"]
            if lat is None or lon is None:
                continue
                
            status = r.get("status", "Unknown")
            ticket_id = r.get("ticket_id", "N/A")
            notes = r.get("technician_notes", "")
            inspector = r.get("inspector_name", "")
            
            # Color-code based on severity status
            if status == "Requires PDF ticket":
                marker_color = "red"
                icon_type = "exclamation-sign"
            elif status == "Completed":
                marker_color = "green"
                icon_type = "ok-sign"
            else:
                marker_color = "blue"
                icon_type = "info-sign"
                
            # Construct popup HTML
            popup_html = f"""
                <div style="font-family: Arial, sans-serif; font-size: 12px; width: 220px;">
                    <h4 style="margin: 0 0 5px 0; color: #1A365D;">Ticket: {ticket_id}</h4>
                    <b>Inspector:</b> {inspector}<br/>
                    <b>Status:</b> <span style="color: {'red' if marker_color=='red' else 'green' if marker_color=='green' else 'blue'};">{status}</span><br/>
                    <b>Details/Notes:</b> {notes}<br/>
                    <b>Location:</b> {lon}, {lat}
                </div>
            """
            
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"Ticket: {ticket_id}",
                icon=folium.Icon(color=marker_color, icon=icon_type)
            ).add_to(m)
            
    m.save(output_path)
    print(f"Successfully generated interactive HTML map at: {output_path}")

def main():
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
    output_path = os.path.join(workspace_dir, "inspections_map.html")
    
    # 1. Fetch records
    records = query_inspections()
    
    # 2. Render Map
    generate_html_map(records, output_path)

if __name__ == "__main__":
    main()
