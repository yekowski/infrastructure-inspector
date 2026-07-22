#!/usr/bin/env python3
"""
Executable script for logging geospatial infrastructure inspections into PostGIS.
Enforces strict argument type validation using argparse and input sanitization.
"""

import os
import sys
import argparse

def validate_coordinates(lon: float, lat: float):
    """
    Validates coordinate bounds (WGS84 EPSG:4326).
    """
    if not (-180.0 <= lon <= 180.0):
        print(f"Error: Longitude {lon} out of bounds [-180.0, 180.0]", file=sys.stderr)
        sys.exit(1)
    if not (-90.0 <= lat <= 90.0):
        print(f"Error: Latitude {lat} out of bounds [-90.0, 90.0]", file=sys.stderr)
        sys.exit(1)

def log_to_database(db_conn_str: str, ticket_id: str, inspector: str, status: str, lon: float, lat: float, notes: str):
    """
    Logs the inspection into the PostGIS database.
    """
    try:
        import psycopg2
    except ImportError:
        # Graceful handling for environments without psycopg2 installed
        print("[MOCK] psycopg2 is not installed. Database connection skipped.", file=sys.stderr)
        print(f"[MOCK] Simulated SQL: INSERT INTO inspection_logs values ('{ticket_id}', '{inspector}', '{status}', Point({lon}, {lat}), '{notes}')")
        return 1

    query = """
        INSERT INTO inspection_logs (ticket_id, inspector_name, status, geom, technician_notes)
        VALUES (%s, %s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326), %s)
        RETURNING id;
    """
    
    try:
        with psycopg2.connect(db_conn_str) as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (ticket_id, inspector, status, lon, lat, notes))
                log_id = cursor.fetchone()[0]
                conn.commit()
                print(f"Successfully logged inspection. Log ID: {log_id}")
                return log_id
    except Exception as e:
        print(f"Database insertion failed: {e}", file=sys.stderr)
        sys.exit(2)

def main():
    parser = argparse.ArgumentParser(
        description="Log geospatial infrastructure inspection into PostGIS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--ticket-id", required=True, help="Unique ticket identifier")
    parser.add_argument("--inspector-name", required=True, help="Name of the field inspector")
    parser.add_argument("--status", required=True, help="Inspection status (e.g., Completed, Pending)")
    parser.add_argument("--lon", type=float, required=True, help="Longitude coordinates (WGS84)")
    parser.add_argument("--lat", type=float, required=True, help="Latitude coordinates (WGS84)")
    parser.add_argument("--notes", default="", help="Additional notes or technician remarks")
    
    args = parser.parse_args()
    
    # 1. Strict Coordinate Validation
    validate_coordinates(args.lon, args.lat)
    
    # 2. Credential Retrieval from Environment
    db_conn_str = os.environ.get("DB_CONN_STR")
    if not db_conn_str:
        # Fallback build from individual env vars if present
        host = os.environ.get("DB_HOST")
        user = os.environ.get("DB_USER")
        password = os.environ.get("DB_PASSWORD")
        dbname = os.environ.get("DB_NAME")
        if all([host, user, password, dbname]):
            db_conn_str = f"postgresql://{user}:{password}@{host}/{dbname}"
        else:
            print("Error: Database connection credentials (DB_CONN_STR or DB_HOST/DB_USER/...) must be provided in the environment.", file=sys.stderr)
            # For testing/scaffolding purposes, we log the parsed args and proceed with a mock connection warning
            db_conn_str = "mock_connection"

    # 3. Log to DB
    log_to_database(
        db_conn_str=db_conn_str,
        ticket_id=args.ticket_id,
        inspector=args.inspector_name,
        status=args.status,
        lon=args.lon,
        lat=args.lat,
        notes=args.notes
    )

if __name__ == "__main__":
    main()
