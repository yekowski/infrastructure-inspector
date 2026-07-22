-- Enable PostGIS extension
CREATE EXTENSION IF NOT EXISTS postgis;

-- Create inspection logs table
CREATE TABLE IF NOT EXISTS inspection_logs (
    id SERIAL PRIMARY KEY,
    ticket_id VARCHAR(50) UNIQUE NOT NULL,
    inspector_name VARCHAR(100) NOT NULL,
    status VARCHAR(20) NOT NULL,
    inspection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    geom GEOMETRY(Point, 4326),
    technician_notes TEXT
);

-- Create spatial index for optimized geographic queries
CREATE INDEX IF NOT EXISTS idx_inspection_logs_geom ON inspection_logs USING GIST(geom);
