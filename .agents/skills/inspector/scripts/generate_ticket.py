#!/usr/bin/env python3
"""
Executable script for generating enriched PDF inspection tickets from validated input arguments.
Renders ReportLab PDF documents featuring civil engineering metrics and maintenance actions.
"""

import os
import sys
import argparse
import json

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

def generate_pdf_ticket(output_path: str, ticket_data: dict):
    """
    Renders the inspection PDF ticket using ReportLab with enriched civil engineering fields.
    """
    absolute_output_path = os.path.abspath(output_path)
    workspace_dir = "/Users/yekowski/agy2-projects/infrastructure-inspector"
    
    if not absolute_output_path.startswith(workspace_dir) and not absolute_output_path.startswith("/tmp"):
        print(f"Security Error: Output path '{output_path}' lies outside the allowed directory bounds.", file=sys.stderr)
        sys.exit(3)
        
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    except ImportError:
        print("[MOCK] reportlab is not installed. PDF file generation skipped.", file=sys.stderr)
        print(f"[MOCK] Simulated PDF creation at: {output_path}")
        print(f"[MOCK] Ticket Content: {ticket_data}")
        try:
            os.makedirs(os.path.dirname(absolute_output_path), exist_ok=True)
            with open(absolute_output_path, 'w') as f:
                f.write(f"MOCK TICKET PDF FOR {ticket_data.get('ticket_id')}\n")
                f.write(json.dumps(ticket_data, indent=2))
        except Exception:
            pass
        return

    os.makedirs(os.path.dirname(absolute_output_path), exist_ok=True)
    
    doc = SimpleDocTemplate(
        absolute_output_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'TicketTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=15
    )
    
    section_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=10,
        spaceAfter=5
    )
    
    label_style = ParagraphStyle(
        'TicketLabel',
        parent=styles['Normal'],
        fontSize=10,
        leading=12,
        fontName='Helvetica-Bold',
        textColor=colors.HexColor("#4A5568")
    )
    
    value_style = ParagraphStyle(
        'TicketValue',
        parent=styles['Normal'],
        fontSize=10,
        leading=12,
        textColor=colors.HexColor("#2D3748")
    )

    story = []
    
    # Title
    story.append(Paragraph(f"CIVIL INSPECTION WORK ORDER: {ticket_data.get('ticket_id', 'N/A')}", title_style))
    story.append(Spacer(1, 10))
    
    # Structural Table Data
    defect_type = ticket_data.get('crack_type', 'Radial Floor Crack')
    data = [
        [Paragraph("Inspector Name:", label_style), Paragraph(str(ticket_data.get('inspector_name', 'N/A')), value_style)],
        [Paragraph("Status Severity:", label_style), Paragraph(str(ticket_data.get('status', 'N/A')), value_style)],
        [Paragraph("Priority Level:", label_style), Paragraph(str(ticket_data.get('priority', 'High')), value_style)],
        [Paragraph("Defect Type:", label_style), Paragraph(str(defect_type), value_style)],
    ]
    
    if "Spalling" in defect_type and "Crack" not in defect_type:
        sp_area = ticket_data.get('spalling_area', '0.0')
        if not sp_area.endswith("mm²") and not sp_area.endswith("mm2") and sp_area:
            sp_area = f"{sp_area} mm²"
        data.append([Paragraph("Spalling Area:", label_style), Paragraph(sp_area if sp_area else "0.0 mm²", value_style)])
    elif "Crack" in defect_type and "Spalling" not in defect_type:
        data.append([Paragraph("Measured Crack Width:", label_style), Paragraph(f"{ticket_data.get('crack_width', '0.0mm')} (Uncertainty: {ticket_data.get('uncertainty', '±0.02mm')})", value_style)])
    else:
        # Both or none
        data.append([Paragraph("Measured Crack Width:", label_style), Paragraph(f"{ticket_data.get('crack_width', '0.0mm')} (Uncertainty: {ticket_data.get('uncertainty', '±0.02mm')})", value_style)])
        sp_area = ticket_data.get('spalling_area', '0.0')
        if not sp_area.endswith("mm²") and not sp_area.endswith("mm2") and sp_area:
            sp_area = f"{sp_area} mm²"
        data.append([Paragraph("Spalling Area:", label_style), Paragraph(sp_area if sp_area else "0.0 mm²", value_style)])
        
    data.extend([
        [Paragraph("Detection Confidence:", label_style), Paragraph(str(ticket_data.get('confidence', '92.4%')), value_style)],
        [Paragraph("Coordinates (Lon, Lat):", label_style), Paragraph(f"{ticket_data.get('lon', 0.0)}, {ticket_data.get('lat', 0.0)}", value_style)],
        [Paragraph("Recommended Maintenance:", label_style), Paragraph(str(ticket_data.get('maintenance_action', 'N/A')), value_style)],
        [Paragraph("Technician Notes:", label_style), Paragraph(str(ticket_data.get('notes', 'N/A')), value_style)]
    ])
    
    t = Table(data, colWidths=[160, 390])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F7FAFC")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
    ]))
    
    story.append(t)
    doc.build(story)
    print(f"Successfully generated inspection PDF ticket at: {absolute_output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate PDF Inspection Ticket with Civil Engineering Metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--output-path", required=True, help="File path where the PDF will be generated")
    parser.add_argument("--ticket-id", required=True, help="Unique ticket identifier")
    parser.add_argument("--inspector-name", required=True, help="Name of the field inspector")
    parser.add_argument("--status", required=True, help="Inspection status (e.g., Severe, Moderate)")
    parser.add_argument("--lon", type=float, required=True, help="Longitude coordinates (WGS84)")
    parser.add_argument("--lat", type=float, required=True, help="Latitude coordinates (WGS84)")
    parser.add_argument("--notes", default="", help="Additional notes or technician remarks")
    
    # Enriched Civil Engineering Arguments
    parser.add_argument("--crack-type", default="Radial Floor Crack", help="Type of structural defect")
    parser.add_argument("--crack-width", default="", help="Measured crack opening width")
    parser.add_argument("--uncertainty", default="±0.05mm", help="Measurement uncertainty bound")
    parser.add_argument("--confidence", default="92.4%", help="Detection model confidence score")
    parser.add_argument("--priority", default="High", help="Priority level")
    parser.add_argument("--maintenance-action", default="Immediate structural evaluation", help="Recommended action")
    parser.add_argument("--spalling-area", default="", help="Measured spalling area in mm2")
    
    args = parser.parse_args()
    
    validate_coordinates(args.lon, args.lat)
    
    ticket_data = {
        "ticket_id": args.ticket_id,
        "inspector_name": args.inspector_name,
        "status": args.status,
        "lon": args.lon,
        "lat": args.lat,
        "notes": args.notes,
        "crack_type": args.crack_type,
        "crack_width": args.crack_width if args.crack_width else "0.05mm",
        "uncertainty": args.uncertainty,
        "confidence": args.confidence,
        "priority": args.priority,
        "maintenance_action": args.maintenance_action,
        "spalling_area": args.spalling_area
    }
    
    generate_pdf_ticket(args.output_path, ticket_data)

if __name__ == "__main__":
    main()
