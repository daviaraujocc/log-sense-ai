from log_sense import SeverityLevel, format_logs_with_id
from rich.console import Console
from rich.table import Table as RichTable
from rich.panel import Panel
from rich.text import Text
from rich import box
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import os
from datetime import datetime

def generate_report(analysis_result_data, logs=None, output_path=None, filename="log_analysis_report.pdf", 
                   severity_levels=None):
    """Generate a professional PDF report from the analysis results.
    
    Args:
        analysis_result_data: AnalysisResults object containing analysis results and metadata
        logs: Original log entries for displaying in the report
        output_path: Directory path for saving the report (defaults to current directory if None)
        filename: Name of the PDF file (defaults to "log_analysis_report.pdf")
        severity_levels: List of severity levels to include in the report 
                         (default: [CRITICAL, ERROR, WARNING], INFO is excluded by default)
    
    Returns:
        str: Path to the generated PDF file
    """
    # Extract data from the results object
    analysis_results = analysis_result_data.results
    source_filename = analysis_result_data.source_filename
    log_type = analysis_result_data.log_type
    
    # Determine full output path
    if output_path is None:
        output_path = "."
    
    # Set default severity levels if none provided - explicitly exclude INFO
    if severity_levels is None:
        severity_levels = [SeverityLevel.CRITICAL, SeverityLevel.ERROR, SeverityLevel.WARNING]
    
    # No need for additional validation - we'll use severity_levels directly
    
    os.makedirs(output_path, exist_ok=True)
    full_path = os.path.join(output_path, filename)
    
    # Create document
    doc = SimpleDocTemplate(full_path, pagesize=letter)
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    subtitle_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Custom styles
    header_style = ParagraphStyle(
        'HeaderStyle',
        parent=styles['Heading3'],
        textColor=colors.darkblue,
        spaceAfter=12
    )
    
    event_header_style = ParagraphStyle(
        'EventHeaderStyle',
        parent=styles['Heading4'],
        textColor=colors.darkslategray,
        spaceBefore=10,
        spaceAfter=6
    )
    
    # Create document elements
    elements = []
    
    # Title
    elements.append(Paragraph(f"Log Analysis Report - {log_type}", title_style))
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", subtitle_style))
    
    # Add source filename if provided
    if source_filename:
        elements.append(Paragraph(f"Source: {source_filename}", subtitle_style))
    
    elements.append(Spacer(1, 0.25 * inch))
    
    # Track if any analyses were included in the report
    any_analysis_included = False
    
    # Process each analysis result
    for i, analysis in enumerate(analysis_results):
        if not analysis:
            continue
            
        # Skip this entire analysis if its highest severity is not in the allowed levels
        if analysis.highest_severity and analysis.highest_severity not in severity_levels:
            continue
            
        any_analysis_included = True
        
        # Use line range info if available, otherwise fall back to batch number
        if hasattr(analysis, 'start_line') and hasattr(analysis, 'end_line') and analysis.start_line is not None and analysis.end_line is not None:
            section_title = f"Lines {analysis.start_line}-{analysis.end_line} Analysis"
        else:
            section_title = f"Batch {i+1} Analysis"
            
        elements.append(Paragraph(section_title, header_style))
        
        # Only show highest severity if it's in the allowed levels
        if analysis.highest_severity and analysis.highest_severity in severity_levels:
            elements.append(Paragraph(f"Highest Severity: {analysis.highest_severity}", normal_style))
        
        elements.append(Paragraph(f"Requires Immediate Attention: {'Yes' if analysis.requires_immediate_attention else 'No'}", normal_style))
        
        # Add observations
        if analysis.observations:
            elements.append(Paragraph("Key Observations:", event_header_style))
            for obs in analysis.observations:
                elements.append(Paragraph(f"• {obs}", normal_style))
        
        # Add events (filtered by severity)
        if analysis.events:
            elements.append(Paragraph("Detected Events:", event_header_style))
            for event in analysis.events:
                # Simple check if this event's severity should be included
                if event.severity not in severity_levels:
                    continue
                
                # Create event section with colored severity indicator
                severity_color = colors.red if event.severity == SeverityLevel.CRITICAL else \
                                colors.orange if event.severity == SeverityLevel.ERROR else \
                                colors.gold if event.severity == SeverityLevel.WARNING else colors.blue
                
                event_style = ParagraphStyle(
                    f'Event{event.severity}Style',
                    parent=normal_style,
                    leftIndent=20,
                    spaceBefore=8
                )
                
                # Add event details
                elements.append(Paragraph(f"<font color='{severity_color.hexval()}'>{event.severity.upper()}</font> - {event.category.capitalize()} Event", event_header_style))
                elements.append(Paragraph(f"<b>Reasoning:</b> {event.reasoning}", event_style))
                elements.append(Paragraph(f"<b>Recommended Action:</b> {event.recommended_action}", event_style))
                
                # Add relevant log IDs
                log_id_text = "Related Log Entries: " + ", ".join([log.log_id for log in event.relevant_log_ids])
                elements.append(Paragraph(log_id_text, event_style))
                
                # Display the actual log entries if logs were provided
                if logs:
                    # Style for log content display
                    log_style = ParagraphStyle(
                        'LogStyle',
                        parent=normal_style,
                        leftIndent=25,
                        fontName='Courier',
                        fontSize=8,
                        backgroundColor=colors.lightgrey,
                        borderWidth=1,
                        borderColor=colors.grey,
                        borderPadding=5,
                        spaceBefore=5,
                        spaceAfter=5
                    )
                    
                    elements.append(Paragraph("<b>Log Content:</b>", event_style))
                    
                    # Format logs with IDs for searching
                    formatted_logs = format_logs_with_id(logs)
                    
                    # Collect all relevant logs first
                    relevant_logs = []
                    for log_id in event.relevant_log_ids:
                        log = log_id.find_in(formatted_logs)
                        if log:
                            relevant_logs.append(log)
                    
                    # If we found any logs, display them in a single paragraph
                    if relevant_logs:
                        combined_logs = "<br/>".join(relevant_logs)
                        elements.append(Paragraph(combined_logs, log_style))
                
                elements.append(Spacer(1, 0.1 * inch))
        
        elements.append(Spacer(1, 0.2 * inch))
    
    # If no analyses were included, add a "no issues found" message
    if not any_analysis_included:
        no_issues_style = ParagraphStyle(
            'NoIssuesStyle',
            parent=normal_style,
            alignment=1,  # Center alignment
            spaceBefore=36,
            spaceAfter=36,
            fontSize=14
        )
        elements.append(Paragraph("No issues matching the selected severity levels were found.", no_issues_style))
        elements.append(Spacer(1, 0.5 * inch))
    
    # Build PDF
    doc.build(elements)
    return full_path


def generate_console_report(analysis_result_data, logs=None, severity_levels=None):
    """Generate a colorful console report from the analysis results using rich library.
    
    Args:
        analysis_result_data: AnalysisResults object containing analysis results and metadata
        logs: Original log entries for displaying in the report
        severity_levels: List of severity levels to include in the report 
                         (default: [CRITICAL, ERROR, WARNING], INFO is excluded by default)
    """
    # Extract data from the results object
    analysis_results = analysis_result_data.results
    source_filename = analysis_result_data.source_filename
    log_type = analysis_result_data.log_type
    
    # Set default severity levels if none provided - explicitly exclude INFO
    if severity_levels is None:
        severity_levels = [SeverityLevel.CRITICAL, SeverityLevel.ERROR, SeverityLevel.WARNING]
    
    # Initialize rich console
    console = Console()
    
    # Define color mappings for severity
    severity_styles = {
        SeverityLevel.CRITICAL: "white on red bold",
        SeverityLevel.ERROR: "red bold",
        SeverityLevel.WARNING: "yellow bold",
        SeverityLevel.INFO: "cyan"
    }
    
    # Print header
    console.print("\n")
    console.rule(f"[bold blue]LOG ANALYSIS REPORT - {log_type.upper()}")
    console.print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if source_filename:
        console.print(f"Source: {source_filename}")
    console.rule()
    console.print("\n")
    
    # Track if any analyses were included
    any_analysis_included = False
    
    # Process each analysis result
    for i, analysis in enumerate(analysis_results):
        if not analysis:
            continue
            
        # Skip this analysis if its highest severity is not in the allowed levels
        if analysis.highest_severity and analysis.highest_severity not in severity_levels:
            continue
            
        any_analysis_included = True
        
        # Section header with line range
        if hasattr(analysis, 'start_line') and analysis.start_line is not None:
            section_title = f"LINES {analysis.start_line}-{analysis.end_line} ANALYSIS"
        else:
            section_title = f"BATCH {i+1} ANALYSIS"
        
        console.print(f"\n[bold green]{section_title}")
        console.rule(style="green")
        
        # Show severity and attention status
        if analysis.highest_severity:
            severity_style = severity_styles.get(analysis.highest_severity, "")
            console.print(f"Highest Severity: [{severity_style}]{analysis.highest_severity.upper()}")
        
        attention_style = "red" if analysis.requires_immediate_attention else "green"
        console.print(f"Requires Immediate Attention: [{attention_style}]{'YES' if analysis.requires_immediate_attention else 'NO'}")
        
        # Print observations
        if analysis.observations:
            console.print(f"\n[blue]KEY OBSERVATIONS:")
            for obs in analysis.observations:
                console.print(f"  • {obs}")
        
        # Print events
        if analysis.events:
            console.print(f"\n[bold]DETECTED EVENTS:")
            
            for event in analysis.events:
                # Skip events with severity not in the filter
                if event.severity not in severity_levels:
                    continue
                
                # Get style for this severity
                severity_style = severity_styles.get(event.severity, "")
                
                # Create a table for event details
                event_table = RichTable(
                    show_header=False, 
                    box=box.ROUNDED,
                    expand=True,
                    pad_edge=True,
                    title=f"[{severity_style}]{event.severity.upper()}[/] - {event.category.capitalize()} Event"
                )
                
                event_table.add_column("Property", style="bold cyan", width=20)
                event_table.add_column("Value")
                
                # Add event details as table rows
                event_table.add_row("Reasoning", event.reasoning)
                event_table.add_row("Recommended Action", event.recommended_action)
                
                # Add related log IDs
                log_ids = [log.log_id for log in event.relevant_log_ids]
                event_table.add_row("Related Log IDs", ", ".join(log_ids))
                
                # Render the table
                console.print(event_table)
                
                # If logs were provided, show them in a panel
                if logs:
                    # Format logs with IDs for searching
                    formatted_logs = format_logs_with_id(logs)
                    
                    # Find relevant logs
                    relevant_logs = []
                    for log_id in event.relevant_log_ids:
                        log = log_id.find_in(formatted_logs)
                        if log:
                            relevant_logs.append(log)
                    
                    if relevant_logs:
                        log_panel = Panel(
                            "\n".join(relevant_logs),
                            title="Log Content",
                            border_style="blue",
                            padding=(1, 2)
                        )
                        console.print(log_panel)
                
                console.print("\n")
    
    # If no analyses were included, add a "no issues found" message
    if not any_analysis_included:
        console.print("\n[green]No issues matching the selected severity levels were found.[/]\n")
    
    console.rule()
    console.print("\n")
    return