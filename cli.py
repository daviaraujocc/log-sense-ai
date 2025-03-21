#!/usr/bin/env python3
"""
LOG-SENSE CLI - Command-line interface for log analysis
"""

import argparse
import os
import sys
import outlines
import json
from transformers import AutoTokenizer
from log_sense import LOGSENSE
from utils import generate_report, generate_console_report

# Disable logging for cleaner output
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

def main():
    """Main function for CLI interface to LOG-SENSE."""
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="LOG-SENSE: AI-Powered Log Analysis System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("log_file", help="Path to the log file to analyze")
    
    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model", "-m",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model to use for analysis"
    )
    model_group.add_argument(
        "--log-type",
        default="server",
        help="Type of log being analyzed (e.g., linux, apache, nginx)"
    )
    model_group.add_argument(
        "--token-max",
        type=int,
        default=32000,
        help="Maximum token context size for processing"
    )
    model_group.add_argument(
        "--gpu-mem-util",
        type=float,
        default=0.95,
        help="GPU memory utilization (0.0 to 1.0)"
    )
    model_group.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum model length"
    )
    model_group.add_argument(
        "--prompt-template",
        default=None,
        help="Path to prompt template file"
    )
    
    # Analysis parameters
    analysis_group = parser.add_argument_group("Analysis Configuration")
    analysis_group.add_argument(
        "--chunk-size",
        type=int,
        default=20,
        help="Number of log lines to process in each batch"
    )
    analysis_group.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of chunks to process (None = no limit)"
    )
    
    # Output configuration
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument(
        "--format", 
        choices=["console", "pdf", "json"],
        default="console",
        help="Output format for the analysis results"
    )
    output_group.add_argument(
        "--severity",
        nargs="+",
        choices=["critical", "error", "warning", "info"],
        default=["critical", "error", "warning"],
        help="Severity levels to include in reports"
    )
    output_group.add_argument(
        "--output", "-o",
        default="reports",
        help="Output location for PDF or JSON files (directory or full path)"
    )
    output_group.add_argument(
        "--filename", "-f",
        default=None,
        help="Filename for the output report (default: <log_file>_report.<ext>)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.log_file):
        print(f"Error: Log file '{args.log_file}' not found.")
        sys.exit(1)
    
    # Determine output path
    output_file_path = None
    if args.format != "console":
        base_name = os.path.basename(args.log_file)
        base_filename = os.path.splitext(base_name)[0]
        
        # Check if output is a directory or full path
        output_path = args.output
        
        # If output is a directory or doesn't have the right extension, use default filename
        if os.path.isdir(output_path) or not output_path.endswith(f".{args.format}"):
            # Create directory if it doesn't exist
            os.makedirs(output_path, exist_ok=True)
            
            # Use provided filename or default to log filename + format
            if args.filename:
                filename = args.filename
                # Ensure filename has the right extension
                if not filename.endswith(f".{args.format}"):
                    filename = f"{os.path.splitext(filename)[0]}.{args.format}"
            else:
                filename = f"{base_filename}_report.{args.format}"
                
            output_file_path = os.path.join(output_path, filename)
        else:
            # Output is a full path with filename
            output_dir = os.path.dirname(output_path)
            if output_dir:  # Skip if output_dir is empty (current dir)
                os.makedirs(output_dir, exist_ok=True)
            output_file_path = output_path
    
    # Load logs
    try:
        print(f"Loading logs from {args.log_file}...")
        with open(args.log_file, "r", encoding="ISO-8859-1") as file:
            logs = file.readlines()
        if args.limit is not None:
            logs = logs[:args.limit]
            print(f"Loaded {len(logs)} log lines (limited by --limit={args.limit}).")
        else:
            print(f"Loaded {len(logs)} log lines.")
    except Exception as e:
        print(f"Error loading log file: {str(e)}")
        sys.exit(1)
    
    # Set up tokenizer and model
    try:
        print(f"Initializing model: {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(args.model)

        if args.max_model_len is None:
            model = outlines.models.vllm(
                args.model,
                dtype="auto",
                enable_prefix_caching=True,
                disable_sliding_window=True,
                gpu_memory_utilization=args.gpu_mem_util,
                enforce_eager=True
            )
        else:
            model = outlines.models.vllm(
                args.model,
                dtype="auto",
                enable_prefix_caching=True,
                disable_sliding_window=True,
                gpu_memory_utilization=args.gpu_mem_util,
                max_model_len=args.max_model_len,
                enforce_eager=True
            )

    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        sys.exit(1)
    
    # Start the analysis
    try:
        print("Initializing LOG-SENSE analyzer...")
        logs_analyzer = LOGSENSE(
            model=model,
            tokenizer=tokenizer,
            prompt_template_path=args.prompt_template,
            log_type=f"{args.log_type}",
            token_max=args.token_max,
        )
        
        print(f"Analyzing logs (chunk size: {args.chunk_size}, limit: {args.limit or 'None'})...")
        results = logs_analyzer.analyze_logs(
            logs,
            chunk_size=args.chunk_size,
            limit=args.limit,
            source_filename=args.log_file
        )
        
        # Generate appropriate output
        if args.format == "console":
            print("Generating console report...")
            generate_console_report(
                results,
                logs,
                severity_levels=args.severity
            )
        
        elif args.format == "pdf":
            print("Generating PDF report...")
            pdf_path = generate_report(
                results,
                output_path=os.path.dirname(output_file_path) or ".",
                filename=os.path.basename(output_file_path),
                severity_levels=args.severity,
                logs=logs
            )
            print(f"PDF report generated at: {pdf_path}")
        
        elif args.format == "json":
            print("Generating JSON output...")
            
            # Access the results properly based on the class structure
            # Determine whether results is an AnalysisResults object or a list
            if hasattr(results, 'results'):
                # If results is an AnalysisResults object
                analysis_results = results.results
                source_file = results.source_filename or args.log_file
                log_type = results.log_type
            else:
                # If results is a list of LogAnalysis objects
                analysis_results = results
                source_file = args.log_file
                log_type = args.log_type
            
            # Filter by severity levels
            filtered_results = []
            for r in analysis_results:
                if r and r.highest_severity and r.highest_severity.lower() in args.severity:
                    filtered_results.append(r)
            
            # Convert LogAnalysis objects to dictionaries for JSON
            json_findings = []
            for r in filtered_results:
                # Convert each LogAnalysis to a dictionary
                finding = {
                    "severity": r.highest_severity,
                    "requires_attention": r.requires_immediate_attention,
                    "observations": r.observations,
                    "events": [event.model_dump() for event in r.events],
                    "start_line": r.start_line,
                    "end_line": r.end_line
                }
                json_findings.append(finding)
            
            # Create JSON-compatible structure
            json_output = {
                "source_file": source_file,
                "log_type": log_type,
                "total_findings": len(filtered_results),
                "findings": json_findings
            }
            
            # Save to file
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(json_output, f)
            
            print(f"JSON output generated at: {output_file_path}")
    
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
