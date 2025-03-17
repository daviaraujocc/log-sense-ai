#!/usr/bin/env python3
"""
LOG-SENSE CLI - Command-line interface for log analysis
"""

import argparse
import os
import sys
import outlines
import torch
from transformers import AutoTokenizer
from log_sense import LOGSENSE
from utils import generate_report, generate_console_report

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
        "--output",
        choices=["console", "pdf"],
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
        "--output-dir",
        default="reports",
        help="Directory to save PDF reports"
    )
    output_group.add_argument(
        "--filename",
        default=None,
        help="Filename for the PDF report (default: <log_file>_report.pdf)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.log_file):
        print(f"Error: Log file '{args.log_file}' not found.")
        sys.exit(1)
    
    # Determine PDF filename if not specified
    if args.output == "pdf" and not args.filename:
        base_name = os.path.basename(args.log_file)
        args.filename = f"{os.path.splitext(base_name)[0]}_report.pdf"
    
    # Create output directory if it doesn't exist
    if args.output == "pdf":
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load logs
    try:
        print(f"Loading logs from {args.log_file}...")
        with open(args.log_file, "r", encoding="latin-1") as file:
            logs = file.readlines()
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
                enforce_eager=True,
            )
        else:
            model = outlines.models.vllm(
                args.model,
                dtype="auto",
                enable_prefix_caching=True,
                disable_sliding_window=True,
                gpu_memory_utilization=args.gpu_mem_util,
                max_model_len=args.max_model_len,
                enforce_eager=True,
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
        if args.output == "console":
            print("Generating console report...")
            generate_console_report(
                results,
                logs,
                severity_levels=args.severity
            )
        
        elif args.output == "pdf":
            print("Generating PDF report...")
            pdf_path = generate_report(
                results,
                output_path=args.output_dir,
                filename=args.filename,
                severity_levels=args.severity,
                logs=logs
            )
            print(f"PDF report generated at: {pdf_path}")
    
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
