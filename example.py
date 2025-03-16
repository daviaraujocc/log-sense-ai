# A tool for analyzing Linux server logs using AI to identify issues and generate reports

import outlines
import torch
from transformers import AutoTokenizer
import json
import os

from log_sense import LOGSENSE, generate_report, generate_console_report

# The model we're using
#model_name = "microsoft/Phi-3-mini-4k-instruct"
model_name = "Qwen/Qwen2.5-7B-Instruct"

# The type of logs we're analyzing
log_type = "linux server"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Setting up the AI model
model = outlines.models.vllm(
    model_name,
    
    # Automatically choose best precision for your GPU
    dtype="auto",
    
    # Speeds up repeated prompt prefixes
    enable_prefix_caching=True,
    
    # So prefix caching can be used
    disable_sliding_window=True,
    
    # Use most of the available GPU memory
    gpu_memory_utilization=0.95,
    
    # Maximum model length for less memory footprint, if you have more VRAM then you can comment this value
    max_model_len=10400,
    
    # Disable CUDA Graph for less resource usage
    enforce_eager=True,
)

test_logs = [
    "data/logs/linux-example.log",
]

# Choose the access log for giggles
log_path = test_logs[0]

# Load the logs into memory
with open(log_path, "r") as file:
    logs = file.readlines()

# Start the analysis
try:
    # Initialize the LOGSSENSE class
    logs_analyzer = LOGSENSE(
        model=model,
        tokenizer=tokenizer,
        log_type=log_type,
        
        # Maximum context window size for processing
        token_max=32000,
    )
    
    # Analyze the logs
    results = logs_analyzer.analyze_logs(logs,
                                         # Process 10 log lines at a time
                                         chunk_size=20,
                                         # Limit to 30 chunks for this example
                                         limit=20,
                                         source_filename=log_path)
    
    
    # Generate a console report with the most important findings
    generate_console_report(results, logs, severity_levels=["critical", "error", "warning"])
    
    # Generate a PDF summary with all the findings organized by severity
    pdf_path = generate_report(results,
        output_path="reports",
        filename="report.pdf",
        # Focus on the most important issues
        severity_levels=["critical", "error", "warning"],
        logs=logs
    )


    print(f"Report generated at: {pdf_path}")
        
except Exception as e:
    print(f"Error during analysis: {str(e)}")
    import traceback
    traceback.print_exc()



