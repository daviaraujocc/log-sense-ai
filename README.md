# LOG-SENSE - AI-Powered Log Analysis System


## About

LOG-SENSE is a Proof-Of-Concept (POC) of AI-powered log analysis system designed to identify potential security issues, service errors, and performance problems in infrastructure logs. The system leverages structured generation techniques using [Outlines](https://github.com/dottxt-ai/outlines) and [Pydantic](https://docs.pydantic.dev/latest) to ensure consistent, strongly-typed output.

LOG-SENSE processes arbitrary logs in chunks, generating a comprehensive analysis including severity assessment, potential security issues, and recommended actions for each identified event.

## Features

- Produces strongly-typed JSON output using [Outlines](https://github.com/dottxt-ai/outlines)
- Analyzes logs with varying degrees of confidence
- Categorizes events by type (security, performance, configuration, etc.)
- Assigns severity levels (critical, error, warning, info)
- Generates both console reports and PDF documentation
- Provides specific recommended actions for each identified issue
- Processes logs in manageable chunks for efficient analysis

## Technologies

- [Outlines](https://github.com/dottxt-ai/outlines) for structured generation
- [Pydantic](https://docs.pydantic.dev/latest) for strongly-typed output
- [Transformers](https://huggingface.co/transformers) for model loading and tokenization
- [vLLM](https://docs.vllm.ai/en/latest/) for LLM serving/inference support

## Requirements

- Python 3.11+
- CUDA-enabled GPU (recommended for performance)

## Installation

### Clone the Repository

```bash
git clone https://github.com/daviaraujocc/log-sense-ai.git
cd log-sense-ai
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

Alternatively, create a conda environment:

```bash
conda env create -f environment.yml
conda activate log-sense-ai
```

## Quick Start

You can check a quick start example in the `example.py` file. Here is a brief overview of the process:

```python
import outlines
import torch
from transformers import AutoTokenizer
from log_sense import LOGSENSE, generate_report, generate_console_report

# Configure model
model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = outlines.models.vllm(
    model_name,
    dtype="auto",
    enable_prefix_caching=True,
    disable_sliding_window=True,
    gpu_memory_utilization=0.95,
    enforce_eager=True, # disable cuda graph
)

# Load log file
log_path = "data/logs/linux-example.log"
with open(log_path, "r") as file:
    logs = file.readlines()

# Initialize analyzer and process logs
logs_analyzer = LOGSENSE(
    # Configure the model 
    model=model,
    # Configure the tokenizer
    tokenizer=tokenizer,
    # Set log type
    log_type="linux server",
    # Set maximum tokens for generation
    token_max=32000,
)

# Analyze logs
results = logs_analyzer.analyze_logs(
    # Pass the logs to analyze
    logs,
    # Set the chunk size (number of log lines to process at once)
    chunk_size=20,
    # Set the maximum number of log lines to process
    limit=100,
    # Set the original log filename (for reporting)
    source_filename=log_path
)

# Generate reports
generate_console_report(results, logs, severity_levels=["critical", "error", "warning"])
pdf_path = generate_report(
    results,
    output_path="reports",
    filename="report.pdf",
    severity_levels=["critical", "error", "warning"],
    logs=logs
)
```

## Command Line Interface

LOG-SENSE includes a command-line interface for quick analysis of log files:

```bash
python cli.py path/to/logfile.log --model Qwen/Qwen2.5-7B-Instruct --log-type "linux server" --chunk-size 20 --limit 100
```

### CLI Options

- **Required positional argument**:
  - Log file path: Path to the log file to analyze

- **Model Configuration**:
  - `--model`, `-m`: Model to use for analysis (default: "Qwen/Qwen2.5-7B-Instruct")
  - `--log-type`: Type of log being analyzed (default: "server")
  - `--token-max`: Maximum token context size for processing (default: 32000)
  - `--gpu-mem-util`: GPU memory utilization (0.0 to 1.0) (default: 0.95)
  - `--max-model-len`: Maximum model length (optional)

- **Analysis Configuration**:
  - `--chunk-size`: Number of log lines to process in each batch (default: 20)
  - `--limit`: Limit the number of chunks to process (default: None)

- **Output Configuration**:
  - `--output`: Output format for the analysis results ("console" or "pdf") (default: "console")
  - `--severity`: Severity levels to include in reports (choices: "critical", "error", "warning", "info") (default: ["critical", "error", "warning"])
  - `--output-dir`: Directory to save PDF reports (default: "reports")
  - `--filename`: Filename for the PDF report (default: <log_file>_report.pdf)

## Downloading Test Logs

LOG-SENSE includes a utility script (`setup_data.py`) to download test log datasets from the [LogHub collection](https://zenodo.org/records/8196385):

```bash
# Download Linux logs (default)
python setup_data.py

# Download a specific log type
python setup_data.py --log-type hadoop

# List all available log types
python setup_data.py --list

# Specify a custom output directory
python setup_data.py --log-type apache --output-dir custom/path/logs
```

### Available Log Types

- linux: Linux system logs
- hadoop: Hadoop HDFS logs
- spark: Apache Spark logs
- zookeeper: Apache ZooKeeper logs
- bgl: Blue Gene/L supercomputer logs
- hpc: High Performance Computing logs
- thunderbird: Thunderbird supercomputer logs
- windows: Windows event logs
- apache: Apache HTTP server logs
- proxifier: Proxifier software logs
- openstack: OpenStack logs

The downloaded logs will be extracted and renamed to `<log_type>.log` in the specified output directory (default: `data/logs`).

## LOGSENSE Parameters

The `LOGSENSE` class accepts the following parameters:

```python
logs_analyzer = LOGSENSE(
    model,                  # Outlines model instance
    tokenizer,              # Tokenizer compatible with the model
    log_type="linux server", # Type of logs being analyzed
    token_max=32000,        # Maximum tokens for generation
    prompt_template=None   # Optional custom prompt template
)
```

### LOGSENSE Methods

```python
# Analyze logs with these parameters
results = logs_analyzer.analyze_logs(
    logs,                    # List of log lines
    chunk_size=20,           # Number of log lines to process at once
    limit=None,              # Maximum number of log lines to process
    source_filename=None    # Original log filename (for reporting)
)
```

## Configuration

### Schema

LOG-SENSE uses a structured schema to ensure consistent output:

```python
class LogAnalysis(BaseModel):
    highest_severity: Optional[SeverityLevel]
    requires_immediate_attention: bool
    observations: List[str]
    events: List[Event]
    # Optional fields
    start_line: Optional[int] = None
    end_line: Optional[int] = None
```

### Prompt Template

The default prompt template is configured in `prompt.txt`. The template must include:

- `{log_type}`: Type of log being analyzed
- `{model_schema}`: JSON schema for the output
- `{logs}`: The log entries to analyze

## Advanced Usage

### Customizing the Model

LOG-SENSE supports any LLM compatible with the Outlines library:

```python
model = outlines.models.vllm(
    "microsoft/Phi-3-mini-4k-instruct",
    dtype=torch.bfloat16,  
    max_model_len=32000,  
    gpu_memory_utilization=0.95, 
)
```

### Processing Options

Adjust these parameters to optimize performance and analysis quality:

- `chunk_size`: Number of log lines processed at once (lower for more detailed analysis)
- `token_max`: Maximum tokens for generation, since vLLM has a relatively low token limit you need to ensure that the token_max is high enough to generate the full structure (json) before the end token
- `max_model_len`: Maximum context length for the model (lower for less memory usage but affects performance)
- `gpu_memory_utilization`: Fraction of GPU memory to use

### Output Formats

LOG-SENSE provides two output formats:

1. Console reports with color-coded formatting for quick assessment
2. PDF reports for documentation and sharing

## Limitations

- Analysis quality depends on the underlying language model
- May generate false positives or miss subtle issues
- Processing large log files requires significant computational resources
- Optimal results require tuning chunk size and model parameters
- Currently optimized for server logs; other log formats may require prompt adjustments

## Troubleshooting

Common issues and solutions:

1. **Out of Memory Errors**
   - Reduce chunk_size
   - Lower max_model_len
   - Increase gpu_memory_utilization

2. **Slow Processing**
   - Increase chunk_size
   - Use a smaller model
   - Enable prefix caching

3. **Poor Analysis Quality Or Validation Errors**
   - Try a different model
   - Adjust prompt template
   - Adjust token_max value
   - Reduce chunk_size for more detailed analysis


