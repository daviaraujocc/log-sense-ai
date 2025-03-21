# LOG-SENSE - AI-Powered Log Analysis System

<img src="https://img.shields.io/badge/license-MIT-blue"/>

## About

LOG-SENSE is a Proof-of-Concept (POC) AI-powered log analysis system designed to identify potential security issues, service errors, and performance problems in infrastructure logs. The system leverages structured generation techniques using Outlines (https://github.com/dottxt-ai/outlines) and Pydantic (https://docs.pydantic.dev/latest) on compatible LLM APIs/Engines to ensure consistent, strongly-typed output. 

## Features

- Analyzes logs using large language models (LLMs) for context-aware insights
- Produces strongly-typed JSON output using [Outlines](https://github.com/dottxt-ai/outlines)
- Categorizes events by type (security, performance, configuration, etc.)
- Assigns severity levels (critical, error, warning, info)
- Generates reports in console, PDF, and JSON formats
- Provides specific recommended actions for each identified issue
- Processes logs in manageable chunks for efficient analysis

## Table of contents

<details>
<summary> Click to Expand </summary>

- [How it works](#how-it-works)
- [Tech stacks](#tech-stacks)
- [Requirements](#requirements)
- [Getting started](#getting-started)
  - [Clone the Repository](#clone-the-repository)
  - [Install Dependencies](#install-dependencies)
  - [Running](#running)
  - [Download Sample Logs (Optional)](#download-sample-logs-optional)
    - [Available Log Types](#available-log-types)
- [LOGSENSE Parameters](#logsense-parameters)
  - [LOGSENSE Methods](#logsense-methods)
- [Configuration](#configuration)
    - [Schema](#schema)
    - [Prompt Template](#prompt-template)
        - [Example Prompt Template](#example-prompt-template)
- [Advanced Usage](#advanced-usage)
    - [Customizing the Model](#customizing-the-model)
    - [Processing Options](#processing-options)
    - [Output Formats](#output-formats)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)
- [License](#license)

</details>


## How it works

Here's how LOG-SENSE makes sense of your log data:

1. It will split your large log files into smaller pieces (usually 20 lines each) so they're easier to analyze and don't overwhelm the AI window context.

2. Every log line gets its own unique ID (like LOGID-xxxxxxx) based on a hash md5, making it easy to point to specific logs when explaining problems.

3. The system examines each chunk of logs using:
   - A prompt template containing the logs and the instructions that tell the AI what to look for
   - A large language model (LLM) that understands the context of your logs and can generate structured data

4. All issues get categorized by:
   - How serious they are (from critical emergencies to just FYI information)
   - What type of problem it is (security threat, system error, performance bottleneck, etc.)
   - What's happening and what you should do about it

5. The findings are delivered in formats that work for you:
   - Instant on-screen results 
   - PDF reports for sharing with your team or management
   - Structured data (JSON) for feeding into other analysis tools

## 🛠️ Tech stacks

- [Outlines](https://github.com/dottxt-ai/outlines) for structured generation
- [Pydantic](https://docs.pydantic.dev/latest) for strongly-typed output
- [Transformers](https://huggingface.co/transformers) for model loading and tokenization
- [vLLM](https://docs.vllm.ai/en/latest/) for LLM serving/inference support

## Requirements

- Python 3.11+
- CUDA 12.1+ and compatible drivers (check if your GPU is CUDA-enabled [here](https://developer.nvidia.com/cuda-gpus))
- Minimum 10GB GPU memory recommended (8GB may work with smaller models)
- 16GB RAM or higher recommended
- Operating Systems: Ubuntu 20.04+, Windows 10/11 with WSL2

## 🚀 Getting started

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

### Running

Via CLI:

```bash
python cli.py data/logs/linux-example.log --model Qwen/Qwen2.5-7B-Instruct \ 
--log-type "linux server" --prompt-template prompt.txt \ 
--chunk-size 20 --limit 100
```

<details>
<summary> CLI Options </summary>

  - `--model`, `-m`: Model to use for analysis (default: "Qwen/Qwen2.5-7B-Instruct")
  - `--log-type`: Type of log being analyzed for LLM context (default: "server")
  - `--token-max`: Maximum token context size for processing (default: 32000)
  - `--gpu-mem-util`: GPU memory utilization (0.0 to 1.0) (default: 0.95)
  - `--max-model-len`: Maximum model length (optional)
  - `--prompt-template`: Path to custom prompt template file (default: "prompt.txt")
  - `--chunk-size`: Number of log lines to process in each batch (default: 20)
  - `--limit`: Limit the number of log lines to process (default: None)
  - `--format`: Output format for the analysis results (choices: "console", "pdf", "json") (default: "console")
  - `--severity`: Severity levels to include in reports (choices: "critical", "error", "warning", "info") (default: ["critical", "error", "warning"])
  - `--output`, `-o`: Output location for PDF or JSON files (directory or full path) (default: "reports")
  - `--filename`, `-f`: Filename for the output report (default: <log_file>_report.<ext>)

</details>

<details>
<summary> Expected Output </summary>

```console
LOG ANALYSIS REPORT - LINUX SERVER 
Generated on: 2025-03-16 23:04:40
Source: data/logs/linux-example.log


LINES 0-19 ANALYSIS

Highest Severity: ERROR
Requires Immediate Attention: YES

KEY OBSERVATIONS:
  • Multiple failed SSH login attempts from the same IP address. This could indicate a brute force attack.
  • Multiple out-of-memory conditions leading to the termination of HTTPD processes. This could indicate a resource exhaustion attack or misconfiguration.

DETECTED EVENTS:

CRITICAL - Security Event

Reasoning:  Multiple failed SSH login attempts from the same IP address. This could indicate a brute force attack.

Recommended Action:  Implement rate limiting on SSH access, use a firewall to block the IP address, and monitor the system for further suspicious activity.                                        

Related Log IDs: LOGID-4453d69a, LOGID-ccd22302, LOGID-2371b831, LOGID-bac785d3, LOGID-d70cb272, LOGID-68de42db                                                            

Log Content                                                                   
LOGID-4453d69a: Aug 29 07:22:24 combo sshd(pam_unix)[794]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=220.82.197.48  user=root                                           
LOGID-ccd22302: Aug 29 07:22:25 combo sshd(pam_unix)[796]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=220.82.197.48  user=root                                              
LOGID-2371b831: Aug 29 07:22:26 combo sshd(pam_unix)[798]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=220.82.197.48  user=root                                              
LOGID-bac785d3: Aug 29 07:22:26 combo sshd(pam_unix)[800]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=220.82.197.48  user=root                                              
LOGID-d70cb272: Aug 29 07:22:26 combo sshd(pam_unix)[801]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=220.82.197.48  user=root                                              
LOGID-68de42db: Aug 29 07:22:27 combo sshd(pam_unix)[802]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=220.82.197.48  user=root                                                                                                                                                                                                                                             

```
</details>

To generate the pdf report, you can use the `--format pdf` option:

```bash
python cli.py data/logs/linux-example.log --model Qwen/Qwen2.5-7B-Instruct \ 
--log-type "linux server" --format pdf --prompt-template prompt.txt \
--chunk-size 20 --limit 100 --output reports
```

For JSON output:

```bash
python cli.py data/logs/linux-example.log --model Qwen/Qwen2.5-7B-Instruct \ 
--log-type "linux server" --format json --prompt-template prompt.txt \ 
--chunk-size 20 --limit 100 --output reports
```

Then check the `reports` directory for the generated reports.

Via Python 🐍: 

You can check a quick start in the `example.py` file to how to use the `LOGSENSE` class. Here is a quick example:

```python
from log_sense import LOGSENSE
import outlines

from transformers import AutoTokenizer

# Load a model and tokenizer
model_name = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = outlines.models.vllm(
    model_name,
    dtype="auto",
    gpu_memory_utilization=0.95,
    disable_sliding_window=True,
    enable_prefix_caching=True,
)

# Create a LOGSENSE instance
logs_analyzer = LOGSENSE(
    model,                  # Outlines model instance
    tokenizer,              # Tokenizer compatible with the model
    log_type="linux server", # Type of logs being analyzed
    token_max=32000,        # Maximum tokens for generation
    prompt_template=None   # Optional custom prompt template
)

# Analyze logs with these parameters
results = logs_analyzer.analyze_logs(
    logs,                    # List of log lines
    chunk_size=20,           # Number of log lines to process at once
    limit=None,              # Maximum number of log lines to process
    source_filename=None    # Original log filename (for reporting)
)

## Check the results
for analysis in results.results:
    print(analysis.model_dump_json(indent=2))
```

### Download Sample Logs (Optional)

If you want to test LOGSENSE with some real-world sample logs, you can download them using the `setup_data.py` script:

```bash
python setup_data.py --log-type linux
```

#### Available Log Types

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


> The downloaded logs will be extracted and renamed to `<log_type>.log` in the specified output directory (default: `data/logs`).

## LOGSENSE Parameters

The `LOGSENSE` class accepts the following parameters:

```python
logs_analyzer = LOGSENSE(
    model,                  # Outlines model instance
    tokenizer,              # Tokenizer compatible with the model
    log_type="linux server", # Type of logs being analyzed, this has nothing to do with the code, it's just a context to help the model
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

#### Example Prompt Template

```txt
You are an expert security analyst specializing in {log_type} analysis.
Your task is to analyze the following log entries and identify potential security issues,
service errors, or performance problems.

Please provide a structured analysis using the following JSON schema:
{model_schema}

Log entries to analyze:

<LOGS>

{logs}

</LOGS>

The analysis should focus on detecting patterns, anomalies, and potential security threats.
```

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

> Instruct and Coder models are highly recommended for structured data generation.

### Processing Options

Adjust these parameters to optimize performance and analysis quality:

- `chunk_size`: Number of log lines processed at once (lower for more detailed analysis)
- `token_max`: Maximum tokens for generation, since vLLM has a relatively low token limit you need to ensure that the token_max is high enough to generate the full structure (json) before the end token
- `max_model_len`: Maximum context length for the model (lower for less memory usage but affects performance)
- `gpu_memory_utilization`: Fraction of GPU memory to use

### Output Formats

LOG-SENSE provides two output formats:

1. Console reports for quick assessment
2. JSON output for structured data analysis and integration with other tools
3. PDF reports for documentation and sharing

## Limitations

- Analysis quality depends on the underlying language model
- May generate false positives or miss subtle issues
- Processing large log files requires significant computational resources
- Optimal results require tuning chunk size and model parameters
- Currently optimized for server logs; other log formats may require prompt adjustments

## Troubleshooting

1. **Out of Memory Errors**
   - Reduce chunk_size
   - Lower max_model_len
   - Use a smaller model
   - Increase gpu_memory_utilization

2. **Slow Processing**
   - Increase chunk_size
   - Use a smaller model

3. **Poor Analysis Quality**
   - Try a different model
   - Adjust prompt template
   - Adjust token_max value
   - Reduce chunk_size for more detailed analysis

4. **JSON Validation Failures**
   - Reduce chunk_size
   - Increase token_max
   - Try a model with better instruction following capabilities


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.