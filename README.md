# LOG-SENSE - AI-Powered Log Analysis System


## About

LOG-SENSE is a Proof-of-Concept (POC) AI-powered log analysis system designed to identify potential security issues, service errors, and performance problems in infrastructure logs. The system leverages structured generation techniques using Outlines (https://github.com/dottxt-ai/outlines) and Pydantic (https://docs.pydantic.dev/latest) on compatible LLM APIs/Engines to ensure consistent, strongly-typed output. 

LOG-SENSE processes log files in smaller chunks, analyzes them with AI language models using predefined prompt templates, and generating structured JSON output based on a Pydantic schema.

## Features

- Produces strongly-typed JSON output using [Outlines](https://github.com/dottxt-ai/outlines)
- Analyzes logs with varying degrees of confidence
- Categorizes events by type (security, performance, configuration, etc.)
- Assigns severity levels (critical, error, warning, info)
- Generates both console reports and PDF documentation
- Provides specific recommended actions for each identified issue
- Processes logs in manageable chunks for efficient analysis

## Libraries

- [Outlines](https://github.com/dottxt-ai/outlines) for structured generation
- [Pydantic](https://docs.pydantic.dev/latest) for strongly-typed output
- [Transformers](https://huggingface.co/transformers) for model loading and tokenization
- [vLLM](https://docs.vllm.ai/en/latest/) for LLM serving/inference support

## Requirements

- Python 3.11+
- CUDA-enabled GPU (check if your GPU is CUDA-enabled [here](https://developer.nvidia.com/cuda-gpus))

## Getting started

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
python cli.py data/logs/linux-example.log --model Qwen/Qwen2.5-7B-Instruct --log-type "linux server" --prompt-template prompt.txt --chunk-size 20 --limit 100
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

```bash
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
python cli.py data/logs/linux-example.log --model Qwen/Qwen2.5-7B-Instruct --log-type "linux server" --format pdf --prompt-template prompt.txt --chunk-size 20 --limit 100 --output reports
```

For JSON output:

```bash
python cli.py data/logs/linux-example.log --model Qwen/Qwen2.5-7B-Instruct --log-type "linux server" --format json --prompt-template prompt.txt --chunk-size 20 --limit 100 --output reports
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
print(results)
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
2. JSON output for structured data analysis and integration with other tools
3. PDF reports for documentation and sharing

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


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.