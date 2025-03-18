"""
LogSense: AI-Powered Log Analysis Framework
==========================================

LogSense uses AI language models to quickly analyze log files, identify critical issues,
and provide actionable recommendations. It automatically detects errors, security threats,
performance bottlenecks, and other system anomalies.

Basic Usage:
-----------
```python
# Initialize
from log_sense import LOGSENSE
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

analyzer = LOGSENSE(model, tokenizer, log_type="nginx", token_max=2048)

# Analyze logs
with open("logs.txt") as f:
    logs = f.readlines()
    
results = analyzer.analyze_logs(logs, chunk_size=20)

# Process results
print(results.model_dump_json(indent=2))
```
"""

import outlines
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, Dict, List
from enum import Enum
import hashlib
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
import os


def format_logs_with_id(log_entries):
    """Format log entries with LogID prefixes and return formatted logs along with ID mapping."""
    formatted_logs = []
    
    for i, log in enumerate(log_entries):
        log = log.strip()
        if not log:
            continue
        
        # Generate more unique hash using the full log content
        hash_obj = hashlib.md5(log.encode())
        hash_digest = hash_obj.hexdigest()[:8]  
        
        log_id = f"LOGID-{hash_digest}"
        
        formatted_logs.append(f"{log_id}: {log}")
    
    return formatted_logs


class SeverityLevel(str, Enum):
    """The severity levels for log events.""" 
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    INFO = "info"  # This will be excluded by default in reports

class EventCategory(str, Enum):
    """Categories of events for better classification."""
    GENERAL = "general"
    PERFORMANCE = "performance"
    SECURITY = "security"
    AVAILABILITY = "availability"
    CONFIGURATION = "configuration"
    COMPLIANCE = "compliance"
    RESOURCE = "resource" 

class LogID(BaseModel):
    log_id: str = Field(
        description="The ID of the log entry in the format of LOGID-<HASH> where <HASH> indicates the hash of the log entry."
    )

    def find_in(self, logs: List[str]) -> Optional[str]:
        for log in logs:
            if self.log_id in log:
                return log
        return None

class Event(BaseModel):
    """Model representing a detected event in logs."""
    relevant_log_ids: List[LogID]
    severity: SeverityLevel
    category: EventCategory
    reasoning: str
    recommended_action: str

class LogAnalysis(BaseModel):
    """Top-level model for log analysis results."""
    # Analysis metadata
    highest_severity: Optional[SeverityLevel]
    requires_immediate_attention: bool
    
    # Observations about the logs
    observations: List[str]
    events: List[Event]
    
    # These fields are added after the model is created
    start_line: Optional[int] = None
    end_line: Optional[int] = None

class AnalysisResults(BaseModel):
    """Container for analysis results and metadata."""
    results: List[Optional[LogAnalysis]]
    source_filename: Optional[str] = None
    log_type: str
    
    def requires_attention(self) -> bool:
        """Check if any analysis requires immediate attention.""" 
        for analysis in self.results:
            if analysis and analysis.requires_immediate_attention:
                return True
        return False
    
    def highest_severity(self) -> Optional[SeverityLevel]:
        """Return the highest severity level found across all analyses.""" 
        highest = None
        severity_order = {
            SeverityLevel.CRITICAL: 3,
            SeverityLevel.ERROR: 2, 
            SeverityLevel.WARNING: 1,
            SeverityLevel.INFO: 0
        }
        
        for analysis in self.results:
            if not analysis or not analysis.highest_severity:
                continue
                
            if highest is None or severity_order[analysis.highest_severity] > severity_order[highest]:
                highest = analysis.highest_severity
                
        return highest


class LOGSENSE:
    def __init__(
        self,
        model,
        tokenizer,
        log_type: str,
        token_max: int,
        prompt_template_path: Optional[str] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.log_type = log_type
        self.token_max = token_max
        self.generator = outlines.generate.json(
            self.model,
            LogAnalysis,
            sampler=outlines.samplers.greedy()
        )
        self.console = Console()

    
        if prompt_template_path is None:
            prompt_template_path = "prompt.txt"

        with open(prompt_template_path, "r") as file:
            self.prompt_template = file.read()

    def analyze_logs(self, logs: List[str], chunk_size: int = 20, limit: Optional[int] = None, source_filename: Optional[str] = None):
        """Analyze a list of log entries.
        
        Args:
            logs: List of log entries to analyze
            chunk_size: Number of logs to process in each batch
            limit: Maximum number of log entries to process (None means no limit)
            source_filename: Original log file name being analyzed
            
        Returns:
            AnalysisResults: Object containing analysis results and metadata
        """

        results = []
        
        # Apply limit if specified
        if limit is not None and limit < len(logs):
            logs = logs[:limit]
        
        total_batches = (len(logs) + chunk_size - 1) // chunk_size
        
        with Progress(
            TextColumn("[blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(f"Analyzing logs", total=total_batches)
            
            for i in range(0, len(logs), chunk_size):
                chunk = logs[i:i + chunk_size]
                formatted_logs = format_logs_with_id(chunk)

                formatted_logs = "\n\n".join(formatted_logs)

                prompt = self._to_prompt(formatted_logs, LogAnalysis)

                try:
                    # TODO: Disable TQDM output
                    resp = self.generator(prompt, max_tokens=self.token_max)
                    
                    # Set start and end line directly on the model fields
                    resp.start_line = i+1
                    resp.end_line = min(i + chunk_size - 1, len(logs) - 1) + 1
                    results.append(resp)
                except ValidationError as ve:
                    self.console.print(f"\n Pydantic validation error for batch starting at line {i}: {str(ve)}", style="bold red")
                    results.append(None)  
                except Exception as e:
                    self.console.print(f"\n Generation failed for batch starting at line {i}: {str(e)}", style="bold red")
                    self.console.print("Exiting...", style="bold red")
                    os._exit(1)
                
                progress.update(task, advance=1)
        
        return AnalysisResults(
            results=results,
            source_filename=source_filename,
            log_type=self.log_type
        )

    def _to_prompt(self, text: str, pydantic_class: BaseModel) -> str:
        
        messages = []
        

        messages.append(
            {"role": "user", "content": self.prompt_template.format(
                log_type=self.log_type,
                logs=text,
                model_schema=pydantic_class.model_json_schema()
            )}
        )

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )



