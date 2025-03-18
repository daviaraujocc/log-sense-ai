import outlines
from transformers import AutoTokenizer
import os
from log_sense import LOGSENSE

# Improve logging for cleaner output
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

# The model we're using
model_name = "Qwen/Qwen2.5-7B-Instruct"

# The template for the prompt
prompt_template_path = "prompt.txt"

# The type of logs we're analyzing
log_type = "linux server"

# Load the tokenizer for the model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Setting up the AI model
model = outlines.models.vllm(
    model_name,
    
    # Automatically choose best precision for your GPU
    dtype="auto",
    
    # Speeds up repeated prompt prefixes
    enable_prefix_caching=True,
    
    # Enable so prefix caching can be used
    disable_sliding_window=True,
    
    # Use most of the available GPU memory
    gpu_memory_utilization=0.95,

    # Maximum length of the model
    # Change this value if you have more GPU memory
    max_model_len=20000,
    
    # Disable CUDA Graph for less resource usage
    enforce_eager=True,
)

test_logs = [
    "data/logs/linux-example.log",
]

# Choose the access log for giggles
log_path = test_logs[0]

# Load the logs into memory
with open(log_path, "r", encoding="ISO-8859-1") as file:
    logs = file.readlines()

# Start the analysis
try:
    # Initialize the LOGSSENSE class
    logs_analyzer = LOGSENSE(
        model=model,
        tokenizer=tokenizer,
        log_type=log_type,
        token_max=32000, # Maximum context window size for processing
        prompt_template_path=prompt_template_path,
    )
    
    # Analyze the logs
    results = logs_analyzer.analyze_logs(logs,   
                                         chunk_size=20, # Process 20 log lines at a time
                                         limit=100 # Limit to 100 lines for this example
                                         )
    
    
    print(results.model_dump_json(indent=2))
        
except Exception as e:
    print(f"Error during analysis: {str(e)}")
    import traceback
    traceback.print_exc()



