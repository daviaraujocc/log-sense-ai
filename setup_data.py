#!/usr/bin/env python3
import os
import requests
import zipfile
import tarfile
import io
import argparse
import sys
import tempfile
import shutil

# Downlaod from LogHub
ZENODO_BASE_URL = "https://zenodo.org/records/8196385/files/"

AVAILABLE_LOGS = {
    "linux": "Linux.tar.gz",
    "hadoop": "HDFS.tar.gz",
    "spark": "Spark.tar.gz",
    "zookeeper": "Zookeeper.tar.gz",
    "bgl": "BGL.tar.gz",
    "hpc": "HPC.tar.gz",
    "thunderbird": "Thunderbird.tar.gz",
    "windows": "Windows.tar.gz",
    "apache": "Apache.tar.gz",
    "proxifier": "Proxifier.tar.gz",
    "openstack": "OpenStack.tar.gz"
}

def download_logs(log_type, output_dir="logs"):
    """Download and extract logs from Zenodo repository."""
    if log_type not in AVAILABLE_LOGS:
        print(f"Error: Log type '{log_type}' is not available.")
        print(f"Available log types: {', '.join(AVAILABLE_LOGS.keys())}")
        return False
    
    # Create dir only if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Add download parameter to URL
    url = f"{ZENODO_BASE_URL}{AVAILABLE_LOGS[log_type]}?download=1"
    
    try:
        print(f"Downloading {log_type} logs from {url}...")
        response = requests.get(url)
        response.raise_for_status()
        
        # Create a temporary directory for extraction
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Extracting logs to temporary directory...")
            with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar_ref:
                tar_ref.extractall(temp_dir)
                
            # Find log files in the temp directory
            extracted_files = []
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith(".log"):
                        extracted_files.append(os.path.join(root, file))
            
            # Move log files to output directory with appropriate names
            destination_path = os.path.join(output_dir, f"{log_type}.log")
            if extracted_files:
                shutil.copy2(extracted_files[0], destination_path)
                print(f"Logs saved as {destination_path}")
            else:
                print("No log files found in the downloaded archive.")
                
        print(f"Successfully downloaded and extracted {log_type} logs!")
        return True
    
    except requests.exceptions.RequestException as e:
        print(f"Error downloading logs: {e}")
        return False
    except tarfile.ReadError:
        print("Error: Downloaded file is not a valid tar.gz file.")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def list_available_logs():
    """Print a formatted list of available log types."""
    print("Available log types:")
    for log_type, path in AVAILABLE_LOGS.items():
        print(f"  - {log_type}: {path}")

def main():
    parser = argparse.ArgumentParser(description='Download logs from Zenodo repository.')
    
    # Add arguments
    parser.add_argument('--log-type', '-t', choices=AVAILABLE_LOGS.keys(),
                        default='linux', help='Type of logs to download (default: linux)')
    parser.add_argument('--output-dir', '-o', default='data/logs',
                        help='Directory to save downloaded logs (default: data/logs)')
    parser.add_argument('--list', '-l', action='store_true', 
                        help='List all available log types')
    
    args = parser.parse_args()
    
    if args.list:
        list_available_logs()
        return
    
    success = download_logs(args.log_type, args.output_dir)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
