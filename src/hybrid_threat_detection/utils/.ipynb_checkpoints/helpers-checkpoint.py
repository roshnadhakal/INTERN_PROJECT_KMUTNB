import os
import logging
import json
from pathlib import Path
from typing import Dict, Any

def create_dir_if_not_exists(path):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)

def validate_file_path(path):
    """Check if file exists"""
    return os.path.exists(path)

def setup_logging(log_file: str = "data_processing.log"):
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def save_json(data: Dict[str, Any], file_path: str):
    """Save dictionary to JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(file_path: str) -> Dict[str, Any]:
    """Load JSON file into dictionary."""
    with open(file_path, 'r') as f:
        return json.load(f)