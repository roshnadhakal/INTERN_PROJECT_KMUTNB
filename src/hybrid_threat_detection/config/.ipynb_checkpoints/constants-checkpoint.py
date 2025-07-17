import os
from pathlib import Path

# Directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Data files
DATA_FILES = {
    "xss": os.path.join(DATA_DIR, "Cross-site_scriptinga_(XSS).csv"),
    "sql_injection": os.path.join(DATA_DIR, "SQL_injection.csv"),
    "path_traversal": os.path.join(DATA_DIR, "Path_traversal.csv"),
    # Add other files as needed
}

# Column mappings for each dataset
COLUMN_MAPPINGS = {
    "xss": {
        "payload_col": "post_data",  # Since there's no payload column
        "label_col": "event_type",    # Need to verify if this indicates malicious/benign
        "source_ip": "src_ip",
        "timestamp": "@timestamp"
    },
    "sql_injection": {
        "payload_col": "payload",
        "label_col": "alert",         # Need to verify
        "source_ip": "src_ip",
        "timestamp": "@timestamp"
    },
    "path_traversal": {
        "payload_col": "payload",
        "label_col": "alert",         # Need to verify
        "source_ip": "src_ip",
        "timestamp": "@timestamp"
    }
}