"""
Configuration settings for the VisionGPT Dataset Manager.
"""

from pathlib import Path

# The root directory where datasets are stored.
# Defaults to a folder named "Dataset" in the workspace root.
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = WORKSPACE_ROOT / "Dataset"

# Download Settings
MAX_THREADS = 4
RETRY_COUNT = 5
RETRY_BACKOFF_FACTOR = 2.0  # Exponential backoff base factor
TIMEOUT = 30  # Timeout in seconds for requests

# Archive Settings
DELETE_ARCHIVES_AFTER_EXTRACTION = False

# Verification Settings
VERIFY_HASH = True
