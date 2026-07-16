"""
Verification module for auditing dataset integrity, file sizes, hashes, and directory structures.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union
from tools.dataset_manager.config import DATASET_ROOT, VERIFY_HASH
from tools.dataset_manager.datasets import DatasetFile, DatasetInfo
from tools.dataset_manager.utils import compute_sha256, format_size

logger = logging.getLogger("DatasetManager.Verifier")


class Verifier:
    """
    Verifies dataset file existence, sizes, checksums, and folder hierarchies.
    """

    def __init__(self, verify_hash: bool = VERIFY_HASH) -> None:
        self.verify_hash = verify_hash

    def _get_status_str(self, success: bool) -> str:
        """
        Returns a formatted colored status string.
        """
        if success:
            return "\033[92m[PASS]\033[0m"  # Green PASS
        else:
            return "\033[91m[FAIL]\033[0m"  # Red FAIL

    def verify_file(self, file_info: DatasetFile) -> Tuple[bool, List[str]]:
        """
        Verifies existence, size, and SHA256 hash of a single archive or data file.
        """
        dest_path = DATASET_ROOT / file_info.sub_dir / file_info.filename
        logs: List[str] = []
        success = True

        # Check existence
        if not dest_path.exists():
            logs.append(f"File {file_info.filename} does not exist.")
            return False, logs

        # Check file size
        actual_size = dest_path.stat().st_size
        if file_info.expected_size is not None:
            if actual_size != file_info.expected_size:
                logs.append(
                    f"Size mismatch for {file_info.filename}. Expected: {format_size(file_info.expected_size)}, Actual: {format_size(actual_size)}"
                )
                success = False
            else:
                logs.append(f"Size match for {file_info.filename} ({format_size(actual_size)}).")
        else:
            logs.append(f"File {file_info.filename} exists ({format_size(actual_size)}). Size not strictly defined.")

        # Check SHA256
        if self.verify_hash and file_info.sha256:
            try:
                logs.append(f"Calculating SHA256 hash for {file_info.filename}...")
                actual_hash = compute_sha256(dest_path)
                if actual_hash != file_info.sha256:
                    logs.append(
                        f"SHA256 mismatch for {file_info.filename}. Expected: {file_info.sha256}, Actual: {actual_hash}"
                    )
                    success = False
                else:
                    logs.append(f"SHA256 match for {file_info.filename}.")
            except Exception as e:
                logs.append(f"Error computing hash for {file_info.filename}: {e}")
                success = False

        return success, logs

    def verify_structure(self, dataset: DatasetInfo) -> Tuple[bool, List[str]]:
        """
        Verifies that the expected directories and files exist after extraction.
        """
        logs: List[str] = []
        success = True

        # Verify directories
        for d in dataset.verify_dirs:
            dir_path = DATASET_ROOT / d
            if not dir_path.is_dir():
                logs.append(f"Expected directory {d} is missing or is not a directory.")
                success = False
            else:
                logs.append(f"Directory {d} verified.")

        # Verify files
        for f in dataset.verify_files:
            file_path = DATASET_ROOT / f
            if not file_path.is_file():
                logs.append(f"Expected file {f} is missing or is not a file.")
                success = False
            else:
                logs.append(f"File {f} verified.")

        return success, logs

    def audit_dataset(self, dataset: DatasetInfo) -> Dict[str, Union[bool, List[str]]]:
        """
        Audits both downloads and extraction status of a dataset.
        """
        logger.info(f"Auditing dataset: {dataset.name}")

        download_success = True
        download_logs: List[str] = []
        for file_info in dataset.files:
            file_success, file_logs = self.verify_file(file_info)
            download_logs.extend(file_logs)
            if not file_success:
                download_success = False

        extraction_success = True
        extraction_logs: List[str] = []
        if dataset.verify_dirs or dataset.verify_files:
            extraction_success, extraction_logs = self.verify_structure(dataset)

        # Print clean CLI summary
        print(f"\n==========================================")
        print(f"AUDIT REPORT: {dataset.name.upper()}")
        print(f"==========================================")
        print(f"Archive files integrity:  {self._get_status_str(download_success)}")
        for log in download_logs:
            print(f"  - {log}")

        print(f"Extracted structure:      {self._get_status_str(extraction_success)}")
        for log in extraction_logs:
            print(f"  - {log}")

        return {
            "download_success": download_success,
            "extraction_success": extraction_success,
            "download_logs": download_logs,
            "extraction_logs": extraction_logs
        }
