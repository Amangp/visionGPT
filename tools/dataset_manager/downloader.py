"""
Downloader module for parallel, resumable file downloads with automatic retries.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Optional
import requests

from tools.dataset_manager.config import (
    MAX_THREADS,
    RETRY_COUNT,
    RETRY_BACKOFF_FACTOR,
    TIMEOUT,
    DATASET_ROOT
)
from tools.dataset_manager.datasets import DatasetFile, DatasetInfo
from tools.dataset_manager.progress import progress_manager

logger = logging.getLogger("DatasetManager.Downloader")


class Downloader:
    """
    Downloads dataset files concurrently, supporting resumed transfers and retries.
    """

    def __init__(
        self,
        max_threads: int = MAX_THREADS,
        retry_count: int = RETRY_COUNT,
        timeout: int = TIMEOUT
    ) -> None:
        self.max_threads = max_threads
        self.retry_count = retry_count
        self.timeout = timeout

    def _get_file_size_and_range_support(self, url: str) -> tuple[Optional[int], bool]:
        """
        Performs a HEAD request to determine file size and if range requests are supported.
        """
        try:
            response = requests.head(url, timeout=self.timeout, allow_redirects=True)
            if response.status_code == 200:
                size = response.headers.get("Content-Length")
                accept_ranges = response.headers.get("Accept-Ranges")
                supports_range = accept_ranges == "bytes" or "bytes" in response.headers.get("Content-Range", "")
                return int(size) if size else None, supports_range
            return None, False
        except Exception as e:
            logger.debug(f"HEAD request failed for {url}: {e}")
            return None, False

    def download_file(self, file_info: DatasetFile) -> bool:
        """
        Downloads a single file. Implements range resumes and automatic retries.
        """
        dest_dir = DATASET_ROOT / file_info.sub_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / file_info.filename

        url = file_info.url
        filename = file_info.filename

        # Step 1: Probe server for size & range support
        server_size, supports_range = self._get_file_size_and_range_support(url)
        expected_size = server_size or file_info.expected_size

        # Check existing size
        current_size = dest_path.stat().st_size if dest_path.exists() else 0

        # If file is already fully downloaded, skip it
        if expected_size and current_size >= expected_size:
            logger.info(f"File {filename} is already fully downloaded ({current_size} bytes). Skipping.")
            return True

        # Setup resume header if partial file exists
        headers = {}
        write_mode = "wb"
        start_bytes = 0

        if current_size > 0 and supports_range:
            headers["Range"] = f"bytes={current_size}-"
            write_mode = "ab"
            start_bytes = current_size
            logger.info(f"Resuming download of {filename} from {current_size} bytes.")
        elif current_size > 0:
            logger.info(f"Partial file exists for {filename} but server does not support Range. Restarting.")

        # Initialize progress bar
        progress_manager.start_task(filename, expected_size or 0, initial_bytes=start_bytes)

        # Download with retry loop
        for attempt in range(1, self.retry_count + 1):
            try:
                response = requests.get(url, headers=headers, timeout=self.timeout, stream=True)
                
                # Check status code (206 is Partial Content, 200 is OK)
                if response.status_code not in [200, 206]:
                    raise requests.HTTPError(f"HTTP error status code: {response.status_code}")

                chunk_size = 1024 * 1024  # 1MB chunks
                with open(dest_path, write_mode) as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            progress_manager.update_progress(filename, len(chunk))
                
                progress_manager.finish_task(filename, success=True)
                logger.info(f"Successfully downloaded {filename}.")
                return True

            except Exception as e:
                logger.warning(
                    f"Attempt {attempt}/{self.retry_count} failed for {filename}: {e}"
                )
                if attempt < self.retry_count:
                    sleep_time = RETRY_BACKOFF_FACTOR ** attempt
                    logger.info(f"Retrying in {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"All download attempts failed for {filename}.")
                    progress_manager.finish_task(filename, success=False)
                    return False

        return False

    def download_dataset(self, dataset: DatasetInfo) -> Dict[str, bool]:
        """
        Downloads all files in a dataset concurrently using a ThreadPoolExecutor.
        """
        logger.info(f"Starting download of dataset: {dataset.name}")
        results: Dict[str, bool] = {}

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = {
                executor.submit(self.download_file, file_info): file_info.filename
                for file_info in dataset.files
            }

            for future in as_completed(futures):
                filename = futures[future]
                try:
                    success = future.result()
                    results[filename] = success
                except Exception as e:
                    logger.error(f"Exception downloading {filename}: {e}")
                    results[filename] = False

        return results

    def download_all(self, datasets: List[DatasetInfo]) -> Dict[str, Dict[str, bool]]:
        """
        Downloads all specified datasets.
        """
        all_results = {}
        for dataset in datasets:
            all_results[dataset.name] = self.download_dataset(dataset)
        return all_results
