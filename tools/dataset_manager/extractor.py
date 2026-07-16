"""
Extraction module for unzipping and untarring dataset archives with progress bars.
"""

import logging
import os
import tarfile
import zipfile
from pathlib import Path
from typing import Union
from tqdm import tqdm

from tools.dataset_manager.config import (
    DELETE_ARCHIVES_AFTER_EXTRACTION,
    DATASET_ROOT
)
from tools.dataset_manager.datasets import DatasetFile, DatasetInfo

logger = logging.getLogger("DatasetManager.Extractor")


class Extractor:
    """
    Handles file extraction for zip, tar, and tar.gz files.
    """

    def __init__(
        self,
        delete_archives: bool = DELETE_ARCHIVES_AFTER_EXTRACTION
    ) -> None:
        self.delete_archives = delete_archives

    def _get_archive_format(self, file_path: Path) -> str:
        """
        Determines the format of the archive based on its extension.
        """
        name = file_path.name.lower()
        if name.endswith(".zip"):
            return "zip"
        elif name.endswith(".tar.gz") or name.endswith(".tgz"):
            return "tar.gz"
        elif name.endswith(".tar"):
            return "tar"
        return "unknown"

    def extract_file(self, file_info: DatasetFile) -> bool:
        """
        Extracts a single archive file to the specified sub-directory.
        """
        archive_path = DATASET_ROOT / file_info.sub_dir / file_info.filename
        if not archive_path.exists():
            logger.error(f"Archive not found for extraction: {archive_path}")
            return False

        fmt = self._get_archive_format(archive_path)
        if fmt == "unknown":
            logger.info(f"Skipping extraction for {file_info.filename} (not an archive).")
            return True

        # Determine extraction directory
        # Most zips are extracted into the parent folder (DATASET_ROOT / sub_dir)
        dest_dir = DATASET_ROOT / file_info.sub_dir
        dest_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Extracting {archive_path.name} to {dest_dir}...")

        try:
            if fmt == "zip":
                with zipfile.ZipFile(archive_path) as z:
                    members = z.infolist()
                    # Show progress bar based on number of files
                    for member in tqdm(
                        members,
                        desc=f"Unzipping {archive_path.name}",
                        unit="file",
                        leave=False
                    ):
                        z.extract(member, path=dest_dir)

            elif fmt in ["tar", "tar.gz"]:
                mode = "r:gz" if fmt == "tar.gz" else "r"
                with tarfile.open(archive_path, mode) as t:
                    members = t.getmembers()
                    for member in tqdm(
                        members,
                        desc=f"Untarring {archive_path.name}",
                        unit="file",
                        leave=False
                    ):
                        t.extract(member, path=dest_dir, filter="data")  # Python 3.12+ safe extraction filter

            logger.info(f"Successfully extracted {archive_path.name}.")

            if self.delete_archives:
                logger.info(f"Deleting archive file: {archive_path}")
                archive_path.unlink()

            return True

        except Exception as e:
            logger.error(f"Failed to extract {archive_path.name}: {e}")
            return False

    def extract_dataset(self, dataset: DatasetInfo) -> bool:
        """
        Extracts all archive files belonging to a dataset.
        """
        logger.info(f"Extracting all archives for dataset: {dataset.name}")
        success = True
        for file_info in dataset.files:
            fmt = self._get_archive_format(Path(file_info.filename))
            if fmt != "unknown":
                result = self.extract_file(file_info)
                if not result:
                    success = False
        return success
