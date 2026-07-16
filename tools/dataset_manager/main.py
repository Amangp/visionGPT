"""
Main CLI entrypoint for the VisionGPT Dataset Manager.
Supports downloading, extracting, verifying, cleaning, and status reporting of datasets.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add workspace root to sys.path to enable absolute imports of tools package
sys.path.append(str(Path(__file__).resolve().parents[2]))

from typing import List, Set

from tools.dataset_manager.config import DATASET_ROOT
from tools.dataset_manager.datasets import DATASETS, DatasetInfo
from tools.dataset_manager.downloader import Downloader
from tools.dataset_manager.extractor import Extractor
from tools.dataset_manager.utils import setup_logger, format_size
from tools.dataset_manager.verifier import Verifier

# Setup logger to save logs inside the workspace
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
logger = setup_logger(WORKSPACE_ROOT / "logs")


def get_dataset_disk_usage(path: Path) -> int:
    """
    Recursively computes the disk usage of a path in bytes.
    """
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def print_status(datasets: List[DatasetInfo]) -> None:
    """
    Reports downloaded, extracted, missing, and corrupted status for all datasets.
    """
    print("\n==========================================")
    print("VISIONGPT DATASET MANAGER STATUS REPORT")
    print("==========================================")
    
    verifier = Verifier(verify_hash=False)  # Quick check without expensive hashing

    downloaded_datasets: List[str] = []
    missing_datasets: List[str] = []
    corrupted_datasets: List[str] = []
    extracted_datasets: List[str] = []

    for dataset in datasets:
        # Check download status
        all_downloaded = True
        any_downloaded = False
        corrupted = False

        for f in dataset.files:
            file_path = DATASET_ROOT / f.sub_dir / f.filename
            if file_path.exists():
                any_downloaded = True
                actual_size = file_path.stat().st_size
                if f.expected_size is not None and actual_size != f.expected_size:
                    corrupted = True
            else:
                all_downloaded = False

        # Check extraction status
        is_extracted, _ = verifier.verify_structure(dataset)

        if is_extracted:
            extracted_datasets.append(dataset.name)

        if all_downloaded and not corrupted:
            downloaded_datasets.append(dataset.name)
        elif corrupted:
            corrupted_datasets.append(dataset.name)
        elif not any_downloaded:
            missing_datasets.append(dataset.name)
        else:
            # Partially downloaded files
            corrupted_datasets.append(f"{dataset.name} (Incomplete)")

    # Print summary
    print(f"Downloaded archives: {', '.join(downloaded_datasets) if downloaded_datasets else 'None'}")
    print(f"Extracted datasets:  {', '.join(extracted_datasets) if extracted_datasets else 'None'}")
    print(f"Missing datasets:    {', '.join(missing_datasets) if missing_datasets else 'None'}")
    if corrupted_datasets:
        print(f"Corrupted/Incomplete: \033[91m{', '.join(corrupted_datasets)}\033[0m")
    else:
        print("Corrupted/Incomplete: None")

    total_bytes = get_dataset_disk_usage(DATASET_ROOT)
    print(f"Total directory size: {format_size(total_bytes)}")
    print("==========================================\n")


def clean_archives(datasets: List[DatasetInfo]) -> None:
    """
    Deletes downladed archive files to save disk space after successful extraction.
    """
    logger.info("Cleaning downloaded archives...")
    for dataset in datasets:
        cleaned_any = False
        for f in dataset.files:
            archive_path = DATASET_ROOT / f.sub_dir / f.filename
            if archive_path.exists():
                # Verify it is actually an archive file before deleting
                name = f.filename.lower()
                if name.endswith((".zip", ".tar", ".tar.gz", ".tgz")):
                    logger.info(f"Removing archive file: {archive_path}")
                    archive_path.unlink()
                    cleaned_any = True
        if cleaned_any:
            print(f"Cleaned archives for {dataset.name}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Production-grade Dataset Manager for VisionGPT v4."
    )

    # Dataset selection flags
    parser.add_argument("--all", action="store_true", help="Select all datasets")
    parser.add_argument("--coco", action="store_true", help="Select COCO 2017 dataset")
    parser.add_argument("--textvqa", action="store_true", help="Select TextVQA dataset")
    parser.add_argument("--gqa", action="store_true", help="Select GQA dataset")
    parser.add_argument("--visualgenome", action="store_true", help="Select Visual Genome dataset")

    # Command flags
    parser.add_argument("--verify", action="store_true", help="Verify dataset files integrity and structure")
    parser.add_argument("--extract", action="store_true", help="Extract downloaded dataset zip/tar files")
    parser.add_argument("--clean", action="store_true", help="Clean downloaded archive files to free disk space")
    parser.add_argument("--status", action="store_true", help="Report status of all datasets")

    args = parser.parse_args()

    # Determine selected datasets
    selected_datasets: List[DatasetInfo] = []

    if args.all:
        selected_datasets = list(DATASETS.values())
    else:
        if args.coco:
            selected_datasets.append(DATASETS["coco"])
        if args.textvqa:
            selected_datasets.append(DATASETS["textvqa"])
        if args.gqa:
            selected_datasets.append(DATASETS["gqa"])
        if args.visualgenome:
            selected_datasets.append(DATASETS["visualgenome"])

    # If no dataset flags are specified, default to selecting all datasets for execution
    if not selected_datasets:
        selected_datasets = list(DATASETS.values())

    # Handle status command
    if args.status:
        print_status(selected_datasets)
        return

    # Handle clean command
    if args.clean:
        clean_archives(selected_datasets)
        return

    # Default flow if no specific execution command is specified:
    # 1. Download
    # 2. Extract
    # 3. Verify
    is_specific_command = args.verify or args.extract

    run_download = not is_specific_command
    run_extract = args.extract or not is_specific_command
    run_verify = args.verify or not is_specific_command

    # Execution Step 1: Download
    if run_download:
        print("\n>>> STEP 1: Downloading selected datasets...")
        downloader = Downloader()
        for dataset in selected_datasets:
            print(f"\nDownloading files for {dataset.name}...")
            downloader.download_dataset(dataset)

    # Execution Step 2: Extract
    if run_extract:
        print("\n>>> STEP 2: Extracting dataset archives...")
        extractor = Extractor()
        for dataset in selected_datasets:
            print(f"\nExtracting archives for {dataset.name}...")
            extractor.extract_dataset(dataset)

    # Execution Step 3: Verify
    if run_verify:
        print("\n>>> STEP 3: Verifying dataset files...")
        verifier = Verifier()
        for dataset in selected_datasets:
            verifier.audit_dataset(dataset)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user. Exiting.")
        sys.exit(1)
