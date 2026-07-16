# VisionGPT Dataset Manager

A production-quality, high-performance dataset management tool built in Python 3.11+ for download, extraction, verification, and status monitoring of deep learning datasets for VisionGPT.

## Features

- **Concurrent Downloads:** Uses a `ThreadPoolExecutor` to retrieve multiple files in parallel.
- **Resumable Transfers:** Supports resuming interrupted downloads automatically using HTTP `Range` headers.
- **Automatic Retries:** Performs connection retry logic with exponential backoff on connection errors or bad response statuses.
- **Auto-Extraction:** Identifies `.zip`, `.tar`, and `.tar.gz` files and extracts them cleanly with recursive file listings and inline progress tracking.
- **Integrity Validation:** Validates expected directories, exact file size, file existence, and computed SHA256 hashes.
- **Interactive UI:** Provides distinct concurrent progress bars (with speed, elapsed time, percentage, and ETA) for downloads using `tqdm`.
- **Structured Logging:** Directs comprehensive logs to `logs/dataset_manager.log` and status updates directly to console.

---

## Supported Datasets

1. **COCO 2017**
   - Training images (`train2017.zip`)
   - Validation images (`val2017.zip`)
   - Annotation files (`annotations_trainval2017.zip`)
2. **TextVQA**
   - Training annotations
   - Validation annotations
   - Combined train/val image archive (`train_val_images.zip`)
3. **GQA**
   - Balanced questions
   - Image archives
4. **Visual Genome**
   - Images Part 1 & 2
   - Region descriptions
   - Scene graphs
   - Question-answers

---

## Installation

Install dependencies using Python 3.11+:

```bash
pip install -r requirements.txt
```

---

## CLI Usage Guide

Run the CLI using flags corresponding to the action you wish to perform:

### 1. General Datasets Configuration
By default, commands apply to all datasets. You can select specific datasets using:
- `--all` (Select all)
- `--coco` (Select COCO 2017)
- `--textvqa` (Select TextVQA)
- `--gqa` (Select GQA)
- `--visualgenome` (Select Visual Genome)

### 2. Actions

#### Monitor Current Status
Displays downloaded directories, extracted datasets, corrupted zip files, and total dataset size:
```bash
python tools/dataset_manager/main.py --status
```

#### Run All Operations (Download, Extract & Verify)
Downloads selected datasets, extracts them, and runs file structure and size checks:
```bash
python tools/dataset_manager/main.py --all
```

#### Verify Datasets
Manually run audits on the datasets (existence, sizes, directories, SHA256 hashes):
```bash
python tools/dataset_manager/main.py --verify
```

#### Extract Archives
Manually trigger extraction of archives:
```bash
python tools/dataset_manager/main.py --extract
```

#### Clean Archives
Delete the raw `.zip` or `.tar` archive files to free up disk space after extraction:
```bash
python tools/dataset_manager/main.py --clean
```

---

## Configuration Settings

You can customize execution constants within [tools/dataset_manager/config.py](tools/dataset_manager/config.py):

* `DATASET_ROOT`: Destination path for datasets storage (defaults to a directory named `Dataset` in workspace root).
* `MAX_THREADS`: Max threads pool size for parallel downloads.
* `RETRY_COUNT`: Number of reconnection retries before failing.
* `TIMEOUT`: Request connection timeout in seconds.
* `DELETE_ARCHIVES_AFTER_EXTRACTION`: Set to `True` to auto-delete zip archives after successful extraction.
* `VERIFY_HASH`: Toggle SHA256 validation.
