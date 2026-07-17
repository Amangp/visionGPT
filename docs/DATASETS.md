# Datasets & Preprocessing Reference

VisionGPT v3 is trained and validated on four core datasets. This document details the specifications of each dataset, their folder structures, CLI tools for concurrent downloads, and data verification steps.

---

## 1. Supported Datasets

### A. MS COCO 2017
Used for general **Image Captioning** evaluations.
* **Images:** 118k training, 5k validation.
* **Annotations:** Captions describing images in natural language.
* **Features Extracted:** Spatial grid features of size `(7, 7, 1280)` from RGB inputs.

### B. TextVQA
Used for **OCR-aware Question Answering** evaluations.
* **Task:** Requires models to read text present within images to answer questions.
* **Annotations:** Questions, multiple ground-truth answers, and OCR text bounding boxes.

### C. GQA
Used for **Visual Question Answering (VQA)** and spatial reasoning.
* **Task:** Answering compositional questions over real-world images.
* **Annotations:** Balanced question-answer pairs and image scene graphs.

### D. Visual Genome
Used for visual relationships and **grounded QA**.
* **Task:** Visual scene understanding, object detection, region descriptions, and question-answering.

---

## 2. Directory Layout Mappings

By default, the Dataset Manager stores all raw archives and extracted assets in a directory named `Dataset` under the workspace root:

```
Dataset/
├── coco/
│   ├── train2017/                  # Extracted Train Images
│   ├── val2017/                    # Extracted Val Images
│   └── annotations/                # captions_train2017.json, etc.
├── textvqa/
│   ├── train_val_images/           # Extracted Images
│   ├── TextVQA_0.5_train.json      # Train annotations
│   └── TextVQA_0.5_val.json        # Val annotations
├── gqa/
│   ├── images/                     # Extracted Images
│   └── questions1.2/               # train_balanced_questions.json, etc.
└── visualgenome/
    ├── VG_100K/                    # Extracted Images
    ├── VG_100K_2/                  # Extracted Images Part 2
    ├── question_answers.json       # QA pairs
    └── region_descriptions.json    # Grounded descriptions
```

---

## 3. Dataset Manager CLI Guide

We provide a concurrent CLI downloader script in `tools/dataset_manager/main.py`. This script uses thread pools to accelerate downloads, supports transfer resumption via HTTP `Range` headers, extracts ZIP/TAR archives automatically, and validates folder structures.

### Command Guide:

#### 1. Monitor Current Download Status
Displays downloaded files, extraction percentages, and total dataset footprint on disk.
```bash
python tools/dataset_manager/main.py --status
```

#### 2. Run Complete Pipeline (Download, Extract, & Verify)
Downloads all selected datasets, extracts them, and runs file structures check.
```bash
python tools/dataset_manager/main.py --all
```

#### 3. Select Specific Datasets
You can filter dataset downloads by using specific flags:
* `--coco`: Only process COCO 2017.
* `--textvqa`: Only process TextVQA.
* `--gqa`: Only process GQA.
* `--visualgenome`: Only process Visual Genome.

Example:
```bash
# Process GQA and TextVQA only
python tools/dataset_manager/main.py --gqa --textvqa
```

#### 4. Audit & Check Integrity Checksums
Performs size audits, directory listings, and SHA256 checksum checks to detect corrupted archives.
```bash
python tools/dataset_manager/main.py --verify
```

#### 5. Delete Archive Archives Post-Extraction
Frees up disk space by deleting raw `.zip` and `.tar` files once extracted:
```bash
python tools/dataset_manager/main.py --clean
```

---

## 4. Expected Footprints & Sizes

Ensure you have enough space before triggering download loops. The table below lists the archive sizes and uncompressed sizes:

| Dataset | Archive Name | Archive Size | Extracted Size |
| :--- | :--- | :---: | :---: |
| **COCO 2017** | `train2017.zip` | 18.0 GB | 19.3 GB |
| | `val2017.zip` | 778 MB | 818 MB |
| | `annotations_trainval2017.zip` | 241 MB | 791 MB |
| **TextVQA** | `train_val_images.zip` | 6.8 GB | 7.2 GB |
| | `textvqa_annotations.zip` | 15 MB | 45 MB |
| **GQA** | `images.zip` | 20.3 GB | 22.1 GB |
| | `questions.zip` | 312 MB | 1.1 GB |
| **Visual Genome** | `images.zip` (Part 1 & 2) | 14.2 GB | 15.6 GB |
| | `annotations.zip` | 890 MB | 2.3 GB |
