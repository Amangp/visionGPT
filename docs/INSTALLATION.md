# Installation Manual

This document details the step-by-step setup required to run VisionGPT v3. It covers python virtual environments, CPU/GPU TensorFlow configuration, CUDA Toolkit/cuDNN configurations, dataset tooling setup, and sanity verification steps.

---

## 1. System Requirements

* **Operating System:** Windows 10/11, Ubuntu 20.04/22.04 LTS, or macOS 12+ (Apple Silicon supported).
* **Python:** Version 3.10 or 3.11 (Python 3.12/3.13 are supported but require matching TensorFlow/NumPy versions).
* **RAM:** Minimum 16 GB system memory (32 GB recommended for dataset processing).
* **GPU (Optional):** NVIDIA GPU with CUDA Compute Capability 7.0 or higher (e.g. RTX 20/30/40 series, A100/V100) and at least 8 GB VRAM.

---

## 2. Python Environment Setup

We recommend managing dependencies using [Conda](https://docs.conda.io/en/latest/).

```bash
# Create a virtual environment for VisionGPT
conda create -n visiongpt python=3.10 -y

# Activate the environment
conda activate visiongpt

# Install base dependencies
pip install -r requirements.txt
```

### Core Dependencies in `requirements.txt`:
```text
requests>=2.31.0
tqdm>=4.66.1
numpy>=1.24.3
tensorflow>=2.12.0
nltk>=3.8.1
rouge-score>=0.1.2
matplotlib>=3.7.1
psutil>=5.9.5
```

---

## 3. TensorFlow & CUDA Installation

### A. CPU-Only Installation
If you do not have an NVIDIA GPU, install the standard TensorFlow package:
```bash
pip install tensorflow>=2.12.0
```

### B. NVIDIA GPU (CUDA) Installation
To enable GPU acceleration, install TensorFlow with matching CUDA and cuDNN libraries. Follow this matrix for compatibility:

| TensorFlow | CUDA Toolkit | cuDNN |
| :--- | :---: | :---: |
| **v2.12.0 / v2.13.0** | 11.8 | 8.6 |
| **v2.14.0 / v2.15.0** | 12.2 | 8.9 |
| **v2.16.0 / v2.20.0** | 12.3 / 12.5 | 8.9 / 9.0 |

#### CUDA & cuDNN Installation Steps:
1. Download and install the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) matching your TensorFlow version.
2. Download [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) binaries, extract them, and copy the files in `bin/`, `include/`, and `lib/` into the corresponding directories of your CUDA installation path.
3. Configure the environment variables (Windows Example):
   ```cmd
   set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin;%PATH%
   set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp;%PATH%
   ```
4. Verify CUDA compiler works:
   ```bash
   nvcc --version
   ```

---

## 4. Dataset Manager Setup

VisionGPT uses a concurrent downloader CLI tool to retrieve raw dataset archives. To verify it works, run:

```bash
# Monitor dataset status
python tools/dataset_manager/main.py --status
```

*For more details on datasets directory mapping, refer to the [Dataset Reference](DATASETS.md).*

---

## 5. Verification Checks

After installing the packages and setting up the folders, run the verification scripts for the submodules to confirm correct operations:

```bash
# 1. Verify Evaluation module
python -m evaluation.test_evaluation

# 2. Verify Benchmarking module
python -m benchmark.test_benchmark

# 3. Verify Monitoring module
python -m monitoring.test_monitoring

# 4. Verify Architecture diagram generators
python -m visualization.test_visualization
```

If all tests display `OK`, your installation is complete and ready.

---

## 6. Troubleshooting FAQs

### A. "ModuleNotFoundError: No module named 'tensorflow'"
This happens if you are running Python inside a different conda environment than the one where you installed the packages. Make sure to activate the environment first:
```bash
conda activate visiongpt
```

### B. "DLL load failed: The specified module could not be found."
On Windows, this indicates that the CUDA or cuDNN `.dll` files are missing from the system `PATH`. Re-check your CUDA installation path and environment variables configuration.

### C. "oneDNN custom operations are on..."
This is a standard informational warning from TensorFlow. You can ignore it, or disable it by setting the environment variable:
```bash
# Linux
export TF_ENABLE_ONEDNN_OPTS=0

# Windows Powershell
$env:TF_ENABLE_ONEDNN_OPTS=0
```
