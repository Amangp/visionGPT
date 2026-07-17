# Contributing Guidelines

We welcome contributions to VisionGPT v3. This document outlines coding styles, testing standards, branching strategies, and our pull request process.

---

## 1. Coding Standards

To maintain code quality, please adhere to these standards:

### A. Style compliance
* Follow **PEP 8** style guidelines.
* Run a linter (e.g. `flake8` or `black`) before committing changes.
* Use 4 spaces for indentation (no tab characters).

### B. Type Hinting
* All new classes, methods, and functions should use type hints from the `typing` module:
  ```python
  def compute_accuracy(predictions: list[str], references: list[str]) -> float:
      ...
  ```

### C. Documentation
* Document modules, classes, and methods using **Google-style docstrings**:
  ```python
  def evaluate_cider(candidate: str, references: list[str]) -> float:
      """Calculate CIDEr metric consensus score.

      Args:
          candidate: Predicted string.
          references: List of ground-truth strings.

      Returns:
          Float score representing consensus.
      """
  ```

---

## 2. Unit Testing & Verification

All new features or bug fixes must include unit tests.

* Store test scripts in the same package directory (e.g. `evaluation/test_evaluation.py`).
* Ensure all tests pass before proposing a pull request:
  ```bash
  python -m unittest discover -s . -p "test_*.py"
  ```

---

## 3. Branching & Git Commit Conventions

### A. Branch Naming
Create a feature branch from the `main` branch before editing files:
* For features: `feature/your-feature-name`
* For bug fixes: `bugfix/issue-description`

### B. Commits Message Formatting
Write clear, concise commit messages:
```text
[feature/monitoring] Add TF-GPU peak VRAM logging to training callbacks

- Integrate get_tf_gpu_memory_info helper.
- Update ExperimentLogger metrics.json output logs.
```

---

## 4. Pull Request (PR) Workflow

1. Fork the repository and create your feature branch.
2. Implement your changes and add matching unit tests.
3. Verify that the test suite passes locally.
4. Push your changes to your fork and open a Pull Request.
5. Provide a summary of the changes and link any related issues in the PR description.
6. A maintainer will review your code. Address any feedback before merge.
