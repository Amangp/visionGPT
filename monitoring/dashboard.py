"""HTML Dashboard generator for VisionGPT training runs.

Compiles training progression metrics, hardware diagnostics, and visual charts
into a responsive, high-aesthetic HTML dashboard in `reports/dashboard.html`.
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def generate_html_dashboard(experiment_logger: Any, reports_dir: str) -> None:
    """Generate a responsive, premium HTML training dashboard.

    Args:
        experiment_logger: The active ExperimentLogger instance.
        reports_dir: Directory where the dashboard file should be written.
    """
    os.makedirs(reports_dir, exist_ok=True)
    dashboard_path = os.path.join(reports_dir, "dashboard.html")

    metadata_raw = experiment_logger.metadata
    from dataclasses import asdict
    metadata = asdict(metadata_raw)
    records = experiment_logger.history.records
    best_metrics = experiment_logger._calculate_best_metrics()

    if not records:
        logger.debug("No records to populate the HTML dashboard.")
        return

    latest_record = records[-1]
    
    # Format elapsed duration
    duration_sec = time_elapsed = experiment_logger._calculate_best_metrics().get(
        "average_epoch_time_sec", 0.0
    ) * len(records)
    hours = int(duration_sec // 3600)
    minutes = int((duration_sec % 3600) // 60)
    seconds = int(duration_sec % 60)
    duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    # Build the metric card values
    latest_loss = latest_record.get("loss", 0.0)
    latest_val_loss = latest_record.get("validation_loss", 0.0)
    val_loss_str = f"{latest_val_loss:.4f}" if latest_val_loss is not None else "N/A"
    
    best_bleu = best_metrics.get("best_bleu_4")
    best_bleu_str = f"{best_bleu:.4f}" if best_bleu is not None else "N/A"
    
    best_cider = best_metrics.get("best_cider")
    best_cider_str = f"{best_cider:.4f}" if best_cider is not None else "N/A"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VisionGPT v3 - Training Dashboard</title>
    <!-- Google Fonts: Inter -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0f111a;
            --card-bg: rgba(22, 28, 45, 0.7);
            --border-color: rgba(255, 255, 255, 0.08);
            --accent-color: #4f46e5;
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --green: #10b981;
            --red: #ef4444;
            --yellow: #f59e0b;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 2rem;
            line-height: 1.5;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        /* Header */
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid var(--border-color);
        }}

        h1 {{
            font-size: 2rem;
            font-weight: 800;
            background: linear-gradient(90deg, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .status-badge {{
            background: rgba(16, 185, 129, 0.15);
            color: var(--green);
            padding: 0.5rem 1rem;
            border-radius: 9999px;
            font-size: 0.875rem;
            font-weight: 600;
            border: 1px solid rgba(16, 185, 129, 0.25);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .status-dot {{
            width: 8px;
            height: 8px;
            background-color: var(--green);
            border-radius: 50%;
            animation: pulse 1.5s infinite;
        }}

        @keyframes pulse {{
            0% {{ transform: scale(0.95); opacity: 0.5; }}
            50% {{ transform: scale(1.2); opacity: 1; }}
            100% {{ transform: scale(0.95); opacity: 0.5; }}
        }}

        /* Top Cards Grid */
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}

        .card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            backdrop-filter: blur(10px);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            transition: transform 0.2s, box-shadow 0.2s;
        }}

        .card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.3);
            border-color: rgba(255, 255, 255, 0.15);
        }}

        .card-title {{
            font-size: 0.875rem;
            color: var(--text-secondary);
            font-weight: 500;
            margin-bottom: 0.5rem;
        }}

        .card-value {{
            font-size: 1.8rem;
            font-weight: 700;
        }}

        /* Main Workspace Grid */
        .workspace-grid {{
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 2rem;
        }}

        @media (max-width: 1024px) {{
            .workspace-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        /* Charts section */
        .plots-panel {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 2rem;
        }}

        .plot-container {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}

        .plot-title {{
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 1rem;
            align-self: flex-start;
        }}

        .plot-img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }}

        /* Config Sidebar */
        .sidebar {{
            display: flex;
            flex-direction: column;
            gap: 2rem;
        }}

        .sidebar-section {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
        }}

        .sidebar-title {{
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 1.2rem;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.5rem;
        }}

        .meta-list {{
            list-style: none;
            display: flex;
            flex-direction: column;
            gap: 0.8rem;
        }}

        .meta-item {{
            display: flex;
            justify-content: space-between;
            font-size: 0.875rem;
        }}

        .meta-label {{
            color: var(--text-secondary);
        }}

        .meta-value {{
            font-weight: 600;
            text-align: right;
            max-width: 180px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>VisionGPT v3 Training Dashboard</h1>
                <p style="color: var(--text-secondary); font-size: 0.875rem; margin-top: 0.25rem;">
                    Experiment: <strong style="color: var(--text-primary);">{metadata.get('experiment_name')}</strong>
                </p>
            </div>
            <div class="status-badge">
                <div class="status-dot"></div>
                <span>LOGGING ACTIVE</span>
            </div>
        </header>

        <!-- Main Stats Overview -->
        <div class="metrics-grid">
            <div class="card">
                <span class="card-title">Completed Epochs</span>
                <span class="card-value">{latest_record.get('epoch', 0)} / {metadata.get('epochs')}</span>
            </div>
            <div class="card">
                <span class="card-title">Training Loss</span>
                <span class="card-value" style="color: #ff8080;">{latest_loss:.4f}</span>
            </div>
            <div class="card">
                <span class="card-title">Validation Loss</span>
                <span class="card-value" style="color: #80b3ff;">{val_loss_str}</span>
            </div>
            <div class="card">
                <span class="card-title">Peak BLEU-4</span>
                <span class="card-value" style="color: var(--green);">{best_bleu_str}</span>
            </div>
            <div class="card">
                <span class="card-title">Peak CIDEr</span>
                <span class="card-value" style="color: var(--yellow);">{best_cider_str}</span>
            </div>
        </div>

        <!-- Main Content Area -->
        <div class="workspace-grid">
            <!-- Left Panel: Metrics Visualization Plots -->
            <div class="plots-panel">
                <div class="plot-container">
                    <span class="plot-title">Loss Convergence Curves</span>
                    <img class="plot-img" src="../plots/loss.png" alt="Loss Plot">
                </div>
                <div class="plot-container">
                    <span class="plot-title">Language Generation Metrics (BLEU)</span>
                    <img class="plot-img" src="../plots/bleu.png" alt="BLEU Plot">
                </div>
                <div class="plot-container">
                    <span class="plot-title">Consensus Image Description (CIDEr)</span>
                    <img class="plot-img" src="../plots/cider.png" alt="CIDEr Plot">
                </div>
                <div class="plot-container">
                    <span class="plot-title">VQA Alignment & Accuracy Profiles</span>
                    <img class="plot-img" src="../plots/accuracy.png" alt="Accuracy Plot">
                </div>
                <div class="plot-container">
                    <span class="plot-title">Memory Footprint (VRAM / RAM)</span>
                    <img class="plot-img" src="../plots/memory.png" alt="Memory Plot">
                </div>
            </div>

            <!-- Right Sidebar: Configuration, Hardware, Vitals -->
            <div class="sidebar">
                <!-- Training Config Info -->
                <div class="sidebar-section">
                    <h3 class="sidebar-title">Training Configurations</h3>
                    <ul class="meta-list">
                        <li class="meta-item">
                            <span class="meta-label">Dataset</span>
                            <span class="meta-value" title="{metadata.get('dataset')}">{metadata.get('dataset')}</span>
                        </li>
                        <li class="meta-item">
                            <span class="meta-label">Batch Size</span>
                            <span class="meta-value">{metadata.get('batch_size')}</span>
                        </li>
                        <li class="meta-item">
                            <span class="meta-label">Optimizer</span>
                            <span class="meta-value">{metadata.get('optimizer')}</span>
                        </li>
                        <li class="meta-item">
                            <span class="meta-label">Learning Rate</span>
                            <span class="meta-value">{metadata.get('learning_rate')}</span>
                        </li>
                        <li class="meta-item">
                            <span class="meta-label">Mixed Precision</span>
                            <span class="meta-value">{metadata.get('mixed_precision')}</span>
                        </li>
                        <li class="meta-item">
                            <span class="meta-label">Elapsed Time</span>
                            <span class="meta-value">{duration_str}</span>
                        </li>
                    </ul>
                </div>

                <!-- Hardware section -->
                <div class="sidebar-section">
                    <h3 class="sidebar-title">Hardware Profile</h3>
                    <ul class="meta-list">
                        <li class="meta-item">
                            <span class="meta-label">Host OS</span>
                            <span class="meta-value" title="{metadata.get('os')}">{metadata.get('os')}</span>
                        </li>
                        <li class="meta-item">
                            <span class="meta-label">CPU Type</span>
                            <span class="meta-value" title="{metadata.get('cpu')}">{metadata.get('cpu')}</span>
                        </li>
                        <li class="meta-item">
                            <span class="meta-label">GPU Unit</span>
                            <span class="meta-value" title="{metadata.get('gpu')}">{metadata.get('gpu')}</span>
                        </li>
                        <li class="meta-item">
                            <span class="meta-label">RAM size</span>
                            <span class="meta-value">{metadata.get('ram')} GB</span>
                        </li>
                        <li class="meta-item">
                            <span class="meta-label">TensorFlow</span>
                            <span class="meta-value">v{metadata.get('tensorflow_version')}</span>
                        </li>
                    </ul>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

    try:
        with open(dashboard_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.debug("Successfully updated HTML training dashboard at %s", dashboard_path)
    except Exception as e:
        logger.error("Failed to write HTML dashboard: %s", e)
