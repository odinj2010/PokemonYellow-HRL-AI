# export_metrics.py
# (FIXED: Added --lesson argument and corrected hyperparameter path logic)

import os
import re
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import json
import platform
import subprocess
import argparse # (NEW) Import argparse

# (NEW) --- Configuration via Arguments ---
parser = argparse.ArgumentParser(description="Export training metrics to Markdown.")
parser.add_argument('--lesson', type=str, default='exploration',
                    help='The name of the lesson to export metrics for (e.g., exploration, battle, manager, meta)')
args = parser.parse_args()

LESSON_NAME = args.lesson
LOG_DIR = f"logs/{LESSON_NAME}"
MODEL_DIR = f"models/{LESSON_NAME}"
OUTPUT_FILE = f"debug_metrics_summary_{LESSON_NAME}.md"
TENSORBOARD_ROOT = LOG_DIR

print(f"--- Exporting metrics for lesson: {LESSON_NAME} ---")
print(f"LOG_DIR: {LOG_DIR}")
print(f"MODEL_DIR: {MODEL_DIR}")

def get_git_info():
    """Retrieves the current Git branch and commit hash."""
    branch = "N/A"
    commit = "N/A"
    try:
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("utf-8").strip()
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception as e:
        print(f"Warning: Could not retrieve Git info: {e}")
    return branch, commit

def find_latest_run_path(log_dir):
    """Finds the path to the latest run within the main TensorBoard root by looking for event files."""
    if not os.path.isdir(log_dir):
        print(f"Error: Log directory not found at {log_dir}")
        return None

    all_run_paths = []

    # Check the log_dir itself
    if any(f.startswith("events.out.tfevents") for f in os.listdir(log_dir)):
        all_run_paths.append(log_dir)

    # Check subdirectories (This is where SB3 logs are)
    for entry in os.listdir(log_dir):
        full_path = os.path.join(log_dir, entry)
        if os.path.isdir(full_path):
            if any(f.startswith("events.out.tfevents") for f in os.listdir(full_path)):
                all_run_paths.append(full_path)

    if not all_run_paths:
        print(f"No TensorBoard event files found in {log_dir} or its immediate subdirectories.")
        return None

    def get_timestamp_from_path(path):
        event_files = [f for f in os.listdir(path) if f.startswith("events.out.tfevents")]
        if not event_files:
            return 0 
        
        match = re.search(r"tfevents\.(\d+)", event_files[0])
        return int(match.group(1)) if match else 0

    all_run_paths.sort(key=get_timestamp_from_path, reverse=True)
    
    latest_run_path = all_run_paths[0]
    print(f"Found latest run at: {latest_run_path}")
    return latest_run_path

def calculate_trend(values, improvement_threshold=0.05, higher_is_better=True):
    """Calculates a simple trend based on the first and last 10% of values."""
    if len(values) < 10:
        return "N/A (not enough data)"
    
    numeric_values = [v for v in values if isinstance(v, (int, float))]
    if not numeric_values:
        return "N/A (no numeric data)"

    first_10_percent_idx = max(1, len(numeric_values) // 10)
    last_10_percent_idx = max(1, len(numeric_values) // 10)

    first_avg = np.mean(numeric_values[:first_10_percent_idx])
    last_avg = np.mean(numeric_values[-last_10_percent_idx:])

    if higher_is_better:
        if last_avg > first_avg * (1 + improvement_threshold):
            return "📈 Improving"
        elif last_avg < first_avg * (1 - improvement_threshold):
            return "📉 Degrading"
        else:
            return "↔️ Stable"
    else: # Lower is better (e.g., loss)
        if last_avg < first_avg * (1 - improvement_threshold):
            return "📈 Improving" 
        elif last_avg > first_avg * (1 + improvement_threshold):
            return "📉 Degrading"
        else:
            return "↔️ Stable"

KEY_SCALARS_CONFIG = {
    'rollout/ep_rew_mean': {'higher_is_better': True},
    'train/loss': {'higher_is_better': False},
    'train/policy_loss': {'higher_is_better': False},
    'train/value_loss': {'higher_is_better': False},
    'train/entropy_loss': {'higher_is_better': False},
}

def extract_scalars(log_path):
    """Extracts all scalar data from a TensorBoard event file..."""
    if not log_path or not os.path.isdir(log_path):
        print(f"Warning: TensorBoard log path is not a valid directory: {log_path}")
        return {}, None, None
    
    try:
        acc = EventAccumulator(log_path)
        acc.Reload()
    except Exception as e:
        print(f"Error loading TensorBoard data from {log_path}: {e}")
        return {}, None, None

    data = {}
    start_time = None
    end_time = None

    for tag in acc.Tags()['scalars']:
        events = acc.Scalars(tag)
        if events:
            values = [e.value for e in events]
            last_event = events[-1]
            
            if start_time is None or events[0].wall_time < start_time:
                start_time = events[0].wall_time
            if end_time is None or last_event.wall_time > end_time:
                end_time = last_event.wall_time

            metric_data = {
                'Step (Last)': int(last_event.step),
                'Value (Last)': f"{last_event.value:.4f}",
            }
            
            if len(events) > 1:
                metric_data['Value (Start)'] = f"{events[0].value:.4f}"
                median_index = len(events) // 2
                metric_data['Value (Median)'] = f"{events[median_index].value:.4f}"
            else:
                metric_data['Value (Start)'] = metric_data['Value (Last)']
                metric_data['Value (Median)'] = metric_data['Value (Last)']
            
            if tag in KEY_SCALARS_CONFIG:
                metric_data['Trend'] = calculate_trend(values, higher_is_better=KEY_SCALARS_CONFIG[tag]['higher_is_better'])
            
            data[tag] = metric_data
    
    return data, start_time, end_time

def analyze_monitor_csv(log_dir):
    """Analyzes the monitor.csv file for episode statistics..."""
    # (FIXED) monitor.csv is directly in LOG_DIR, not the run_path
    monitor_path = os.path.join(LOG_DIR, "monitor.csv")
    if not os.path.exists(monitor_path):
        # (FIXED) Fallback for SB3, which puts it in the run_path
        monitor_path = os.path.join(log_dir, "monitor.csv")
        if not os.path.exists(monitor_path):
            return f"monitor.csv not found in {LOG_DIR} or {log_dir}."

    try:
        df = pd.read_csv(monitor_path, skiprows=1)
        if df.empty:
            return "monitor.csv is empty."
            
        df.columns = [col.strip() for col in df.columns] 

        stats = {
            "Total Episodes": len(df),
            "Mean Reward (Overall)": f"{df['r'].mean():.2f}",
            "Std Reward (Overall)": f"{df['r'].std():.2f}",
            "Max Reward (Overall)": f"{df['r'].max():.2f}",
            "Min Reward (Overall)": f"{df['r'].min():.2f}",
            "Mean Episode Length (Overall)": f"{df['l'].mean():.0f}",
        }
        
        if len(df) >= 10:
            first_10_percent = df['r'].head(max(1, len(df) // 10)).mean()
            last_10_percent = df['r'].tail(max(1, len(df) // 10)).mean()
            
            stats["Mean Reward (First 10%)"] = f"{first_10_percent:.2f}"
            stats["Mean Reward (Last 10%)"] = f"{last_10_percent:.2f}"
            
            if last_10_percent > first_10_percent * 1.05:
                stats["Reward Trend"] = "📈 Improving"
            elif last_10_percent < first_10_percent * 0.95:
                stats["Reward Trend"] = "📉 Degrading"
            else:
                stats["Reward Trend"] = "↔️ Stable"
        elif len(df) > 1:
            stats["Reward Trend"] = "Not enough data for trend (need >= 10 episodes)"
        else:
            stats["Reward Trend"] = "N/A"

        stats["Episode Length (25th Percentile)"] = f"{df['l'].quantile(0.25):.0f}"
        stats["Episode Length (50th Percentile / Median)"] = f"{df['l'].quantile(0.50):.0f}"
        stats["Episode Length (75th Percentile)"] = f"{df['l'].quantile(0.75):.0f}"
        
        summary_df = pd.DataFrame.from_dict(stats, orient='index', columns=['Value'])
        summary_df.index.name = 'Statistic'
        return summary_df.to_markdown()

    except Exception as e:
        return f"Error reading monitor.csv: {e}"

def get_requirements():
    """Reads requirements.txt and returns a formatted string..."""
    output = []
    try:
        with open("requirements.txt", "r") as f:
            output.append("### From requirements.txt:\n")
            output.extend([f"- `{line.strip()}`" for line in f if line.strip()])
    except FileNotFoundError:
        output.append("requirements.txt not found.")
    
    output.append("\n### From pip freeze:\n")
    try:
        pip_freeze_output = subprocess.check_output(["pip", "freeze"]).decode("utf-8")
        output.extend([f"- `{line.strip()}`" for line in pip_freeze_output.splitlines() if line.strip()])
    except Exception as e:
        output.append(f"Error running pip freeze: {e}")
        
    return "\n".join(output)

def get_python_version():
    """Returns the current Python version."""
    return platform.python_version()

def list_model_checkpoints():
    """Lists all model checkpoint files in MODEL_DIR."""
    if not os.path.isdir(MODEL_DIR):
        return f"Model directory not found at {MODEL_DIR}."
    
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith((".zip", ".pkl"))]
    
    if not model_files:
        return "No model checkpoints found."
    
    model_files.sort(key=lambda f: os.path.getmtime(os.path.join(MODEL_DIR, f)), reverse=True)
    
    formatted_list = "\n".join([f"- `{f}`" for f in model_files])
    return formatted_list

def extract_hyperparameters(run_path):
    """Extracts hyperparameters from a JSON file within the run directory."""
    # (FIXED) Look in the parent directory of the run_path
    parent_dir = os.path.dirname(run_path)
    hyperparams_path = os.path.join(parent_dir, "hyperparameters.json")
    
    if not os.path.exists(hyperparams_path):
        # (FIXED) Fallback to check the root log dir as well
        hyperparams_path = os.path.join(LOG_DIR, "hyperparameters.json")
        if not os.path.exists(hyperparams_path):
            return f"hyperparameters.json not found in {parent_dir} or {LOG_DIR}."
    
    try:
        with open(hyperparams_path, "r") as f:
            hyperparams = json.load(f)
            return "```json\n" + json.dumps(hyperparams, indent=2) + "\n```"
    except Exception as e:
        return f"Error reading hyperparameters.json: {e}"

def generate_markdown_report(metrics, monitor_stats, requirements, checkpoints, hyperparameters, run_path, git_info, start_time, end_time):
    """Generates a comprehensive Markdown report."""
    
    git_branch, git_commit = git_info
    training_duration_str = "N/A"
    if start_time and end_time:
        duration_seconds = end_time - start_time
        hours, remainder = divmod(duration_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        training_duration_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

    progress_metrics = {k: v for k, v in metrics.items() if k.startswith('progress/')}
    game_state_metrics = {k: v for k, v in metrics.items() if k.startswith('game_state/')}
    battle_metrics = {k: v for k, v in metrics.items() if k.startswith('battle/')}
    training_metrics = {k: v for k, v in metrics.items() if k.startswith(('rollout/', 'train/'))}
    
    def metrics_to_markdown(metric_dict, title):
        if not metric_dict:
            return f"No {title.lower()} found."
        df = pd.DataFrame.from_dict(metric_dict, orient='index')
        df.index.name = 'Metric Tag'
        columns_to_display = ['Value (Last)', 'Value (Start)', 'Value (Median)']
        if 'Trend' in df.columns:
            columns_to_display.append('Trend')
        df = df[columns_to_display]
        df.sort_index(inplace=True)
        return df.to_markdown()

    progress_table = metrics_to_markdown(progress_metrics, "Progress Metrics")
    game_state_table = metrics_to_markdown(game_state_metrics, "Game State Metrics")
    battle_table = metrics_to_markdown(battle_metrics, "Battle Metrics")
    training_table = metrics_to_markdown(training_metrics, "Training Metrics")

    report = f"""
# Training Debug Summary: {LESSON_NAME.upper()}

**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**Run Source:** `{run_path}`

---

## 1. Run Information

- **Git Branch:** `{git_branch}`
- **Git Commit:** `{git_commit}`
- **Training Start Time:** {pd.Timestamp(start_time, unit='s').strftime('%Y-%m-%d %H:%M:%S') if start_time else 'N/A'}
- **Training End Time:** {pd.Timestamp(end_time, unit='s').strftime('%Y-%m-%d %H:%M:%S') if end_time else 'N/A'}
- **Training Duration:** {training_duration_str}

---

## 2. Environment & Dependencies

- **Python Version:** `{get_python_version()}`

This run used the following core packages:
{requirements}

---

## 3. Episode Performance (`monitor.csv`)

Summary of agent's performance across all completed episodes.

{monitor_stats}

---

## 4. Hyperparameters

Extracted from `{os.path.join(LOG_DIR, "hyperparameters.json")}`.

{hyperparameters}

---

## 5. Model Checkpoints

Models saved during this training run can be found in `{MODEL_DIR}`.

{checkpoints}

---

## 6. Training Metrics

Snapshot of the last, first, and median values for key training metrics.

{training_table}

### Interpretation Guidance
- **`rollout/ep_rew_mean`**: The primary indicator of performance. Should trend UPWARD.
- **`train/loss`**: The overall training loss. Should trend DOWNWARD.
- **`train/policy_loss` & `train/value_loss`**: Components of the main loss. Should also trend DOWNWARD.
- **`train/entropy_loss`**: Represents exploration.

---

## 7. Game Progress Metrics

Custom metrics tracking the agent's progress in the game world.

{progress_table}

---

## 8. Game State Metrics

Low-level state information from the game environment.

{game_state_table}

---

## 9. Battle Metrics

Metrics related to in-game battles.

{battle_table}
"""
    return report

if __name__ == "__main__":
    latest_run_path = find_latest_run_path(TENSORBOARD_ROOT)
    
    if latest_run_path:
        # --- Data Extraction ---
        git_info = get_git_info()
        metrics, start_time, end_time = extract_scalars(latest_run_path)
        monitor_summary = analyze_monitor_csv(latest_run_path) # (FIXED) Pass run_path
        reqs = get_requirements()
        model_files = list_model_checkpoints()
        hyperparams = extract_hyperparameters(latest_run_path) # (FIXED) Pass run_path

        # --- Report Generation ---
        report = generate_markdown_report(
            metrics, 
            monitor_summary, 
            reqs, 
            model_files, 
            hyperparams,
            latest_run_path,
            git_info,
            start_time,
            end_time
        )
        
        with open(OUTPUT_FILE, "w") as f:
            f.write(report)
            
        print(f"✅ Successfully exported comprehensive debug summary to: {os.path.abspath(OUTPUT_FILE)}")
    else:
        print(f"Could not generate report. Please ensure a training run has started and logs exist in {LOG_DIR}.")