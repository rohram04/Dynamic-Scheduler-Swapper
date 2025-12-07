import pandas as pd
import re
from typing import Dict
import numpy as np

LOG_PATTERN = re.compile(
    r'^\s*([\w\-]+)\s+(\d+)\s+\[(\d+)]\s+(\d+\.\d+):\s+(.*)$'
)

def parse_and_calculate_worker_metrics(log_file_path: str) -> Dict[str, float]:
    """
    Parses a perf script trace log and calculates metrics for all processes
    that appear in the trace.
    """
    worker_data: Dict[int, Dict[str, float]] = {}

    experiment_start_time = float('inf')
    experiment_end_time = float('-inf')

    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                match = LOG_PATTERN.match(line)
                if not match:
                    continue

                command, pid_str, cpu, timestamp_str, details = match.groups()
                timestamp = float(timestamp_str)
                pid = int(pid_str)

                # Update experiment duration
                experiment_start_time = min(experiment_start_time, timestamp)
                experiment_end_time = max(experiment_end_time, timestamp)

                # Initialize worker if not seen yet
                if pid not in worker_data:
                    worker_data[pid] = {
                        'T_Arrival': timestamp,  # First time we see it
                        'T_First_CPU': None,
                        'T_Completion': None,
                        'Total_CPU_Time': 0.0,
                        'T_Last_Scheduled_In': None
                    }

                # --- Fork event ---
                if 'sched:sched_process_fork' in details:
                    child_pid_match = re.search(r'child_pid=(\d+)', details)
                    if child_pid_match:
                        child_pid = int(child_pid_match.group(1))
                        if child_pid not in worker_data:
                            worker_data[child_pid] = {
                                'T_Arrival': timestamp,
                                'T_First_CPU': None,
                                'T_Completion': None,
                                'Total_CPU_Time': 0.0,
                                'T_Last_Scheduled_In': None
                            }

                # --- Exit event ---
                elif 'sched:sched_process_exit' in details:
                    worker_data[pid]['T_Completion'] = timestamp

                # --- Context switch ---
                elif 'sched:sched_switch' in details:
                    # Track running PID leaving CPU
                    if pid in worker_data and worker_data[pid]['T_Last_Scheduled_In'] is not None:
                        slice_time = timestamp - worker_data[pid]['T_Last_Scheduled_In']
                        worker_data[pid]['Total_CPU_Time'] += slice_time
                        worker_data[pid]['T_Last_Scheduled_In'] = None

                    # Track next PID scheduled in
                    next_pid_match = re.search(r'==>\s+[\w\-]+:(\d+)\s+', details)
                    if next_pid_match:
                        next_pid = int(next_pid_match.group(1))
                        if next_pid not in worker_data:
                            worker_data[next_pid] = {
                                'T_Arrival': timestamp,
                                'T_First_CPU': None,
                                'T_Completion': None,
                                'Total_CPU_Time': 0.0,
                                'T_Last_Scheduled_In': None
                            }
                        if worker_data[next_pid]['T_First_CPU'] is None:
                            worker_data[next_pid]['T_First_CPU'] = timestamp
                        worker_data[next_pid]['T_Last_Scheduled_In'] = timestamp

    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file_path}")
        return {}

    # --- Compute metrics ---
    total_tat = 0.0
    total_rt = 0.0
    valid_tat_count = 0
    valid_rt_count = 0
    cpu_times = []

    for pid, data in worker_data.items():
        # Finalize CPU slice if running at end
        if data['T_Last_Scheduled_In'] is not None:
            data['Total_CPU_Time'] += experiment_end_time - data['T_Last_Scheduled_In']

        # Only include processes with TAT
        if data['T_Completion'] is not None:
            tat = data['T_Completion'] - data['T_Arrival']
            total_tat += tat
            valid_tat_count += 1

            if data['T_First_CPU'] is not None:
                rt = data['T_First_CPU'] - data['T_Arrival']
                total_rt += rt
                valid_rt_count += 1

            cpu_times.append(data['Total_CPU_Time'])

    avg_tat = total_tat / valid_tat_count if valid_tat_count > 0 else 0.0
    avg_rt = total_rt / valid_rt_count if valid_rt_count > 0 else 0.0

    if cpu_times:
        cpu_times_array = np.array(cpu_times)
        mean_cpu = np.mean(cpu_times_array)
        std_cpu = np.std(cpu_times_array)
        cv_fairness = std_cpu / mean_cpu if mean_cpu > 0 else 0.0
    else:
        cv_fairness = 0.0

    return {
        'Average_TAT': avg_tat,
        'Average_RT': avg_rt,
        'Total_Experiment_Duration': experiment_end_time - experiment_start_time,
        'CV_Fairness': float(cv_fairness)
    }
