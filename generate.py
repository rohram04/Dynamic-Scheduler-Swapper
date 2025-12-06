import signal
import subprocess
import time
import psutil
import pandas as pd
import itertools
from datetime import datetime
import threading
import os
import random
from parse import parse_and_calculate_worker_metrics

perf_folder = os.path.join('.', 'perf_data')

if not os.path.exists(os.path.join(perf_folder, "binary")):
    os.makedirs(os.path.join(perf_folder, "binary"), exist_ok=True)

if not os.path.exists(os.path.join(perf_folder, "logs")):
    os.makedirs(os.path.join(perf_folder, "logs"), exist_ok=True)

# --- 1. Experiment Configuration ---
# CPU_CONFIGS = [1, 2, 4]
# IO_CONFIGS = [0, 4, 8]
# MEM_CONFIGS = [0, 4, 8]
# MUTEX_VALUES = [0, 4, 8]

CPU_CONFIGS = [4]
IO_CONFIGS = [2]
MEM_CONFIGS = [0]
DURATION = 1                  # CRITICAL CHANGE: Reduced duration to 2 seconds

# Fixed parameter for memory runs to ensure swap churn (since we removed its variance)
FIXED_VM_WORKERS = 4

# Total unique runs: 5 * 7 * 6 * 4 = 840 Total Workload Configurations

# Global data structures and synchronization tools
GLOBAL_TIME_SERIES = []
TS_LOCK = threading.Lock()
STOP_EVENT = threading.Event()

# --- 2. Stressor Control Function ---
# --- 2. Stressor Control Function (Updated) ---
# --- 2. Stressor Control Function (CRITICAL UPDATE FOR SWAP) ---
def start_stressor(cpu_workers, io_workers, mem_load, duration, experiment_id):
    """
    Starts stress-ng, now including a fixed, high memory stress (e.g., 5GB)
    to guarantee swap pressure, independent of io_workers.
    """
    # CRITICAL CHANGE: Set a fixed, large memory request (e.g., 5GB).
    # This value MUST be larger than your physical RAM to force swapping.
    # Adjust '5G' based on your physical RAM size (e.g., if you have 8GB RAM, use '10G').
    vm_workers = FIXED_VM_WORKERS if mem_load > 0 else 0
    vm_bytes = f'{mem_load}G'
    command_str = f"sudo perf record -e sched:sched_process_fork,sched:sched_process_exit,sched:sched_switch \
    -a -o {os.path.join(perf_folder, "binary", f'perf.data.{experiment_id}')} -- "

    command = command_str.split() + [
        "stress-ng",
        f"--cpu", str(cpu_workers),
        "--cpu-method", "loop",
        f"--io", str(io_workers),
    ]

    # Only add VM stressors if io_workers is > 0
    if vm_workers > 0:
         command.extend([
            f"--vm", str(vm_workers), 
            f"--vm-bytes", vm_bytes,
            f"--vm-populate",
            f"--vm-hang", "1",
         ])
    
    command.extend([
        f"--timeout", f"{duration}s",
    ])
    
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process

# --- 3. Feature Collection (The Core Logic) ---
def metric_monitor(stress_pid, experiment_id, interval=0.5):
    """
    Runs in a separate thread to continuously collect all 8 ML features.
    """
    # Initialize previous state for rate calculation (NVCsw/s, VCsw/s)
    prev_nvcsw = 0
    prev_vcsw = 0
    prev_io_read = 0
    prev_io_write = 0
    prev_time = time.time()
    
    try:
        stress_proc = psutil.Process(stress_pid)
    except psutil.NoSuchProcess:
        return

    while not STOP_EVENT.is_set():
        current_time = time.time()
        delta_time = current_time - prev_time
        
        # --- Collect Raw Data ---
        load_avg = os.getloadavg()
        cpu_percent = psutil.cpu_percent(interval=None) 
        io_counters = psutil.disk_io_counters()
        swap_mem = psutil.swap_memory()
        
        # Aggregate process-specific metrics
        total_nvcsw = 0
        total_vcsw = 0
        try:
            worker_pids = [p.pid for p in stress_proc.children(recursive=True)]
            worker_pids.append(stress_pid)
            for pid in worker_pids:
                try:
                    csw = psutil.Process(pid).num_ctx_switches()
                    total_nvcsw += csw.involuntary
                    total_vcsw += csw.voluntary
                except (psutil.NoSuchProcess):
                    continue
        except psutil.NoSuchProcess:
            break

        # --- Calculate Rate Features (NVCsw/s, VCsw/s, I/O rates) ---
        snapshot = {
            'exp_id': experiment_id,
            'timestamp': current_time,
            
            # 1. Contention & Latency
            'T_run_queue': load_avg[0], 
            'T_nvcsw_rate': (total_nvcsw - prev_nvcsw) / delta_time if delta_time > 0 else 0,
            'T_vcsw_rate': (total_vcsw - prev_vcsw) / delta_time if delta_time > 0 else 0,
            
            # 2. Resource Utilization
            'T_cpu_percent': cpu_percent,
            'T_io_read_rate': (io_counters.read_bytes - prev_io_read) / delta_time if delta_time > 0 else 0,
            'T_io_write_rate': (io_counters.write_bytes - prev_io_write) / delta_time if delta_time > 0 else 0,
            
            # 3. Memory Pressure
            'T_swap_rate': swap_mem.sin / delta_time if delta_time > 0 else 0, # Swap-in rate (pages/sec)
        }
        
        with TS_LOCK:
            GLOBAL_TIME_SERIES.append(snapshot)

        # Update previous state
        prev_nvcsw = total_nvcsw
        prev_vcsw = total_vcsw
        prev_io_read = io_counters.read_bytes
        prev_io_write = io_counters.write_bytes
        prev_time = current_time
        
        time.sleep(interval)
        
# --- 4. Main Execution Loop ---
def run_feature_generation_suite(scheduler:str):
    local_summary_df = pd.DataFrame()
    # Hidden Variables to generate workload variety
    all_experiments = list(itertools.product(CPU_CONFIGS, IO_CONFIGS, MEM_CONFIGS))
    
    for i, (cpu, io, mem_load) in enumerate(all_experiments):
        exp_id = f"EXP_{i:02d}_{random.randint(1000, 9999)}"
        print(f"\n--- Running {exp_id} (CPU={cpu}, IO={io}, Mem={mem_load}G) ---")
        
        # Reset monitoring state
        STOP_EVENT.clear()
        
        # Start the stressor
        stress_proc = start_stressor(cpu, io, mem_load, DURATION, exp_id)
        
        # Start the monitoring thread
        monitor_thread = threading.Thread(target=metric_monitor, args=(stress_proc.pid, exp_id))
        monitor_thread.start()

        # Wait for the stressor to complete
        stress_proc.wait()

        stdout, stderr = stress_proc.communicate(timeout=5) # Collect output after wait
        if stress_proc.returncode != 0:
            print(f"  > Stressor exited with ERROR code {stress_proc.returncode}")
            print(f"  > STDOUT: {stdout.decode()}")
            print(f"  > STDERR: {stderr.decode()}")
        
        # Stop the monitoring thread
        STOP_EVENT.set()
        monitor_thread.join()

        input = os.path.join(perf_folder, "binary", f'perf.data.{exp_id}')
        output = os.path.join(perf_folder, "logs", f'trace_log.{exp_id}')
        log_proc = subprocess.run(f"sudo perf script -i {input} > {output}", shell= True)
        os.remove(input)

        metrics = parse_and_calculate_worker_metrics(output, stress_proc.pid)
        os.remove(output)

                # 1. Start with the experiment configuration details
        results_row = {
            'exp_id': exp_id,
            'scheduler': scheduler,
            'cpu': cpu,
            'io': io,
            'mem_load': mem_load,
        }
        
        # 2. Add the calculated metrics to the row
        results_row.update(metrics)

        local_summary_df = pd.concat([local_summary_df, pd.DataFrame([results_row])], ignore_index=True)

        # Clean up process
        try:
            stress_proc.terminate()
            stress_proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            print("  > Stress process cleanup timed out.")
        
        
    # --- 5. Final Data Processing ---
    final_df = pd.DataFrame(GLOBAL_TIME_SERIES)
    
    # Clean up and prepare for ML
    df_features = final_df.drop(columns=['timestamp']).copy()
    
    # Group by experiment run and calculate the mean of all live features
    df_features_mean = df_features.groupby('exp_id').mean().reset_index()

    merged_df = pd.merge(
        local_summary_df, 
        df_features_mean, 
        on='exp_id', 
        how='left'
    )

    return merged_df

features_df = run_feature_generation_suite("CFS")
features_df.to_csv('output.csv', index=False)

for scheduler_name, scheduler_path in {'fifo': "/home/ubuntu/bin/scx_simple", "rlfifo": "/home/ubuntu/.cargo/bin/scx_rlfifo"}.items():
    # --- Execute and Display ---
    scx_process = subprocess.Popen(["sudo", scheduler_path], start_new_session=True)
    try:
        # 2. Run the workload and feature generation suite
        # This function should contain your perf tracing and parsing logic.
        features_df = pd.concat([features_df, run_feature_generation_suite(scheduler_name)]).reset_index(drop=True)

        features_df.to_csv('output.csv', index=False)
        print("Feature generation complete. Terminating scheduler...")

    finally:
        # --- CLEANUP PHASE ---
        # 3. Terminate the SCX process.
        # We use a proper signal (SIGTERM) to allow a graceful exit, then SIGKILL if needed.
        
        try:
            # Negative PID (os.killpg) sends the signal to the entire process group
            os.kill(scx_process.pid, signal.SIGTERM) 
            
            # Wait for the process group leader (the shell) to exit
            scx_process.wait(timeout=5) 
        except ProcessLookupError:
            # Already terminated, which is fine
            pass 
        except subprocess.TimeoutExpired:
            print("SCX scheduler process group did not terminate gracefully. Forcing kill...")
            # If SIGTERM fails, send SIGKILL to the entire group
            try:
                os.kill(scx_process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass # Already dead

print("\n" + "="*80)
print("## 🚀 Final ML Feature Set (Averaged System State per Experiment)")
print("="*80)
print(features_df.head(10))

features_df.to_csv('output.csv', index=False)

print("\nTotal features collected per experiment (excluding ID): 8")
