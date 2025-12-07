import uuid
import signal
import subprocess
import time
import psutil
from datetime import datetime
import threading
import os
from parse import parse_and_calculate_worker_metrics
from state import RLStateBuilder

schedulers = {
    "bpfland": "/home/ubuntu/.cargo/bin/scx_bpfland",
    "fifo": "/home/ubuntu/bin/scx_simple"
}

perf_folder = os.path.join('.', 'perf_data')

def experiment(scheduler, cpu, cpu_method, io, mem_load, duration, interval):
    exp_id = uuid.uuid4()
    results = {}
    if scheduler != "CFS":
        scx_process = subprocess.Popen(["sudo", schedulers[scheduler]], start_new_session=True)    
    # Start the stressor
    try:
        stress_proc = start_stressor(cpu, cpu_method, io, mem_load, duration, exp_id)
        # Start the monitoring thread
        metric_monitor = RLStateBuilder(stress_proc.pid, exp_id, duration, interval)
        metric_monitor.start_monitoring()

        # Wait for the stressor to complete
        stress_proc.wait()
        metric_monitor.stop_monitoring()

        stdout, stderr = stress_proc.communicate(timeout=5) # Collect output after wait
        if stress_proc.returncode != 0:
            print(f"  > Stressor exited with ERROR code {stress_proc.returncode}")
            print(f"  > STDOUT: {stdout.decode("utf-8", errors="replace")}")
            print(f"  > STDERR: {stdout.decode("utf-8", errors="replace")}")

        input = os.path.join(perf_folder, "binary", f'perf.data.{exp_id}')
        output = os.path.join(perf_folder, "logs", f'trace_log.{exp_id}')
        log_proc = subprocess.run(f"sudo perf script -i {input} > {output}", shell= True)
        os.remove(input)
        metrics = metric_monitor.build_flat_state()
        metrics.update(parse_and_calculate_worker_metrics(output))

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
        
        results = results_row
    except Exception as e:
        print(e)
    finally:
        if scheduler != "CFS":
            try:
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
        return results


def start_stressor(cpu_workers, cpu_method, io_workers, mem_load, duration, experiment_id):
    """
    Starts stress-ng, now including a fixed, high memory stress (e.g., 5GB)
    to guarantee swap pressure, independent of io_workers.
    """
    # CRITICAL CHANGE: Set a fixed, large memory request (e.g., 5GB).
    # This value MUST be larger than your physical RAM to force swapping.
    # Adjust '5G' based on your physical RAM size (e.g., if you have 8GB RAM, use '10G').
    vm_workers = 4 if mem_load > 0 else 0
    vm_bytes = f'{mem_load}G'
    # command_str = f"sudo perf record -e sched:sched_process_fork,sched:sched_process_exit,sched:sched_switch \
    # -a -o {os.path.join(perf_folder, "binary", f'perf.data.{experiment_id}')} -- ./cpu_stress --duration {duration}"

    command = [
    "sudo", "perf", "record",
    "-e", "sched:sched_process_fork,sched:sched_process_exit,sched:sched_switch",
    "-a",
    "-o", f"{perf_folder}/binary/perf.data.{experiment_id}",
    "--",
    # "./cpu_stress", "--duration", str(duration)
]

    command = command + [
        "stress-ng",
        f"--cpu", str(cpu_workers),
        "--cpu-method", cpu_method,
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

    print(command)
    
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process

# ackerman, fibonacci for interactivity
# fft matrixprod for heavy
# rand, rand48, and prime for moderate