import psutil
import time
from collections import deque
from dataclasses import dataclass
import threading

@dataclass
class Snapshot:
    experiment_id: int

    # CPU
    T_cpu_percent: float
    T_cpu_user_percent: float
    T_iowait_percent: float
    T_irq_percent: float
    T_softirq_percent: float
    
    # Threads / Queue
    T_run_queue: int
    T_active_threads: int
    T_blocked_threads: int
    T_io_blocked_threads: int
    
    # Memory
    T_mem_used: int
    T_mem_available: int
    T_swap_used: int
    T_cache_mem: int
    T_buffers_mem: int
    
    # Totals / Cumulative
    T_swap_in_total: int
    T_swap_out_total: int
    T_io_read_total: int
    T_io_write_total: int
    T_nvcsw_total: int
    T_vcsw_total: int

class RLStateBuilder:
    def __init__(self, pid, experiment_id, duration=4, interval=0.25):
        self.pid=pid
        self.experiment_id=experiment_id
        self.buffer_size = int(duration // interval)
        self.raw_buffer = deque(maxlen=self.buffer_size)
        self._stop_event = threading.Event()
        self.interval=interval
        self._thread = None

        # Define which metrics are averages vs cumulative totals
        self.avg_metrics = [
            "T_cpu_percent", "T_iowait_percent", "T_irq_percent", "T_softirq_percent",
            "T_run_queue", "T_active_threads", "T_blocked_threads", "T_io_blocked_threads",
            "T_mem_used", "T_mem_available", "T_swap_used", "T_cache_mem", "T_buffers_mem"
        ]
        self.total_metrics = [
            "T_swap_in_total", "T_swap_out_total",
            "T_io_read_total", "T_io_write_total",
            "T_nvcsw_total", "T_vcsw_total"
        ]

    def collect_raw(self):
        """Collect a snapshot and store in the buffer."""
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        io = psutil.disk_io_counters()
        cpu_percent = psutil.cpu_times_percent()
        proc = psutil.Process()
        threads = proc.threads()

        T_vcsw_total = 0
        T_nvcsw_total = 0

        for p in psutil.process_iter():
            try:
                T_vcsw_total += p.num_ctx_switches().voluntary
                T_nvcsw_total += p.num_ctx_switches().voluntary
            except psutil.NoSuchProcess:
                # Process ended before we could query it, ignore
                pass

        snapshot = Snapshot(
            experiment_id=self.experiment_id,
            T_cpu_percent=psutil.cpu_percent(),
            T_cpu_user_percent=cpu_percent.user,
            T_iowait_percent=cpu_percent.iowait,
            T_irq_percent=getattr(cpu_percent, "irq", 0.0),
            T_softirq_percent=getattr(cpu_percent, "softirq", 0.0),

            T_run_queue=len(threads),
            T_active_threads=sum(1 for t in threads if t.user_time > 0),
            T_blocked_threads=0,      # fill if available
            T_io_blocked_threads=0,   # fill if available

            T_mem_used=mem.used,
            T_mem_available=mem.available,
            T_swap_used=swap.used,
            T_cache_mem=getattr(mem, "cached", 0),
            T_buffers_mem=getattr(mem, "buffers", 0),

            T_swap_in_total=swap.sin,
            T_swap_out_total=swap.sout,
            T_io_read_total=io.read_bytes,
            T_io_write_total=io.write_bytes,
            T_nvcsw_total=T_nvcsw_total,
            T_vcsw_total=T_vcsw_total
        )

        self.raw_buffer.append(snapshot)

    def build_state(self):
        """Compute 1-second averages and deltas for the RL state."""
        if len(self.raw_buffer) < self.buffer_size:
            return None

        samples = list(self.raw_buffer)
        newest = samples[-1]
        oldest = samples[0]

        state = {}

        # Averages + delta/pct change
        for m in self.avg_metrics:
            values = [getattr(s, m) for s in samples]
            avg = sum(values) / len(values)
            delta = getattr(newest, m) - getattr(oldest, m)
            oldest_val = getattr(oldest, m)
            pct_change = (delta / oldest_val * 100) if oldest_val != 0 else 0
            state[m] = {"avg": avg, "delta": delta, "pct_change": pct_change}

        # Only delta/pct change for totals
        for m in self.total_metrics:
            delta = getattr(newest, m) - getattr(oldest, m)
            oldest_val = getattr(oldest, m)
            pct_change = (delta / oldest_val * 100) if oldest_val != 0 else 0
            state[m] = {"delta": delta, "pct_change": pct_change}

        return state
    
    def build_flat_state(self):
        """Return a flattened RL state vector/dict for the last 1 second."""
        state = self.build_state()
        if state is None:
            return {}

        flat_state = {}

        for m, metrics in state.items():
            for k, v in metrics.items():
                flat_state[f"{m}_{k}"] = v

        return flat_state


    def start_monitoring(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop)
        self._thread.start()

    def _monitor_loop(self):
        while psutil.pid_exists(self.pid) and not self._stop_event.is_set():
            self.collect_raw()
            time.sleep(self.interval)

    def stop_monitoring(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join()