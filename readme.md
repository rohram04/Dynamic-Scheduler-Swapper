# Dynamic ML Scheduler Swapper

## Problem Statement/Task:
    Build an agent that uses an ML model to swap a systems running scheduler based off it's live system metrics (CPU Load, Memory Load, etc)

Features (important ones for the readme are **bolded**):

- **T_vcsw_total_delta**: The number of voluntary context switches that occurred
- **T_nvcsw_total_delta**: The number of involuntary context switches that occured
- T_cpu_percent_avg: Average CPU load
- T_io_wait_percent_avg: Avg percentage of time spent waiting for io
- T_irq_percent_avg: indicates the average percentage of time the CPU is spending on handling hardware interrupts (runs immediately when kernel triggers interrupt)
- T_softirq_percent_avg: the average percentage of time the CPU spent handling hardware interrupts (that run shortly after shceduled y the kernel)
- T_run_queue_avg: The average number of tasks waiting to be run (run queue length)
- T_active_threads_avg: the average number of threads actively running on the cpu
- T_blocked_threads_avg: the average number of threads blocked due to I/O, locks, or a condition variable
- T_io_blocked_threads_avg: Average number of threads blocked due to I/O
- T_mem_used_avg: the average memory used
- T_mem_available: the average memory available
- T_swap_used_avg: average amount of swap memory used
- T_cache_mem_avg: average cache memory used
- T_buffers_mem_avg: the average amount of memory used by kernel buffers
- T_swap_in_total_delta: The amount of memory swapped in
- T_swap_out_total_delta: The amount of memory swapped out
- T_io_write_total_delta: the amount of data written through I/O
- T_io_read_total_delta: The amount of data read through I/O

>**Aside**: Note I collected total delta (the total change throughout the experiment) and percent change to capture trends for all the features with the intention of removing unecessary features later on. Thus I have ommitted them in the features above. Also note, even features above might not be important, I just had a guess they might be useful.
>
> Additionally, this repo contains data where a lot of these values are 0. Certain values require certain workloads to be run. For example, if T_swap_used_avg is to not be 0, then stress-ng must be run with the params that overload memory and force disk swapping. Thus, if a column has 0 for data in this repo it just means the stress-ng params I used for the important examples in this readme did not involve the params required to see a non zero value in these columns.

## Special Tools and Libraries:
- stress-ng: A library used to stress test machines with a variety of workloads including, cpu, io, memory, and disk stress tests.
- sched_ext: "sched_ext is a Linux kernel feature which enables implementing kernel thread schedulers in BPF and dynamically loading them." It contains a variety scheduler implementations the two I cared about were: 
    - scx_simple (fifo scheduler written in c)
    - scx_bpfland (a low latency scheduler good for gaming, audio streaming, etc)

## ML model design: Classification?

### Idea 1:
My original idea was to treat this as a classification problem. Based of the current system metrics scheduler A is the best Scheduler so switch to A.
Began with the idea of creating a random set of workloads and running each scheduler on the random set. Then classified which scheduler was the best by how well it performed based on the following cost function with turnaround time, response time, and coefficient of variance of fairness:
    
$ L = TAT_{avg} + RT_{avg} + CV Fairness_{avg} $

The best scheduler has the lowest cost

#### What are the issues with this idea?

- There set of possible workloads is HUGE. It was becoming difficult for me to even imagine how to capture a large portion of this set especially with stress-ng. A classification model wouldn't generalize well to this set.
- The second point is I am optimizing for a cost function and then labeling a classifier based off the cost (...hint that this isn't a classification problem it's an optimization problem)

### If classification isn't right then what is? Idea 2:

**Reinforcement Learning**. I specifically wanted to implement a policy gradient algorithm (based off Professor Young Wu's advice)

The idea was some sort of dual layer hidden network that feeds into a softmax function to produce a vector of probabilities for the likelihood of the algorithm choosing a specific policy.

softmax outputs $a = [a_1, a_2, a_3]$ (1 for each scheduling policy)

The reward would be a weighted cross-entropy function where the cost function L would be multiplied by the log of the probabilities and summed.

$CE = \sum_{i} L_i \cdot \log a_i$

The agent would then explore the action space using an Epsilon-Greedy approach (ideally [Upper Confidence Bound Algorithm](https://www.turing.com/kb/guide-on-upper-confidence-bound-algorithm-in-reinforced-learning) but did not have time for that). 

An epsilon greedy approach is a function that determines whether the algorithm should choose the optimal choice based off it's current training (argmax $a_i$) or whether it should choose some other policy (random)
to explore the action space (could there be a better action). The probability that it chooses a random policy is usually something low ~ 1/sqrt(n)

![Epsilon Greedy function](/epsilongreedy.png)

<!-- $
\text{Epsilon-Greedy} =
\begin{cases}
\arg\max a & \text{with probability } 1 - \epsilon,\\[2mm]
\text{random action from action space} & \text{with probability } \epsilon,
\end{cases}
\qquad
\text{where } \epsilon = \frac{1}{\sqrt{n}},
$ -->

I would feed the algorithm an episode: a defined sequence of varying workloads (specifically, the live system metrics produced from these workloads) each iteration.

![Reinforcement Learning Loop](/reinforcementlearning.png)

## Metric Collection:
Metrics were collected using two libraries - psutil and perf.
- Perf works by writing to a trace file in binary which must be consumed and translated to a txt file AFTER THE TRACE IS COMPLETED (see aside below). This means there isn't an easy way to calculate a rolling average of these kernel based metrics live. The following metrics were calculated from perf:
    - Average turnaround time
    - Average Response time
    - Coefficient of Variation Fairness
- All other metrics were collected from psutil with simple function calls and calculations.

> **Aside:**  This influenced me to look into why we can't get these kernel metrics in real time? Perf uses a library called eBPF to access kernel data. eBPF works in the kernel, runs calculations, and updates data structure (file, map, etc) that is shared with the user space. The user then consumes the data from this map or file and performs their actions. Some of the sched_ext schedulers (like bpfland) are built almost completely in eBPF with minimal user space
functions. 
>
> So easy, read this data structure live, do your user actions and delete the used data from the data structure and we have live kernel based metrics, right? Well the quantity of metrics the kernel produces is so high that the stream to user space would get flooded and user space would not be able to keep up. User space won't be able to delete data fast enough to keep up with the kernels update and the data structure will grow infinitely due to the throughput in this stream. Thus getting live metrics through eBPF is not so easy and requires heavy filtering before updating the data structure. Datadog actually dealt with this problem here https://www.datadoghq.com/blog/engineering/workload-protection-ebpf-fim/. Anyways, getting live turnaround time, response time, fairness, etc through eBPF is quite difficult.

**Please note all metrics below were collected from stress-ng runs using the *--no-rand-seed* flag which means the runs for each combination of params are deterministic**

## Experiment Design and Data

### Experiment setup 1: VM
I would run the experiment on an ubuntu server virtual machine (through multipass) and on each iteration the ML model would consume the previous timestep (workload metrics) choose an action and then run a stress-ng workload for 5 seconds* based on the episode definition, and then calculate its reward based on the CE function defined earlier.

> **aside**: 5 seconds was chosen as the timestep because the data was incredibly instable in the first 2 seconds of workload. This could be due to a variety of reasons like stress-ng was just starting up, or the scheduler did not make many decisions, etc. Extending the duration of the experiment made the metrics more stable (however still not stable).

There is however another problem with this setup: Running the ML model on the same machine as the stress-ng workload simulation will result in metrics influenced by the ML model workload itself. Thus the stress-ng workload simulation must be isolated on it's own machine.

### Experiment setup 2: Host -> HTTP server on VM
The ML model runs on my host computer, and the stress-ng simulation will run inside the VM. The VM will 
have an http server which recieves simulation requests, runs the stress-ng simulation, and returns the results. Thus the ML
model will post to this server when it needs to run a simulation. This means the stress-ng workload will control the
VMs metrics completely (I hoped) and the results would be more stable.

The collected Average Turnaround Time and Response Time, were still incredibly unstable.
I realized that the VM could be influencing these metrics. The host OS has the ability to pre-empt and interrupt the VM at any moment (and since I could not pin the VM to a cpu on my machine).
Thus these metrics could be skewed by the host OS.


### CPU / RT / Context-Switch Metrics (normalized per core)
|                          |            mean |           std |   cv_% |
|:-------------------------|----------------:|--------------:|--------:|
| Average_RT               |     0.000622516 |   0.000193763 |   31.13 |
| T_nvcsw_total_delta      | 65169.5         | 709.252       |    1.09 |
| T_nvcsw_total_pct_change |     4.04011     |   2.5783      |   63.82 |
| T_vcsw_total_delta       | 65169.6         | 709.26        |    1.09 |
| T_vcsw_total_pct_change  |     4.04012     |   2.57831     |   63.82 |

[/stats/vm/cpu_variability.csv](/stats/vm/cpu_variability.csv)

*Averages of 50 runs. Raw data can be found in [/stats/vm/results.csv](/stats/vm/results.csv)*


### Other Metrics
|                             |             mean |              std |   cv_% |
|:----------------------------|-----------------:|-----------------:|--------:|
| T_cpu_percent_avg           |     23.5424      |      3.15693     |   13.41 |
| T_iowait_percent_avg        |      0.003       |      0.0212132   |  707.11 |
| T_mem_used_avg              |      1.12577e+09 |      8.73157e+07 |    7.76 |
| T_mem_available_avg         |      2.96016e+09 |      8.73157e+07 |    2.95 |
| T_mem_available_delta       |     -3.47845e+07 |      4.32038e+07 | -124.2  |
| T_mem_available_pct_change  |     -1.04949     |      1.26346     | -120.39 |
| T_swap_used_avg             |      0           |      0           |  nan    |
| T_cache_mem_avg             |      8.30524e+08 |      1.21207e+06 |    0.15 |
| T_cache_mem_delta           |      2.51138e+07 | 284256           |    1.13 |
| T_cache_mem_pct_change      |      3.10116     |      0.036219    |    1.17 |
| T_buffers_mem_avg           |      8.50009e+07 |      1.2235e+07  |   14.39 |
| T_buffers_mem_delta         | 891372           | 664454           |   74.54 |
| T_buffers_mem_pct_change    |      1.22006     |      1.16375     |   95.38 |
| T_io_write_total_delta      |      3.23921e+07 | 462546           |    1.43 |
| T_io_write_total_pct_change |      3.62033     |      2.02909     |   56.05 |
| Average_TAT                 |      0.525148    |      0.0675086   |   12.86 |
| CV_Fairness                 |      0.844093    |      0.0867632   |   10.28 |

[/stats/vm/other_variability.csv](/stats/vm/other_variability.csv)

*Averages of 50 runs. Raw data can be found in [/stats/vm/results.csv](/stats/vm/results.csv)*

So I bought a PC containing an i3-1010F (4 cores, 8 threads), 8gb of memory, 128gb SSD from facebook marketplace and installed ubuntu server 25.10 on it.

### Final Experiment Setup: Bare Metal HTTP Server
The HTTP server would now run on the PC and allowing me to collect data on baremetal. Thus no host os, and hopefully stable metrics.

And the metrics? They were better but still not meaningful

### CPU / RT / Context-Switch Metrics (normalized per core)
|                          |           mean |   normalized_mean |          std |   cv_% |
|:-------------------------|---------------:|------------------:|-------------:|--------:|
| Average_RT               |      0.0155396 |         0.0038849 |    0.0330805 |  851.51 |
| T_nvcsw_total_delta      | 110888         |     27722         | 7559.21      |   27.27 |
| T_nvcsw_total_pct_change |      0.599884  |         0.149971  |    0.0843382 |   56.24 |
| T_vcsw_total_delta       | 110890         |     27722.4       | 7559.2       |   27.27 |
| T_vcsw_total_pct_change  |      0.599892  |         0.149973  |    0.0843397 |   56.24 |

[/stats/bare_metal/cpu_variability.csv](/stats/bare_metal/cpu_variability.csv)

*Averages of 50 runs. Raw data can be found in [/stats/bare_metal/results.csv](/stats/bare_metal/results.csv)*

### Other Metrics
|                             |         mean |   normalized_mean |              std |   cv_% |
|:----------------------------|-------------:|------------------:|-----------------:|--------:|
| T_cpu_percent_avg           | 64.572       |      16.143       |      1.29521     |    2.01 |
| T_iowait_percent_avg        |  9.663       |       2.41575     |      0.743585    |    7.7  |
| T_mem_used_avg              |  1.6055e+09  |       4.01375e+08 |      2.99798e+07 |    1.87 |
| T_mem_available_avg         |  6.00118e+09 |       1.5003e+09  |      2.99798e+07 |    0.5  |
| T_mem_available_delta       |  1.06427e+09 |       2.66067e+08 |      3.46643e+07 |    3.26 |
| T_mem_available_pct_change  | 18.6075      |       4.65188     |      0.682002    |    3.67 |
| T_swap_used_avg             |  5.16211e+08 |       1.29053e+08 |      0           |    0    |
| T_cache_mem_avg             |  8.87669e+08 |       2.21917e+08 |      2.52858e+06 |    0.28 |
| T_cache_mem_delta           |  5.66641e+07 |       1.4166e+07  |      3.18613e+06 |    5.62 |
| T_cache_mem_pct_change      |  6.64774     |       1.66193     |      0.38097     |    5.73 |
| T_buffers_mem_avg           |  3.94417e+08 |       9.86043e+07 |      3.67734e+07 |    9.32 |
| T_buffers_mem_delta         |  2.11747e+06 |  529367           | 597162           |   28.2  |
| T_buffers_mem_pct_change    |  0.555394    |       0.138848    |      0.20123     |   36.23 |
| T_io_write_total_delta      |  1.57984e+08 |       3.94959e+07 |      8.42689e+06 |    5.33 |
| T_io_write_total_pct_change |  0.140742    |       0.0351855   |      0.00779883  |    5.54 |
| Average_TAT                 |  4.97263     |       1.24316     |      0.175754    |    3.53 |
| CV_Fairness                 |  1.99492     |       0.49873     |      0.283126    |   14.19 |

[/stats/bare_metal/other_variability.csv](/stats/vm/other_variability.csv)

*Averages of 50 runs. Raw data can be found in [/stats/bare_metal/results.csv](/stats/bare_metal/results.csv)*

Comparing this to the metrics above for the VM:

- The coefficient of variation, in general, has decreased; Specifically, Average Turnaround Time coefficient of variation has decreased. However it's interesting that the turnaround time on bare metal is ~5s while on the VM its < 1s. (see aside below)
- The normalized mean for context switches is far less on bare metal than on the VM. This supports the earlier guess that the Host is pre-empting/interupting the VM to a great extent resulting in skewed metrics*

> **Aside**: Looking into this further. I found that the VM had ~90 different processes that completed (~20 of which had ~5s turnaround time, these are the stress-ng processes) of which ~70 were super short turnaround time. This is much more than the 20 processes completed on bare metal for the same experiment. This explains why the context switches were much higher on the VM than the bare metal and why the turnaround time was much lower on the VM. It must be the VM switching to the Host, time slicing amongst cores, etc. See the data below - you can observe the lower turnaround times at the bottom:
>
>| PID  | T_Arrival    | T_First_CPU | T_Completion | TAT     | RT       | Total_CPU_Time|
>|------|--------------|-------------|--------------|---------|----------|---------------|
>| 44245 | 2756.035301 | 2756.037448 | 2761.239355 | 5.204054 | 0.002147 | 0.508635      |
>| 44213 | 2756.032778 | 2756.032778 | 2761.313197 | 5.280419 | 0.000000 | 0.003342      |
>| 44246 | 2756.035332 | 2756.036239 | 2761.279171 | 5.243839 | 0.000907 | 0.438215      |
>| 44247 | 2756.035366 | 2756.036327 | 2761.302199 | 5.266833 | 0.000961 | 0.372372      |
>| 44248 | 2756.035403 | 2756.036418 | 2761.312789 | 5.277386 | 0.001015 | 0.376296      |
>| ...   | ...         | ...         | ...         | ...      | ...      | ...           |
>| 44314 | 2760.023552 | 2760.023801 | 2760.036905 | 0.013353 | 0.000249 | 2.552706      |
>| 44315 | 2760.024630 | 2760.031452 | 2760.034861 | 0.010231 | 0.006822 | 2.555692      |
>| 44316 | 2760.025678 | 2760.035504 | 2760.035753 | 0.010075 | 0.009826 | 2.552691      |
>| 44317 | 2760.026805 | 2760.027887 | 2760.036126 | 0.009321 | 0.001082 | 2.554203      |
>| 44318 | 2760.037389 | 2760.037422 | 2760.039281 | 0.001892 | 0.000033 | 2.549101      |

## Data Problems:

### 1. Poor Cost function
The cost function $L$ defined earlier requires several processes with varying turnaround times indicating how well the scheduler performed. However The turnaround time created by stress-ng indicates all the workers run for the whole timeout duration set. They are not being introduced to the scheduler randomly. In addition, response time has too much variation resulting in a bad cost function. This means my attempt to influence these parameters through a stress-ng workload was not great. **Could I influence these params with a custom script that benefits a specific scheduler?** which leads into problem 3

### 2. If stress-ng doesn't work, how do I create a variety of workloads?
It's unclear what could be used. I specifically need a workload simulator that creates a variety of threads/processes. Some of the following could have worked:
- Filebench: This has the ability to simulate the workload of a fileserver, webserver, dataserver, etc (still limited workloads but can be used to write more). However it is ~10 years old so support for modern OSes is lacking. I couldn't compile it on ubuntu server 25.10
- Phoronix Suite: A great option with a variety of tests but the tests are several minutes long. Running a perf trace on these tests would create a ridiculously large binary that must be processed after to get kernel based metrics.
- There's a variety of other options like locust, wrk2, and more that are great for creating workloads for specific niches like http traffic, or file writing, but there do not seem to be comprehensive workload simulators.

> **An aside**: Workload simulators like stress-ng often have a random seed. The testing tools need to allow the user to input a seed to create deterministic workloads between the 3 schedulers, ideally. Otherwise, for each workload, I would need to run it multiple times on each scheduler and average the results.

### 3. Can I create workloads that favor a specific scheduler?
I tried a variety of custom scripts that spawned new threads and forked to create new processes trying to benefit a specific scheduler. For example, creating short burst scripts with one long running script in the background to benefit scx_bpfland (the low latency scheduler). I did notice the average turnaround time decreased for such scripts (and was not ~the script duration) but the sched_ext schedulers did not beat CFS. In many cases they performed similar or worse than CFS. 

**However** there was one situation in which I noticed bpfland had better results than CFS. This was when I remote accessed the bare metal server with remote ssh and ran the http server from vs code terminal (I did not observe the same results when I sshed and ran the http server from the mac terminal). This is consistent because bpfland is supposed to be better for interactive low latency tasks (ie vs code). Looking at the number of tasks from the perf trace, it is clear several short burst tasks are run and completed. My attempts above were to create a script that simulates what VSCode (or an interactive app) does however I was not able to do so.

**CFS with vscode running**
| metrics   |   Average_TAT |
|:----------|--------------:|
| count     |    50         |
| mean      |     1.29537   |
| std       |     0.0407553 |
| min       |     1.04232   |
| 25%       |     1.28326   |
| 50%       |     1.31687   |
| 75%       |     1.31703   |
| max       |     1.3172    |
| nan       |   nan         |

**bpfland with vscode running**
| metrics   |   Average_TAT |
|:----------|--------------:|
| count     |    50         |
| mean      |     0.978578  |
| std       |     0.0922685 |
| min       |     0.914645  |
| 25%       |     0.931119  |
| 50%       |     0.965393  |
| 75%       |     0.965512  |
| max       |     1.34343   |

[VS Code demo](./vscodedemo.mov) - *Make sure to look at the average turnaround time for each run in the postman response*

The mean is of course interesting here. It is far lower with bpfland. However, I find it more interesting that the max of bpfland is as close to the average for CFS (~1.3s) but the 75th percentile and below is < 0.96s. The reverse also holds, the min for CFS is close to the average for bpfland but the 25th percentile and above is > 1.28s. This shows that bpfland certainly performs better in turnaround time compared to CFS as expected.

## Conclusion

I was going to try to at least gather data for a simple ML classifier but the poor data quality and the inability to rapidly generate a variety of workloads hindered me from even doing that. On the otherhand though, I learned quite a bit and was really able to explore the operating system, specifically scheduler effects. It turns out testing scheduler performance on an actual machine is very difficult and there are so many factors at play. I was certainly in over my head as I think my feature engineering skills are probably not up to par, so even if I had gotten to creating a reinforcement learning or classification model, would my selected features, cost function, and attempts at tweaking the algorithm have worked? I am not sure (see aside below). However it remains clear why an RL agent for scheduler related activites is limited to theory in papers: from the initial dataset, to the parameters to the RL algorithm, it is difficult to engineer.

> **Aside**: I believe a hurdle the agent would have faced is correlating the system metrics to the cost function (and ultimately the optimal scheduler). My guess is metrics such as CPU load, disk swap rate, memory usage, etc, will not paint a clear picture for the model. But I guess that's why we use complex models in the first place, its possible the RL agent would have been able to learn something in the jungle of numbers.
