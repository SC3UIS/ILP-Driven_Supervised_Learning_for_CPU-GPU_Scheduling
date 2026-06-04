# ILP-Driven Supervised Learning for Co-Optimizing Performance and Energy in CPU-GPU Architectures

## Overview

This repository contains the full experimental infrastructure for a hybrid **ILP+ML** scheduling methodology for heterogeneous CPU-GPU systems. The proposed approach uses Integer Linear Programming (ILP) to generate provably optimal CPU-GPU task assignments, which are then used as labeled training data for supervised classification models (Random Forest and XGBoost). At runtime, the trained models predict near-optimal assignments in near-constant time, enabling energy-aware scheduling without the computational overhead of solving an ILP at every decision point.

The methodology is validated on three representative HPC computational kernels:

- **GEMM** — General Matrix-Matrix Multiplication (compute-bound)
- **SpMV** — Sparse Matrix-Vector Multiplication (memory-bound)
- **FFT** — Fast Fourier Transform (mixed intensity)

Experiments were conducted on two hardware platforms at the SC3 HPC facility of UIS:

| Platform | CPU | GPU |
|---|---|---|
| HPE ProLiant XL290n G10+ | Intel Xeon Gold 5315Y | NVIDIA A100 40GB |
| HPE ProLiant DL580 G9 | Intel Xeon E5-4640 | NVIDIA Tesla M40 24GB |

---

## Benchmarks

Each benchmark measures **execution time** and **energy consumption** for GEMM, SpMV, and FFT across a range of problem sizes, CPU core configurations, and (for SpMV) matrix sparsity levels.

- **`benchmark_gpu_nvidia.cu`** — Implements GEMM via `cublasDgemm`, SpMV via `cusparseSpMV`, and FFT via `cuFFT`. Energy is sampled via NVML at 10 ms intervals from a dedicated pthread.
- **`benchmark_cpu_intel.cpp`** — Uses Intel MKL (`cblas_dgemm`, Inspector-Executor SpMV, DFTI) with AVX-512 vectorization. Energy via Intel RAPL sysfs.

All benchmarks follow the same measurement protocol: **3 warm-up runs** (discarded) followed by **5 measured repetitions**, reporting the median execution time and integrated energy.

---

## ILP Solver

The ILP model formulates CPU-GPU task assignment as a binary optimization problem minimizing a weighted combination of total execution latency and energy consumption:

$$\min\ Z(\mathbf{x}) = Z_L(\mathbf{x}) + \alpha \cdot \sigma \cdot Z_E(\mathbf{x})$$

with $\alpha = 0.3$ (energy policy weight) and hardware constraints on GPU memory capacity, memory bandwidth, and maximum GPU occupancy. The model is solved offline using **PuLP with the CBC solver**.

- **`ILP_pacca.py`** — Configured for the HPE XL290n G10+ node (NVIDIA A100, 40 GB VRAM, 1555 GB/s bandwidth).
- **`ILP_thor.py`** — Configured for the HPE DL580 G9 node (Tesla M40, 24 GB VRAM, 288 GB/s bandwidth).

Both scripts read a raw results CSV, solve the ILP for each CPU-core configuration, and output a labeled dataset with the optimal binary assignment (`ilp_label`: 0 = CPU, 1 = GPU) appended as a column.

---

## ML Pipeline

**`ML_models.py`** trains and evaluates two ensemble classifiers on the ILP-labeled dataset:

- **Random Forest** — Bootstrap aggregation with random feature subsets (MDI importance).
- **XGBoost** — Additive gradient boosting with second-order Taylor approximation of cross-entropy loss.

Hyperparameter optimization is performed via **Optuna TPE** (100 trials per model, 5-fold stratified CV, ROC-AUC as objective). The script produces the following outputs:

---

## Datasets

The labeled CSV files contain one row per experimental observation with the following columns:

| Column | Description |
|---|---|
| `kernel` | Kernel type: GEMM, SpMV, FFT |
| `N` | Problem size |
| `sparsity` | Sparsity fraction (SpMV only) |
| `cpu_cores` | Number of active CPU cores |
| `cpu_time_s` | Median CPU execution time (seconds) |
| `gpu_time_s` | Median GPU execution time (seconds) |
| `cpu_energy_j` | CPU energy consumption (Joules) |
| `gpu_energy_j` | GPU energy consumption (Joules) |
| `speedup` | Ratio $T_\text{cpu} / T_\text{gpu}$ |
| `op_intensity_cpu` | Operational intensity on CPU (FLOP/byte) |
| `op_intensity_gpu` | Operational intensity on GPU (FLOP/byte) |
| `ilp_label` | ILP-optimal assignment: 0 = CPU, 1 = GPU |

---

## Citation


```bibtex
@inproceedings{lemus2026ilpml,
  title     = {ILP-Driven Supervised Learning for Co-Optimizing Performance
               and Energy in CPU-GPU Architectures},
  author    = {Lemus Ram{\'i}rez, Anderson Jahir and
               Galvis Beltr{\'a}n, Johan Sebastian and
               Torres Ni{\~n}o, Luis Alejandro and
               Jaimes Barrios Hernandez, Carlos},
  booktitle = {Proceedings of CARLA 2026 -- Latin American High Performance
               Computing Conference},
  series    = {Communications in Computer and Information Science},
  publisher = {Springer},
  year      = {2026}
}
```

---

## Authors

- **Anderson Jahir Lemus Ramírez** — Universidad Industrial de Santander / SC3UIS
- **Johan Sebastian Galvis Beltrán** — Universidad Industrial de Santander / SC3UIS
- **Luis Alejandro Torres Niño** — Universidad Industrial de Santander / SC3UIS
- **Carlos Jaimes Barrios Hernandez** — Universidad Industrial de Santander / SC3UIS / LIG-INRIA Grenoble / INSA Lyon

---

*This work was carried out using the HPC infrastructure of the SC3UIS research group at the Universidad Industrial de Santander, Bucaramanga, Colombia.*
