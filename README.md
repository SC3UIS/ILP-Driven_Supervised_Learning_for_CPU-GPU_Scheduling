# ILP-Driven Supervised Learning for Co-Optimizing Performance and Energy in CPU-GPU Architectures

> **CARLA 2026** — Latin American High Performance Computing Conference  
> Universidad Industrial de Santander (UIS) — [SC3UIS Research Group](https://github.com/SC3UIS)

---

## What is this project?

This repository implements a hybrid ILP+ML scheduling framework for heterogeneous CPU-GPU systems. The core idea is simple: instead of solving a computationally expensive Integer Linear Program at every scheduling decision, we solve it **once offline** to generate provably optimal CPU-GPU task assignments, then train supervised classifiers (Random Forest and XGBoost) to **replicate those decisions in near-constant time** at runtime.

The result is a scheduler that is:
- **Optimal-quality** — decisions are grounded in ILP-optimal labels
- **Fast** — ML inference replaces online optimization
- **Energy-aware** — the ILP objective jointly minimizes latency and energy consumption

The methodology is validated on three representative HPC kernels across two hardware platforms:

| Kernel | Type | Libraries |
|---|---|---|
| GEMM | Compute-bound | cuBLAS / MKL |
| SpMV | Memory-bound | cuSPARSE / MKL Inspector-Executor |
| FFT | Mixed intensity | cuFFT / MKL DFTI |

| Platform | Alias | CPU | GPU |
|---|---|---|---|
| HPE ProLiant XL290n G10+ | PACCA | Intel Xeon Gold 5315Y (3.2 GHz) | NVIDIA A100 40 GB |
| HPE ProLiant DL580 G9 | THOR | Intel Xeon E5-4640 (2.67 GHz) | NVIDIA Tesla M40 24 GB |

---

## How to Use

### Prerequisites

**Python (≥ 3.9):**
```bash
pip install pulp pandas numpy scikit-learn xgboost optuna matplotlib seaborn
```

**C++/CUDA (per platform):**

| Platform | Compiler | Required libraries |
|---|---|---|
| NVIDIA GPU | `nvcc` (CUDA 12.x) | cuBLAS, cuSPARSE, cuFFT, NVML |
| Intel CPU | `icpx` or `g++` | Intel MKL (oneAPI 2024) |

---

### Step 1 — Compile and Run Benchmarks

#### NVIDIA GPU (PACCA)
```bash
nvcc -O3 -arch=sm_52 -std=c++11  src/benchmark_gpu_nvidia.cu -L/usr/lib64 -lnvidia-ml -lcublas -lcusparse -lcufft -lpthread -o bin/benchmark_titanx_2
```

#### NVIDIA GPU (THOR)
```bash
nvcc -O3 -arch=sm_52 -allow-unsupported-compiler benchmark_gpu_nvidia.cu -L/usr/lib64 -lnvidia-ml -lcublas -lcusparse -lcufft -lpthread -o benchmark_titanx
```

#### Intel CPU (PACCA)
```bash
g++ -O3 -march=native -std=c++17 src/benchmark_cpu_intel.cpp -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -o bin/cpu_intel_dataset
```

#### Intel CPU (THOR)
```bash
g++ -O3 -march=native -std=c++17 src/benchmark_cpu_intel.cpp -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -fopenmp -lpthread -lm -ldl -o bin/cpu_intel_dataset_16
```

---

### Step 2 — Generate ILP Labels

Each ILP script reads a merged CPU+GPU results CSV for its platform, solves the binary assignment problem for each CPU-core configuration, and appends the `ilp_label` column (0 = CPU, 1 = GPU):

```bash
# PACCA platform
python src/ILP_pacca.py --input results_pacca.csv

# THOR platform
python src/ILP_thor.py --input results_thor.csv
```

The solver prints a per-kernel assignment summary to stdout:

```
  ── 8 cores ──────────────────────────────────────────────
        Total  CPU  GPU  GPU_%  Speedup
kernel
FFT        28   25    3   10.7    14.66
GEMM       33   20   13   39.4    19.02
SpMV       79   77    2    2.5    10.96
  Total:  18 GPU (12.9%)  /  122 CPU (87.1%)
  Z obj : 7.865   |  Solve time: 0.043s
```

> Pre-generated labeled datasets are available in [`data/`](data/) — skip this step if you want to use them directly.

---

### Step 3 — Train ML Models

Merge both labeled datasets and run the ML pipeline:

```bash
# Train Random Forest and XGBoost with Optuna HPO (100 trials each)
python src/ML_models.py --input results.csv
```

The script outputs five diagnostic figures. Pre-generated versions are in [`data/`](data/):

| Figure | Description | Link |
|---|---|---|
| `fig1_model_evaluation.png` | Confusion matrices, ROC/PR curves, metric comparison | [view](data/fig1_model_evaluation.png) |
| `fig2_feature_importance.png` | RF MDI vs XGBoost importance by gain | [view](data/fig2_feature_importance.png) |
| `fig3_learning_dynamics.png` | Learning curves and per-kernel accuracy | [view](data/fig3_learning_dynamics.png) |
| `fig4_confidence_analysis.png` | Prediction confidence vs speedup ratio | [view](data/fig4_confidence_analysis.png) |
| `fig5_optuna_history.png` | Optuna HPO history and fANOVA hyperparameter importance | [view](data/fig5_optuna_history.png) |

---

## Results Summary

| Model | Accuracy | F1 Macro | ROC-AUC | CV-AUC |
|---|---|---|---|---|
| Random Forest | 0.944 | 0.805 | 0.896 | 0.939 ± 0.035 |
| XGBoost | 0.913 | 0.770 | 0.935 | 0.939 ± 0.030 |

Per-kernel accuracy: **FFT** 1.000 / 0.958 — **SpMV** 0.976 / 0.976 — **GEMM** 0.865 / 0.788 (RF / XGBoost).

Kernel performance profiles are available in [`data/`](data/):

| Kernel | PACCA | THOR |
|---|---|---|
| GEMM | [view](data/GEMM-PACCA.png) | [view](data/GEMM-THOR.jpeg) |
| SpMV (sparsity=1e-5) | [view](data/SPMV2-PACCA.png) | [view](data/SPMV2-THOR.jpeg) |
| SpMV (sparsity=1e-2) | [view](data/SPMV5-PACCA.png) | [view](data/SPMV5-THOR.jpeg) |
| FFT | [view](data/FFT-PACCA.png) | [view](data/FFT-THOR.jpeg) |

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
