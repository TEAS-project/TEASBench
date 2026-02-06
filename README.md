# TEAS-Bench

Uniting Models, Algorithms, and System Innovators with Top-Down Evolutionary Benchmarks.

🌐 **Website:** [www.teasbench.com](https://www.teasbench.com)

## Benchmark Categories

| Category | Description |
|----------|-------------|
| **MoE-Benchmark** | Benchmarks for Mixture-of-Experts models |
| **TTS-Benchmark** | Benchmarks for Test-Time Scaling methods |
| **Agentic-Benchmark** | Benchmark for Agentic Workflows (Under construction) |

## Core Components

Most of our benchmarking code is developed in the following projects:

| Component | Repository | Description |
|-----------|------------|-------------|
| **MoE-CAP** | [GitHub](https://github.com/Auto-CAP/MoE-CAP.git) | MoE benchmarking framework (~4K LoC, Python) |
| **AgentCAP**| [GitHub](https://github.com/Jingxue9/AgentCAP) | Agentic Workflow benchmarking framework (currently ~3K LoC, Python) |
| **ServerlessLLM + Pylet** | [GitHub](https://github.com/ServerlessLLM/ServerlessLLM/tree/v1-beta) | Benchmark platform (~103K LoC, Python + C++) |

## Quick Start

### Option 1: Direct Testing

Run benchmarks directly using Python scripts without any infrastructure setup.

> **Note:** Direct test scripts are developed and tested on Kubernetes clusters with NVIDIA GPU support. They currently only work on Kubernetes environments.

**Prerequisites:**
- Python 3.10+
- GPU with sufficient VRAM for the model

**Installation:**

**For SGLang backend:**
```bash
conda create -n sglang python=3.10 -y
conda activate sglang
pip install sglang==0.5.8 transformers datasets
```

**For vLLM backend:**
```bash
conda create -n vllm python=3.10 -y
conda activate vllm
pip install vllm==0.11.0 transformers==4.56.0 datasets
```

### Option 2: Serverless Deployment

For Kubernetes-based serverless deployment with auto-scaling, see the detailed guide:

**[Serverless Deployment Guide](serverless_deployments/README.md)**

> **Note:** Serverless scripts are developed and tested on Kubernetes clusters with NVIDIA GPU support. They currently only work on Kubernetes environments.

This approach is recommended for:
- Production deployments
- Multi-model serving
- Auto-scaling based on load
- Kubernetes-native infrastructure

## Contributing

1. Add new benchmark scripts to the appropriate category folder
2. For direct tests, add to `<Category>/direct-test-scripts/`
3. For serverless tests, add to `<Category>/serverless-scripts/`
4. Update documentation as needed