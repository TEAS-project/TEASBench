# TEAS-Bench

Uniting Models, Algorithms, and System Innovators with Top-Down Evolutionary Benchmarks.

🌐 **Website:** [www.teasbench.com](https://www.teasbench.com)

## Benchmark Categories

| Category | Description |
|----------|-------------|
| **MoE-Benchmark** | Benchmarks for Mixture-of-Experts models |
| **TTS-Benchmark** | Benchmarks for Test-Time Scaling methods |

> **Note:** Currently tested on Kubernetes clusters with NVIDIA GPU support.

## Prerequisites

- Kubernetes cluster with GPU nodes
- `kubectl` configured to access your cluster
- PersistentVolumeClaim named `model-pvc` for model storage
- Docker installed (for building custom benchmark containers)

## Quick Start

### Step 1: Build Your Docker Container

Create a Docker container with your testing scripts and ServerlessLLM installed.

Your container must have ServerlessLLM installed:

```bash
pip install serverlessllm
```

**Supported Backends:**
- `vllm` - Standard vLLM backend
- `sglang` - Standard SGLang backend  
- `moe-cap-sglang` - MoE-CAP optimized SGLang backend
- `moe-cap-vllm` - MoE-CAP optimized vLLM backend

### Step 2: Create Model Deployment Config

Add a deployment configuration JSON file like those in `config_examples/`. This defines how your model will be deployed.

See example: [`config_examples/sglang-moe-qwen3-30b-1worker-4gpu.json`](config_examples/sglang-moe-qwen3-30b-1worker-4gpu.json)

**Config Fields:**
| Field | Description |
|-------|-------------|
| `model` | HuggingFace model name |
| `backend` | One of: `vllm`, `sglang`, `moe-cap-sglang`, `moe-cap-vllm` |
| `num_gpus` | Number of GPUs per model instance |
| `tensor_parallel_size` | Tensor parallelism degree |
| `max_instances` | Maximum number of model instances that can run concurrently |
| `keep_alive` | Time in seconds the instance stays alive before being deleted after idle |

**Note on `max_instances`:** This depends on your deployment's total GPU capacity. For example, if you have 4 workers × 2 GPUs each (8 GPUs total) and each model instance requires 2 GPUs, you can set `max_instances = 4`.


### Step 3: Write K8s Benchmark Job YAML

Create a Kubernetes Job YAML in `MoE-Benchmark/` that runs your benchmark.

See example: [`MoE-Benchmark/sglang-qwen3-30b-moe-gsm8k.yaml`](MoE-Benchmark/sglang-qwen3-30b-moe-gsm8k.yaml)

### Step 4: Deploy the ServerlessLLM Infrastructure

Deploy the SLLM infrastructure (head nodes and GPU workers):

**Example: 1 Worker with 4 H100 GPUs**

```bash
kubectl apply -f deployments/sllm-deployment-1-worker-4-H100.yaml
```

Wait for all pods to be ready:

```bash
kubectl get pods -w
```

You should see:
- `pylet-head-xxx` - Running
- `sllm-head-xxx` - Running  
- `pylet-worker-0-xxx` - Running

### Step 5: Run the Benchmark

Once all deployments are ready, run the benchmark job:

```bash
kubectl create -f MoE-Benchmark/sglang-qwen3-30b-moe-gsm8k.yaml
```

The benchmark job will:
1. Wait for the head nodes to be ready
2. Deploy the Qwen3-30B-A3B MoE model
3. Run evaluation on the GSM8K dataset
4. Save results to `/models/moecap_results`

### Step 6: Check Results

Monitor the benchmark progress:

```bash
kubectl logs -f job/sllm-benchmark-xxxxx
```

Results are saved to the PVC at `/models/moecap_results`.

## Cleanup

To remove the deployment:

```bash
kubectl delete -f deployments/sllm-deployment-1-worker-4-H100.yaml
kubectl delete job -l app=sllm-benchmark
```