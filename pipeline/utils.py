#!/bin/python3

"""
Supported GPU products on EIDF:
    nvidia.com/gpu.product: 'NVIDIA-A100-SXM4-80GB'
    nvidia.com/gpu.product: 'NVIDIA-A100-SXM4-40GB'
    nvidia.com/gpu.product: 'NVIDIA-A100-SXM4-40GB-MIG-3g.20gb'
    nvidia.com/gpu.product: 'NVIDIA-A100-SXM4-40GB-MIG-1g.5gb'
    nvidia.com/gpu.product: 'NVIDIA-H100-80GB-HBM3'
    nvidia.com/gpu.product: 'NVIDIA-H200'
"""

MODEL_SHORT_NAME_MAP={
    "gpt-oss-20b": "gpt-oss-20b",
    "gpt-oss-120b": "gpt-oss-120b",
    "Qwen3-235B-A22B-Instruct-2507": "qwen3-235b",
    "Qwen3-235B-A22B-Instruct-2507-FP8": "qwen3-235b-fp8",
    "DeepSeek-R1": "deepseek-r1",
    "Kimi-K2.5": "kimi-k2.5"
}

DATASET_SHORT_NAME_MAP={
    "gsm8k": "gsm8k",
    "arena-hard": "arena-hard",
    "longbench_v1": "longbench"
    }

HF_MODEL_MAP={
    "gpt-oss-20b": "unsloth/gpt-oss-20b",
    "gpt-oss-120b": "unsloth/gpt-oss-120b",
    "Qwen3-235B-A22B-Instruct-2507": "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "Qwen3-235B-A22B-Instruct-2507-FP8": "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
    "DeepSeek-R1": "deepseek-ai/DeepSeek-R1",
    "Kimi-K2.5": "moonshotai/Kimi-K2.5"
}


EIDF_GPU_MAP={
    "A100":"NVIDIA-A100-SXM4-80GB",
    "H100":"NVIDIA-H100-80GB-HBM3",
    "H200":"NVIDIA-H200"
}

def get_run_name(p: dict):
    name = f"{p['inference_engine']}_{MODEL_SHORT_NAME_MAP[p['model']]}_{DATASET_SHORT_NAME_MAP[p['dataset']]}_ns{p['num_samples']}_{p['gpu']}x{p['num_gpu']}"
    name += f"_bs{p['batch_size']}"
    return name

def k8s_friendlify(unfriendly_string):
    return unfriendly_string.replace("_", "-").lower()

def results_repo_dir(p: dict):
    dir = f"moe/eidf/{p['inference_engine']}/{p['model'].lower()}/{p['dataset']}_{p['num_samples']}samples/{p['gpu'].lower()}x{p['num_gpu']}"
    if p['batch_size'] == "default":
        dir += f"/batch-size-default"
    else:
        dir += f"/batch-size-{p['batch_size']}"
    return dir


