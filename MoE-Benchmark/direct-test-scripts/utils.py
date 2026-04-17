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
    "DeepSeek-R1": "deepseek-r1",
    "Kimi-K2.5": "kimi-k2.5"
}

DATASET_SHORT_NAME_MAP={
    "gsm8k": "gsm8k",
    "arena-hard":, "arena-hard",
    "longbench_v1": "longbench"
    }

HF_MODEL_MAP={
    "gpt-oss-20b": "unsloth/gpt-oss-20b",
    "gpt-oss-120b": "unsloth/gpt-oss-120b",
    "Qwen3-235B-A22B-Instruct-2507": "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "DeepSeek-R1": "deepseek-ai/DeepSeek-R1",
    "Kimi-K2.5": "moonshotai/Kimi-K2.5"
}

VLLM_REASONING_PARSER_MAP={
    "gpt-oss-20b": False,
    "gpt-oss-120b": False,
    "Qwen3-235B-A22B-Instruct-2507": False,
    "DeepSeek-R1": "deepseek_r1",
    "Kimi-K2.5": "kimi_k2"
}


EIDF_GPU_MAP={
    "A100":"NVIDIA-A100-SXM4-80GB",
    "H100":"NVIDIA-H100-80GB-HBM3",
    "H200":"NVIDIA-H200"
}


def get_run_name(inference_engine, model, dataset, num_samples, gpu, num_gpu, batch_size):
    name = f"{inference_engine}_{MODEL_SHORT_NAME_MAP[model]}_{DATASET_SHORT_NAME[dataset]}_ns{num_samples}_{gpu}x{num_gpu}"
    name += f"_bs{batch_size}"
    return name

def k8s_friendlify(unfriendly_string):
    return unfriendly_string.replace("_", "-").lower()

def results_repo_dir(inference_engine, model, dataset, num_samples, gpu, num_gpu, batch_size):
    dir = f"moe/eidf/{inference_engine}/{model}/{dataset}_{num_samples}samples/{gpu.lower()}x{num_gpu}"
    if batch_size == "default":
        dir += f"/batch-size-default"
    else:
        dir += f"/batch-size-{batch_size}"
    return dir


