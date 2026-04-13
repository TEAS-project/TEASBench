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

HF_MODEL_MAP={
    "gpt-oss-20b": "unsloth/gpt-oss-20b",
    "gpt-oss-120b": "unsloth/gpt-oss-120b",
    "Qwen3-235B-A22B": "Qwen/Qwen3-235B-A22B",
    "DeepSeek-R1": "deepseek-ai/DeepSeek-R1",
    "Kimi-K2.5": "moonshotai/Kimi-K2.5"
}


EIDF_GPU_MAP={
    "A100":"NVIDIA-A100-SXM4-80GB",
    "H100":"NVIDIA-H100-80GB-HBM3",
    "H200":"NVIDIA-H200"
}


def get_run_name(inference_engine, model, dataset, num_samples, gpu, num_gpu, batch_size):
    name = f"{inference_engine}_{model}_{dataset}_ns{num_samples}_{gpu}x{num_gpu}"
    if batch_size != "default":
        name += f"_bs{batch_size}"
    return name

def k8s_friendlify(unfriendly_string):
    return unfriendly_string.replace("_", "-").lower()

def results_repo_dir(inference_engine, model, dataset, num_samples, gpu, num_gpu, batch_size):
    dir = f"moe/eidf/{inference_engine}/{model}/{dataset}_{num_samples}samples/{gpu}x{num_gpu}"
    if batch_size != "default":
        dir += f"_batchsize{batch_size}"
    return dir


