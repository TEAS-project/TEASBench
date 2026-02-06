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
GPU_MAP={
    "A100":"NVIDIA-A100-SXM4-80GB",
    "H100":"NVIDIA-H100-80GB-HBM3",
    "H200":"NVIDIA-H200",
}

TOKEN_LENGTH_MAP={
    "1K": 1000,
    "4K": 4000,
    "13K": 13000
}

def get_run_name(model_name, gpu, num_gpu, target_input_tokens, target_output_tokens, batch_size, dataset, token_abbrev=True):
    model_name_clean=model_name.split("/")[1].replace(".", "-")

    if token_abbrev: # 4K
        token_in = target_input_tokens
        token_out = target_output_tokens
    else: # 4000
        token_in = TOKEN_LENGTH_MAP[target_input_tokens]
        token_out = TOKEN_LENGTH_MAP[target_output_tokens]

    run_name=f"{model_name_clean}_{gpu}x{num_gpu}_{token_in}_{token_out}_bs{batch_size}_{dataset}"
    return run_name