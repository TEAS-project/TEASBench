#!/bin/python3

import os
from utils import get_run_name, k8s_friendlify

class Template:
    def __init__(self):
        return

    def get(self, \
        model: str, \
        hf_model_path: str, \
        dataset: str, \
        num_samples: int
        gpu: str, \
        gpu_product: str, \
        num_gpu: int, \
        tensor_parallel_size: int, \
        batch_size: str): # batch size str to accommodate "default"

        name = get_run_name("sglang", model, dataset, num_samples, gpu, num_gpu, batch_size)
        
        name_k8s = k8s_friendlify(name)

        with open(os.path.expanduser("yaml_templates/template_sglang.yaml"), "r") as f:
          template = f.read()

        template = template.replace("@hf_model_path@", str(hf_model_path))
        template = template.replace("@model@", str(model))
        template = template.replace("@tensor_parallel_size@", str(tensor_parallel_size))
        template = template.replace("@dataset@", str(dataset))
        template = template.replace("@num_gpu@", str(num_gpu))
        template = template.replace("@num_samples@", str(num_samples))
        template = template.replace("@gpu@", str(gpu))
        template = template.replace("@gpu_product@", str(gpu_product))
        template = template.replace("@name_k8s@", str(name_k8s))
        
        return template


