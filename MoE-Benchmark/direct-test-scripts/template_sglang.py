#!/bin/python3

import os
from utils import get_run_name

class Template:
    def __init__(self):
        return

    def get(self, \
        model_name: str, \
        tensor_parallel_size: int, \
        dataset: str, \
        target_input_tokens: int, \
        target_output_tokens: int, \
        num_samples: int, \
        batch_size: int, \
        num_gpu: int, \
        gpu_product: str):

        model_name_clean=model_name.split("/")[1].replace(".", "-")
        gpu=gpu_product.split("-")[1] 

        run_name = get_run_name("sglang", model_name, gpu_product, num_gpu, target_input_tokens, target_output_tokens, batch_size, dataset)
        run_name_lower = run_name.replace("_", "-").lower()

        with open(os.path.expanduser("yaml_templates/template_sglang.yaml"), "r") as f:
          template = f.read()

        template = template.replace("@model_name@", str(model_name))
        template = template.replace("@tensor_parallel_size@", str(tensor_parallel_size))
        template = template.replace("@dataset@", str(dataset))
        template = template.replace("@num_gpu@", str(num_gpu))
        template = template.replace("@target_input_tokens@", str(target_input_tokens))
        template = template.replace("@target_output_tokens@", str(target_output_tokens))
        template = template.replace("@num_samples@", str(num_samples))
        template = template.replace("@gpu_product@", str(gpu_product))
        template = template.replace("@batch_size@", str(batch_size))
        template = template.replace("@run_name@", str(run_name))
        
        return template


