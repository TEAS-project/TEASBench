#!/bin/python3

import os

class Template:
    def __init__(self):
        return

    def get(self, \
        tensor_parallel_size: int, \
        num_gpu: int, \
        gpu_product: str, \
        completions: int, \
        line_array: str, \
        filename: str):

        gpu=gpu_product.split("-")[1] 

        with open(os.path.expanduser("yaml_templates/template_sglang_loop.yaml"), "r") as f:
            template = f.read()

        template = template.replace("@tensor_parallel_size@", str(tensor_parallel_size))
        template = template.replace("@num_gpu@", str(num_gpu))
        template = template.replace("@gpu_product@", str(gpu_product))
        template = template.replace("@completions@", str(completions))
        template = template.replace("@line_array@", str(line_array))
        template = template.replace("@filename@", str(filename))
    
        return template
