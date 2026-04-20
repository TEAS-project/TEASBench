#!/bin/python3

import os
from datetime import datetime
import yaml
import re
from utils import get_run_name, k8s_friendlify, results_repo_dir, VLLM_REASONING_PARSER_MAP

class Template:
    def __init__(self):
        return

    def get(self,
            inference_engine: str,
            model: str,
            hf_model_path: str,
            dataset: str,
            num_samples: int,
            gpu: str,
            gpu_product: str,
            num_gpu: int,
            tensor_parallel_size: int,
            batch_size: str,
            results_repo: str): 
        
        name = get_run_name(inference_engine, model, dataset, num_samples, gpu, num_gpu, batch_size)
        name_k8s = k8s_friendlify(name)

        # Load the generic (inference-engine-agnostic) template
        template_path = os.path.join("yaml_templates/template.yaml")
        with open(template_path, "r") as f:
            template = f.read()
        
        # Load the inference-engine-specific template config
        config_path = os.path.join("yaml_templates", f"{inference_engine}-config.yaml")
        with open(config_path, "r") as f:
            engine_config = yaml.safe_load(f)

        # Combine any information from config to make up replacements
        image = engine_config.pop('image')
        image_name = image['base'] + ":v" \
            + engine_config['inference_engine_version'] \
            + image['cuda_variant']['EIDF'][gpu]
        
        
        conditional_flags = engine_config['conditional_flags']
        extra_server_flags=""
        extra_client_flags=""

        # Batch size
        if batch_size != "default":
            server_set_batch_size = conditional_flags['server_set_batch_size'].replace("@batch_size@", str(batch_size))
            client_notify_batch_size = conditional_flags['client_notify_batch_size'].replace("@batch_size@", str(batch_size))
            extra_server_flags += f"\\\n{server_set_batch_size}"
            extra_client_flags += f"\\\n{client_notify_batch_size}"

        # MoE-CAP Reasoning parser
        if inference_engine == "vllm" and VLLM_REASONING_PARSER_MAP[model] != False:
            reasoning_parser = conditional_flags['reasoning_parser'].replace("@reasoning_parser@", str(VLLM_REASONING_PARSER_MAP[model]))
            extra_server_flags += f"\\\n{reasoning_parser}"
            
        # Inject config into template 
        config = template
        if dataset == "arena-hard":
            engine_config['client_run_command'] = engine_config['client_run_command_arena-hard']
            
        for key, value in engine_config.items():
            placeholder = "{{ " + key + " }}"
            replacement = str(value).rstrip('\n') if value is not None else ""

            #Find any leading whitespace before the placeholder to preserve indentation
            match = re.search(r'^([ \t]*)' + re.escape(placeholder), template, flags=re.MULTILINE)
            if match and '\n' in replacement:
                indent_spaces = match.group(1)
                # Replace newlines in the replacement string with a newline + the correct indentation
                replacement = replacement.replace('\n', '\n' + indent_spaces)

            config = config.replace(placeholder, replacement)


        output_repo_dir = results_repo_dir(inference_engine, model, dataset, num_samples, gpu, num_gpu, batch_size)
        
        # Inject experiment parameters 
        replacements = {
            "@inference_engine@": str(inference_engine),
            "@image_name@": str(image_name),
            "@hf_model_path@": str(hf_model_path),
            "@model@": str(model),
            "@dataset@": str(dataset),
            "@num_samples@": str(num_samples),
            "@gpu@": str(gpu.lower()),
            "@gpu_product@": str(gpu_product),
            "@num_gpu@": str(num_gpu),
            "@tensor_parallel_size@": str(tensor_parallel_size),
            "@batch_size@": str(batch_size),
            "@name_k8s@": str(name_k8s),
            "@extra_server_flags@": str(extra_server_flags),
            "@extra_client_flags@": str(extra_client_flags),
            "@output_repo_dir@": str(output_repo_dir),
            "@results_repo@": str(results_repo),
            "@reasoning_parser@": str(engine_config.pop('reasoning_parser')[model])
        }

        
        for placeholder, actual_value in replacements.items():
            # Match the start of the line (^), capture the spaces ([ \t]*), 
            # and match any characters (.*?) up to the placeholder.
            match = re.search(r'^([ \t]*).*?' + re.escape(placeholder), config, flags=re.MULTILINE)
            if match and '\n' in actual_value:
                indent_spaces = match.group(1)
                # Replace newlines in the replacement string with a newline + the correct indentation
                actual_value = actual_value.replace('\n', '\n' + indent_spaces)

            config = config.replace(placeholder, actual_value)

        return config

            
