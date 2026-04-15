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
            results_repo: str): # batch size str to accommodate "default"
        
        name = get_run_name(inference_engine, model, dataset, num_samples, gpu, num_gpu, batch_size)
        name_k8s = k8s_friendlify(name)

        # Load the inference-engine-specific template config
        config_path = os.path.join("yaml_templates", f"{inference_engine}-config.yaml")
        with open(config_path, "r") as f:
            inference_engine_config = yaml.safe_load(f)

        # Load the generic (inference-engine-agnostic) template
        template_path = os.path.join("yaml_templates/template.yaml")
        with open(template_path, "r") as f:
            template = f.read()

        # Inject inference-engine-specific config into template 
        # Replaces {{ key }} in template with the value from the inference engine config file
        config = template
        for key, value in inference_engine_config.items():
            placeholder = "{{ " + key + " }}"
            replacement = str(value).rstrip('\n') if value is not None else ""

            #Find any leading whitespace before the placeholder to preserve indentation
            match = re.search(r'^([ \t]*)' + re.escape(placeholder), template, flags=re.MULTILINE)
            if match and '\n' in replacement:
                indent_spaces = match.group(1)
                # Replace newlines in the replacement string with a newline + the correct indentation
                replacement = replacement.replace('\n', '\n' + indent_spaces)

            config = config.replace(placeholder, replacement)

            
        # Extract definitions of any conditional flags and decide whether/how to include these based on experiment parameters
        conditional_flags = inference_engine_config.get("conditional_flags", {})
        extra_server_flags=""
        extra_client_flags=""

        if batch_size != "default":
            server_set_batch_size = conditional_flags.get('server_set_batch_size').replace("@batch_size@", str(batch_size))
            client_notify_batch_size = conditional_flags.get('client_notify_batch_size').replace("@batch_size@", str(batch_size))
            extra_server_flags += f"\\\n{server_set_batch_size}"
            extra_client_flags += f"\\\n{client_notify_batch_size}"

        if inference_engine == "vllm" and VLLM_REASONING_PARSER_MAP[model] != False:
            reasoning_parser = conditional_flags.get('reasoning_parser').replace("@reasoning_parser@", str(VLLM_REASONING_PARSER_MAP[model]))
            extra_server_flags += f"\\\n{reasoning_parser}"
            

        timestamp = datetime.now().strftime('%Y%m%d-%H%M')
        output_repo_dir = results_repo_dir(inference_engine, model, dataset, num_samples, gpu, num_gpu, batch_size)
            
        # Inject experiment parameters 
        replacements = {
            "@hf_model_path@": str(hf_model_path),
            "@model@": str(model),
            "@dataset@": str(dataset),
            "@num_samples@": str(num_samples),
            "@gpu@": str(gpu),
            "@gpu_product@": str(gpu_product),
            "@num_gpu@": str(num_gpu),
            "@tensor_parallel_size@": str(tensor_parallel_size),
            "@batch_size@": str(batch_size),
            "@name_k8s@": str(name_k8s),
            "@extra_server_flags@": str(extra_server_flags),
            "@extra_client_flags@": str(extra_client_flags),
            "@output_repo_dir@": str(output_repo_dir),
            "@timestamp@": str(timestamp),
            "@results_repo@": str(results_repo),
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

    
