#!/bin/python3

import yaml
import os
import re
from utils import get_run_name, k8s_friendlify, results_repo_dir

class Template:
    def __init__(self):
        pass

    def matches(self, rule_match_condition, experiment_parameters):
        for parameter, rule_value in rule_match_condition.items():
            experiment_value = experiment_parameters.get(parameter)
            if isinstance(rule_value, list):
                if experiment_value not in rule_value:
                    return False
            else:
                if experiment_value != rule_value:
                    return False
        return True

    
    def get_matching_rules(self, rules, experiment_parameters):
        matching = []
        for rule in rules:
            if self.matches(rule["match"], experiment_parameters):
                score = self.specificity(rule["match"])
                matching.append((score, rule))
        matching.sort(key=lambda x: x[0])
        return [rule for _, rule in matching]

    
    def specificity(self, match):
        return len(match)

    
    def resolve_params(self, flags_key, matching_rules):
        resolved = {}
        for rule in matching_rules:
            if flags_key in rule:
                for param, value in rule[flags_key].items():
                    resolved[param] = value
        return resolved

    
    def resolve_generic_variable(self, var_name, config, matching_rules, parameters):
        """
        Resolves a non-command variable by checking defaults and applying rules.
        """
        # 1. Start with the default value from variables_defaults
        value = config["variables_defaults"].get(var_name)
        
        # If it's an engine-specific dict, narrow it down
        engine = parameters.get('inference_engine')
        if isinstance(value, dict) and engine in value:
            value = value[engine]

        # 2. Overwrite with values from matching rules
        for rule in matching_rules:
            if var_name in rule:
                rule_val = rule[var_name]
                # If rule_val is a dict, merge it; otherwise overwrite
                if isinstance(value, dict) and isinstance(rule_val, dict):
                    value.update(rule_val)
                else:
                    value = rule_val
        
        # If the result is a dict with a 'default_value' key, simplify it
        if isinstance(value, dict) and "default_value" in value:
            value = value["default_value"]
            
        return "" if value is None else value

    
    def _build_flag(self, flag_def, param, value):
        flag_str = flag_def["flag"] if isinstance(flag_def, dict) else flag_def
        if value is True:
            return flag_str
        if value is not None and value != "":
            return f"{flag_str} {value}"
        return None

    
    def build_command(self, cmd_type, config, parameters, matching_rules):
        """Generalized command builder for both server and client."""
        engine = parameters.get('inference_engine')
        
        # Handle server vs client config paths
        if cmd_type == "server":
            flags_def = config["variables_defaults"]["server_flags"][engine]
            cmd_cfg = config["variables_defaults"]["server_command"][engine]
            rule_key = "server_flags"
        else:
            flags_def = config["variables_defaults"]["client_flags"]
            cmd_cfg = config["variables_defaults"]["client_command"]
            rule_key = "client_flags"

        cmd = cmd_cfg["base_command"] + " \\\n"
        
        # Base parameters
        for param in cmd_cfg["base_parameters"]:
            if param not in flags_def: continue
            flag_def = flags_def[param]
            value = parameters.get(param, flag_def.get("default_value") if isinstance(flag_def, dict) else None)
            
            rendered = self._build_flag(flag_def, param, value)
            if rendered: cmd += f"  {rendered} \\\n"

        # Conditional overrides
        conditional = self.resolve_params(rule_key, matching_rules)
        for param, value in conditional.items():
            if param not in flags_def: continue
            rendered = self._build_flag(flags_def[param], param, value)
            if rendered: cmd += f"  {rendered} \\\n"

        return cmd.strip(" \\\n")

    
    def get(self, parameters: dict, results_repo: str):
        with open("yaml_templates/config.yaml", "r") as f:
            config = yaml.safe_load(f)

        rules = config.get("rules", [])
        matching_rules = self.get_matching_rules(rules, parameters)
        engine = parameters.get('inference_engine')

        # Resolve complex commands
        server_cmd = self.build_command("server", config, parameters, matching_rules)
        client_cmd = self.build_command("client", config, parameters, matching_rules)

        # Resolve static variables
        extra_env = self.resolve_generic_variable("extra_container_env", config, matching_rules, parameters)
        arena_dl = self.resolve_generic_variable("download_arena_hard_baseline_answers", config, matching_rules, parameters)
        
        # Construct Image Name: base + version + variant
        img_cfg = self.resolve_generic_variable("image", config, matching_rules, parameters)
        img_version = self.resolve_generic_variable("inference_engine_version", config, matching_rules, parameters)
        cuda_variant = img_cfg.get("cuda_variant", "") if isinstance(img_cfg, dict) else ""
        image_name = f"{img_cfg['base']}:v{img_version}{cuda_variant if cuda_variant else ''}"

        # Prepare replacements for substitution in template.yaml
        replacements={
            "@name_k8s@": k8s_friendlify(get_run_name(parameters)),
            "@inference_engine@": parameters.get('inference_engine'),
            "@image_name@": image_name,
            "@extra_container_env@": extra_env,
            "@download_arena_hard_baseline_answers@": arena_dl,
            "@server_start_command@": server_cmd,
            "@client_run_command@": client_cmd,
            "@hf_model_path@": parameters.get("hf_model_path"),
            "@num_gpu@": str(parameters.get("num_gpu")),
            "@gpu_product@": str(parameters.get("gpu_product")),
            "@results_repo@": results_repo,
            "@output_repo_dir@": results_repo_dir(parameters)
        }


        # Load the job template
        template_path = os.path.join("yaml_templates/template.yaml")
        with open(template_path, "r") as f:
            template = f.read()

        job_config = template    
            
        for placeholder, actual_value in replacements.items():
            # Match the start of the line (^), capture the spaces ([ \t]*), 
            # and match any characters (.*?) up to the placeholder.
            match = re.search(r'^([ \t]*).*?' + re.escape(placeholder), job_config, flags=re.MULTILINE)
            if match and '\n' in actual_value:
                indent_spaces = match.group(1)
                # Replace newlines in the replacement string with a newline + the correct indentation
                actual_value = actual_value.replace('\n', '\n' + indent_spaces)

            job_config = job_config.replace(placeholder, actual_value)

        
        return job_config

    
