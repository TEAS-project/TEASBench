#!/bin/python3

import yaml
import os
import re
import subprocess
from utils import get_run_name, k8s_friendlify, results_repo_dir, benchmark_family

DEFINED_SENTINEL = "<defined>"

class Template:
    def __init__(self):
        pass


    def matches(self, rule_match_condition, experiment_parameters):
        for parameter, rule_value in rule_match_condition.items():
            experiment_value = experiment_parameters.get(parameter)
            if rule_value == DEFINED_SENTINEL:
                if experiment_value is None or experiment_value == "":
                    return False
                continue
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
        return sum(1 if v == DEFINED_SENTINEL else 2 for v in match.values())
    

    def resolve_params(self, flags_key, matching_rules):
        resolved = {}
        for rule in matching_rules:
            if flags_key in rule:
                for param, value in rule[flags_key].items():
                    resolved[param] = value
        return resolved



    def resolve_generic_variable(self, var_name, config, matching_rules, parameters):
        # 1. Start with the default value
        value = config["variables_defaults"].get(var_name)

        engine = parameters.get('inference_engine')
        if isinstance(value, dict) and engine in value:
            value = value[engine]

        # 2. Accumulate values from all matching rules
        for rule in matching_rules:
            if var_name in rule:
                rule_val = rule[var_name]

                # Handle dictionary merging (existing logic)
                if isinstance(value, dict) and isinstance(rule_val, dict):
                    for k, v in rule_val.items():
                        # If the key exists and both are strings, concatenate them
                        if k in value and isinstance(value[k], str) and isinstance(v, str):
                            if v not in value[k]:
                                # Strip trailing/leading newlines and join with a single newline
                                value[k] = value[k].rstrip('\n') + "\n" + v.lstrip('\n')
                        # Otherwise, behave like a normal update
                        else:
                            value[k] = v

                # Handle String/Environment Variable Aggregation
                elif isinstance(value, str) and isinstance(rule_val, str):
                    # Ensure we don't duplicate exact blocks and maintain spacing
                    if rule_val not in value:
                        value = value.strip() + "\n" + rule_val.strip()

                # Handle list aggregation
                elif isinstance(value, list) and isinstance(rule_val, list):
                    value.extend(x for x in rule_val if x not in value)

                # Fallback for initial assignment if default was None/empty
                else:
                    value = rule_val

        if isinstance(value, dict) and "value" in value:
            value = value["value"]

        return "" if value is None else value


    def _build_flag(self, flag_def, param, value):
        flag_str = flag_def["flag"] if isinstance(flag_def, dict) else flag_def
        if value is True:
            return flag_str
        if value is not None and value != "":
            return f"{flag_str} {value}"
        return None


    def _resolve_flag_value(self, param, flag_def, parameters):
        """Determine the value to use for a flag in the flags loop.

        If the flag definition has 'from_parameter', look up that key in
        parameters; when the referenced parameter is missing or empty the
        flag is omitted (since _build_flag treats None and "" as 'omit').
        Otherwise, fall back to the parameter named after the flag itself,
        then to the flag's hard-coded 'value' default.
        """
        if isinstance(flag_def, dict):
            if "from_parameter" in flag_def:
                return parameters.get(flag_def["from_parameter"])
            return parameters.get(param, flag_def.get("value"))
        return parameters.get(param)

    
    def build_command(self, cmd_type, config, parameters, matching_rules):
        """Generalized command builder for server, client and agentic commands.

        'server' and 'agentic' are engine-keyed (sglang/vllm); 'client' is
        engine-agnostic. 'agentic_server' builds the standalone SGLang server
        command used by the new sglang agent_cap.agents path.
        """
        engine = parameters.get('inference_engine')

        # Handle server vs client vs agentic config paths
        if cmd_type == "server":
            flags_def = config["variables_defaults"]["server_flags"][engine]
            cmd_cfg = config["variables_defaults"]["server_command"][engine]
            rule_key = "server_flags"
        elif cmd_type == "agentic_server":
            flags_def = config["variables_defaults"]["agentic_server_flags"]
            cmd_cfg = config["variables_defaults"]["agentic_server_command"][engine]
            rule_key = "agentic_server_flags"
        elif cmd_type == "agentic":
            flags_def = config["variables_defaults"]["agentic_flags"]
            cmd_cfg = config["variables_defaults"]["agentic_command"][engine]
            rule_key = "agentic_flags"
        elif cmd_type == "agentic_server_sidecar":
            flags_def = config["variables_defaults"]["agentic_server_flags"]
            cmd_cfg = config["variables_defaults"]["agentic_server_sidecar_command"][engine]
            rule_key = "agentic_server_flags"
        elif cmd_type == "agentic_sidecar":
            flags_def = config["variables_defaults"]["agentic_flags"]
            cmd_cfg = config["variables_defaults"]["agentic_sidecar_command"][engine]
            rule_key = "agentic_flags"
        else:
            flags_def = config["variables_defaults"]["client_flags"]
            cmd_cfg = config["variables_defaults"]["client_command"]
            rule_key = "client_flags"

        cmd = cmd_cfg["base_command"] + " \\\n"
        
        # Base parameters
        for param in cmd_cfg["flags"]:
            if param not in flags_def: continue
            flag_def = flags_def[param]
            value = self._resolve_flag_value(param, flag_def, parameters)
            
            rendered = self._build_flag(flag_def, param, value)
            if rendered: cmd += f"  {rendered} \\\n"

        # Conditional overrides
        conditional = self.resolve_params(rule_key, matching_rules)
        for param, value in conditional.items():
            if param not in flags_def: continue
            rendered = self._build_flag(flags_def[param], param, value)
            if rendered: cmd += f"  {rendered} \\\n"

        return cmd.strip(" \\\n")

    
    def _moe(self, config, parameters, matching_rules):
        """MoE family: server+client split, engine-keyed image."""
        server_cmd = self.build_command("server", config, parameters, matching_rules)
        client_cmd = self.build_command("client", config, parameters, matching_rules)

        extra_env = self.resolve_generic_variable("extra_container_env", config, matching_rules, parameters)
        arena_dl = self.resolve_generic_variable("download_arena_hard_baseline_answers", config, matching_rules, parameters)

        # Construct Image Name: base + version + variant
        img_cfg = self.resolve_generic_variable("image", config, matching_rules, parameters)
        img_version = self.resolve_generic_variable("inference_engine_version", config, matching_rules, parameters)
        cuda_variant = img_cfg.get("cuda_variant", "") if isinstance(img_cfg, dict) else ""
        image_name = f"{img_cfg['base']}:v{img_version}{cuda_variant if cuda_variant else ''}"

        replacements = {
            "@image_name@": image_name,
            "@extra_container_env@": extra_env,
            "@download_arena_hard_baseline_answers@": arena_dl,
            "@server_start_command@": server_cmd,
            "@client_run_command@": client_cmd,
            "@hf_model_path@": parameters.get("hf_model_path"),
        }
        return "templates/template.yaml", replacements

    def _agentic(self, config, parameters, matching_rules):
        """Agentic family: separate server process + agent_cap.agents for all engines."""
        return self._agentic_agents(config, parameters, matching_rules)

    def _agentic_mcp(self, config, parameters, matching_rules):
        """sglang agentic path: standalone SGLang server + agent_cap.agents.

        Builds four commands:
          @agentic_server_command@  -- sglang.launch_server invocation
          @agentic_client_command@  -- python -m agent_cap.agents --config ...
          @side_car@ -- additional container
          @side_car_ready@ -- check side car ready to benchmark
        and routes to the agentic-agents.yaml template which starts the server
        in the background, waits for readiness, then runs the client.
        """
        server_cmd = self.build_command("agentic_server_sidecar", config, parameters, matching_rules)
        client_cmd = self.build_command("agentic_sidecar", config, parameters, matching_rules)

        image_name = self.resolve_generic_variable("agentcap_image", config, matching_rules, parameters)
        agentcap_repo = self.resolve_generic_variable("agentcap_repo", config, matching_rules, parameters)
        agentcap_ref = self.resolve_generic_variable("agentcap_ref", config, matching_rules, parameters)

        env_setup_path = self.resolve_generic_variable("agentic_env_setup_script", config, matching_rules, parameters)
        with open(env_setup_path, "r") as f:
            env_setup = f.read().strip()

        replacements = {
            "@image_name@": image_name,
            "@agentic_server_command@": server_cmd,
            "@agentic_client_command@": client_cmd,
            "@agentic_env_setup@": env_setup,
            "@agentcap_repo@": agentcap_repo,
            "@agentcap_ref@": agentcap_ref,
            "@model@": parameters.get("model"),
            "@hf_model_path@": parameters.get("hf_model_path"),
        }
        return "templates/agentic-sidecar.yaml", replacements

    def _agentic_agents(self, config, parameters, matching_rules):
        """sglang agentic path: standalone SGLang server + agent_cap.agents.

        Builds two commands:
          @agentic_server_command@  -- sglang.launch_server invocation
          @agentic_client_command@  -- python -m agent_cap.agents --config ...
        and routes to the agentic-agents.yaml template which starts the server
        in the background, waits for readiness, then runs the client.
        """
        server_cmd = self.build_command("agentic_server", config, parameters, matching_rules)
        client_cmd = self.build_command("agentic", config, parameters, matching_rules)

        image_name = self.resolve_generic_variable("agentcap_image", config, matching_rules, parameters)
        agentcap_repo = self.resolve_generic_variable("agentcap_repo", config, matching_rules, parameters)
        agentcap_ref = self.resolve_generic_variable("agentcap_ref", config, matching_rules, parameters)

        env_setup_path = self.resolve_generic_variable("agentic_env_setup_script", config, matching_rules, parameters)
        with open(env_setup_path, "r") as f:
            env_setup = f.read().strip()

        replacements = {
            "@image_name@": image_name,
            "@agentic_server_command@": server_cmd,
            "@agentic_client_command@": client_cmd,
            "@agentic_env_setup@": env_setup,
            "@agentcap_repo@": agentcap_repo,
            "@agentcap_ref@": agentcap_ref,
            "@model@": parameters.get("model"),
            "@hf_model_path@": parameters.get("hf_model_path")
        }
        return "templates/agentic-agents.yaml", replacements

    def get(self, parameters: dict, results_repo: str):
        with open("configs/config.yaml", "r") as f:
            config = yaml.safe_load(f)

        rules = config.get("rules", [])
        matching_rules = self.get_matching_rules(rules, parameters)

        teasbench_commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()

        # Substitutions common to every family
        replacements = {
            "@name_k8s@": k8s_friendlify(get_run_name(parameters)),
            "@inference_engine@": parameters.get('inference_engine'),
            "@num_gpu@": str(parameters.get("num_gpu")),
            "@gpu_product@": str(parameters.get("gpu_product")),
            "@results_repo@": results_repo,
            "@output_repo_dir@": results_repo_dir(parameters),
            "@teasbench_commit@": teasbench_commit,
        }

        if benchmark_family(parameters) == "agentic":
            print("test1")
            template_path, family_replacements = self._agentic(
                config, parameters, matching_rules)
        elif benchmark_family(parameters) == "agentic-sidecar":
            print("test2")
            template_path, family_replacements = self._agentic_mcp(
                config, parameters, matching_rules)
        else:
            template_path, family_replacements = self._moe(
                config, parameters, matching_rules)
        replacements.update(family_replacements)

        # Load the job template
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
