#!/bin/python3

import yaml
import os
import re
import subprocess
from utils import needs_login_node_driver, swe_bench_lite_k8s, get_run_name, k8s_friendlify, results_repo_dir, benchmark_family, TEAS_GPU_NAME_MAP, PVC_ARCHIVE_DIR

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
            if isinstance(rule_value, dict) and "not" in rule_value:
                excluded = rule_value["not"]
                if not isinstance(excluded, list):
                    excluded = [excluded]
                if DEFINED_SENTINEL in excluded:
                    if experiment_value is not None and experiment_value != "":
                        return False
                    excluded = [e for e in excluded if e != DEFINED_SENTINEL]
                if experiment_value in excluded:
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

        'server' and 'client' are the MoE server+client split. 'agentic_server'
        launches the standalone inference-engine server used by the agentic
        family (engine-keyed, like 'server'); 'agentic_client' runs
        agent_cap.agents against it (engine-agnostic, like 'client', but keeps
        a per-engine variables_defaults entry for structural symmetry since
        the flags list is identical across engines). 'agentic_engine_server' is
        the swe-bench-lite EIDF variant of 'agentic_server': a separate flags/
        command set (not just separate values) because that Job runs on a bare
        @agentic_engine_image@ with the engine pip-installed at generation-validated
        versions, not the AgentCAP image agentic_server_command assumes -- see
        _agentic() and docs/agentic-pipeline-design.md.
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
        elif cmd_type == "agentic_engine_server":
            flags_def = config["variables_defaults"]["agentic_engine_server_flags"]
            cmd_cfg = config["variables_defaults"]["agentic_engine_server_command"][engine]
            rule_key = "agentic_engine_server_flags"
        elif cmd_type == "agentic_client":
            flags_def = config["variables_defaults"]["agentic_client_flags"]
            cmd_cfg = config["variables_defaults"]["agentic_client_command"][engine]
            rule_key = "agentic_client_flags"
        else:
            flags_def = config["variables_defaults"]["client_flags"]
            cmd_cfg = config["variables_defaults"]["client_command"]
            rule_key = "client_flags"

        # Resolve every flag to a single value before rendering any of it, so a
        # rule that overrides a flag already in the base list REPLACES it rather
        # than appending a second copy. Rendering in two passes emitted e.g.
        # `--output-dir $OUTPUT_ROOT ... --output-dir $RUN_DIR`, which argparse
        # happens to resolve last-wins -- but only by luck, and it made the
        # generated command actively misleading to read.
        overrides = {p: v for p, v in self.resolve_params(rule_key, matching_rules).items()
                     if p in flags_def}

        cmd = cmd_cfg["base_command"] + " \\\n"

        # Base parameters, in base order -- skipping any a rule overrides, which
        # are emitted below instead. Skipping (rather than emitting both) is what
        # stops a flag appearing twice, e.g. `--output-dir $OUTPUT_ROOT ...
        # --output-dir $RUN_DIR`; argparse resolved that last-wins, but only by
        # luck, and it made the generated command misleading to read.
        for param in cmd_cfg["flags"]:
            if param not in flags_def or param in overrides: continue
            value = self._resolve_flag_value(param, flags_def[param], parameters)
            rendered = self._build_flag(flags_def[param], param, value)
            if rendered: cmd += f"  {rendered} \\\n"

        # Rule overrides last, preserving the position the duplicate copy used
        # to occupy -- so removing the duplication changes no existing output.
        for param, value in overrides.items():
            rendered = self._build_flag(flags_def[param], param, value)
            if rendered: cmd += f"  {rendered} \\\n"

        return cmd.strip(" \\\n")


    def _moe(self, config, parameters, matching_rules):
        """MoE family: server+client split, engine-keyed image. Returns
        exactly the replacements the pre-refactor single get() produced,
        unchanged, so generated MoE YAMLs stay byte-identical."""
        server_cmd = self.build_command("server", config, parameters, matching_rules)
        client_cmd = self.build_command("client", config, parameters, matching_rules)

        extra_env = self.resolve_generic_variable("extra_container_env", config, matching_rules, parameters)
        arena_dl = self.resolve_generic_variable("download_arena_hard_baseline_answers", config, matching_rules, parameters)
        expert_dist_copy_cmd = self.resolve_generic_variable("expert_distribution_copy_command", config, matching_rules, parameters)

        # Construct Image Name: base + version + variant
        img_cfg = self.resolve_generic_variable("image", config, matching_rules, parameters)
        img_version = self.resolve_generic_variable("inference_engine_version", config, matching_rules, parameters)
        cuda_variant = img_cfg.get("cuda_variant", "") if isinstance(img_cfg, dict) else ""
        image_name = f"{img_cfg['base']}:v{img_version}{cuda_variant if cuda_variant else ''}"

        replacements = {
            "@image_name@": image_name,
            "@extra_container_env@": extra_env,
            "@download_arena_hard_baseline_answers@": arena_dl,
            "@expert_distribution_copy_command@": expert_dist_copy_cmd,
            "@server_start_command@": server_cmd,
            "@client_run_command@": client_cmd,
            "@hf_model_path@": parameters.get("hf_model_path"),
        }
        return "templates/template.yaml", replacements


    def _agentic(self, config, parameters, matching_rules):
        """Agentic family: agent_cap.agents drives the benchmark against a
        separately-started inference-engine server, optionally alongside a
        tool-server sidecar (mcp-atlas) or an external sandbox/exec provider
        (swe-bench-lite). One template (templates/agentic.yaml) for all three
        benchmark tiers; per-benchmark differences are supplied by config.yaml
        rules resolved below, never by forking the template.
        """
        # swe-bench-lite on any k8s cluster (EIDF today; PortForwardK8sProvider
        # or a future InClusterK8sProvider on a cluster that grants pod RBAC --
        # see swe_bench_lite_k8s) uses its own engine image/flags/command set,
        # deliberately NOT agentcap_image / agentic_server_command: those are
        # built for the all-in-one job (imo-answerbench/mcp-atlas), which needs
        # the full AgentCAP client stack baked in. swe-bench-lite's engine
        # build is instead pinned to what was actually validated: the image,
        # engine version and launch flags below are copied from the validated
        # swe-bench-lite EIDF runs recorded in
        # TEAS_Results_Private/agentic/eidf/{sglang,vllm}/gpt-oss-120b/swe-bench-lite/
        # host/port/served_model_name are the exception -- those follow the current
        # PortForwardK8sProvider deployment (loopback + kubectl port-forward,
        # litellm's openai routing convention).
        # Vast.ai never reaches this branch : pipeline/vast/resolve_commands.py
        # calls build_command directly, not via _agentic().
        swe_bench_lite_engine = swe_bench_lite_k8s(parameters)
        if swe_bench_lite_engine:
            server_cmd = self.build_command("agentic_engine_server", config, parameters, matching_rules)
        else:
            server_cmd = self.build_command("agentic_server", config, parameters, matching_rules)
        client_cmd = self.build_command("agentic_client", config, parameters, matching_rules)

        agentcap_repo = self.resolve_generic_variable("agentcap_repo", config, matching_rules, parameters)
        agentcap_ref = self.resolve_generic_variable("agentcap_ref", config, matching_rules, parameters)
        agentic_engine_version = self.resolve_generic_variable("agentic_inference_engine_version", config, matching_rules, parameters)

        if swe_bench_lite_engine:
            image_name = self.resolve_generic_variable("agentic_engine_image", config, matching_rules, parameters)
            env_setup = self.resolve_generic_variable("agentic_engine_install_command", config, matching_rules, parameters)
        else:
            image_name = self.resolve_generic_variable("agentcap_image", config, matching_rules, parameters)
            env_setup_path = self.resolve_generic_variable("agentic_env_setup_script", config, matching_rules, parameters)
            with open(env_setup_path, "r") as f:
                env_setup = f.read().strip()

        # Composable blocks (see docs/agentic-pipeline-design.md): each
        # defaults to "" in variables_defaults, so a benchmark that doesn't
        # need e.g. a sidecar just renders an empty block in the template.
        sidecar_containers = self.resolve_generic_variable("sidecar_containers", config, matching_rules, parameters)
        sidecar_wait = self.resolve_generic_variable("sidecar_wait", config, matching_rules, parameters)
        extra_setup = self.resolve_generic_variable("extra_setup", config, matching_rules, parameters)
        service_account = self.resolve_generic_variable("service_account", config, matching_rules, parameters)
        extra_container_env = self.resolve_generic_variable("extra_container_env", config, matching_rules, parameters)
        teas_env_exports = self.resolve_generic_variable("teas_env_exports", config, matching_rules, parameters)

        # TEAS_GPU_TYPE wants a human-readable GPU name, distinct from the
        # k8s nodeSelector value (@gpu_product@, e.g. 'NVIDIA-A100-SXM4-80GB').
        teas_gpu_name = TEAS_GPU_NAME_MAP.get(parameters.get("gpu"), parameters.get("gpu"))

        # Recorded in provenance.json (not just embedded in the client
        # command string) because sandbox placement is part of the measured
        # scenario -- see docs/agentic-pipeline-design.md's measurement
        # caveat. "" for imo-answerbench/mcp-atlas, which use no sandbox
        # provider at all.
        sandbox_provider = self.resolve_params("agentic_client_flags", matching_rules).get("sandbox_provider", "")

        replacements = {
            "@image_name@": image_name,
            "@agentic_server_command@": server_cmd,
            "@agentic_client_command@": client_cmd,
            "@agentic_env_setup@": env_setup,
            "@agentcap_repo@": agentcap_repo,
            "@agentcap_ref@": agentcap_ref,
            "@agentic_engine_version@": str(agentic_engine_version),
            "@teas_gpu_name@": str(teas_gpu_name),
            "@model@": parameters.get("model"),
            "@hf_model_path@": parameters.get("hf_model_path"),
            "@model_path@": parameters.get("model_path"),
            "@benchmark@": parameters.get("benchmark"),
            "@num_tasks@": str(parameters.get("num_tasks")),
            "@concurrency@": str(parameters.get("concurrency")),
            "@sandbox_provider@": str(sandbox_provider),
            "@sidecar_containers@": sidecar_containers,
            "@sidecar_wait@": sidecar_wait,
            "@extra_setup@": extra_setup,
            "@service_account@": service_account,
            "@extra_container_env@": extra_container_env,
            "@teas_env_exports@": teas_env_exports,
        }
        if needs_login_node_driver(parameters):
            # Two artifacts instead of one: an engine-only Job, plus the driver
            # that submits it, tunnels to it, runs the benchmark and tears it
            # down. get_artifacts() below renders both.
            return "templates/agentic-driver.sh", replacements
        return "templates/agentic.yaml", replacements


    def get_artifacts(self, parameters: dict, results_repo: str):
        """Render one experiment row into the file(s) needed to run it.

        Returns [(filename, content, mode)]. Almost every row is a single Job
        YAML. A row that needs a login-node driver (see
        utils.needs_login_node_driver) is a driver script plus the engine-only
        Job manifest it submits.
        """
        stem = k8s_friendlify(get_run_name(parameters))
        if not needs_login_node_driver(parameters):
            return [(f"{stem}.yaml", self.get(parameters, results_repo), 0o644)]

        engine_name = f"{stem}.engine.yaml"
        driver = self.get(parameters, results_repo,
                          extra={"@engine_manifest@": engine_name})
        engine = self.get(parameters, results_repo,
                          template_override="templates/agentic-engine.yaml")
        return [(f"{stem}.sh", driver, 0o755),
                (engine_name, engine, 0o644)]

    def get(self, parameters: dict, results_repo: str,
            extra: dict = None, template_override: str = None):
        with open("configs/config.yaml", "r") as f:
            config = yaml.safe_load(f)

        rules = config.get("rules", [])
        matching_rules = self.get_matching_rules(rules, parameters)

        teasbench_commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()

        # Substitutions common to every family
        replacements={
            "@name_k8s@": k8s_friendlify(get_run_name(parameters)),
            "@inference_engine@": parameters.get('inference_engine'),
            "@num_gpu@": str(parameters.get("num_gpu")),
            "@gpu_product@": str(parameters.get("gpu_product")),
            "@results_repo@": results_repo,
            "@pvc_archive_dir@": PVC_ARCHIVE_DIR,
            "@output_repo_dir@": results_repo_dir(parameters),
            "@teasbench_commit@": teasbench_commit
        }

        if benchmark_family(parameters) == "agentic":
            template_path, family_replacements = self._agentic(config, parameters, matching_rules)
        else:
            template_path, family_replacements = self._moe(config, parameters, matching_rules)
        replacements.update(family_replacements)

        # The driver runs from a checkout on the login node and imports the
        # provider from it, so it needs the repo root as an absolute path.
        replacements["@teasbench_root@"] = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        if extra:
            replacements.update(extra)
        if template_override:
            template_path = template_override

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
