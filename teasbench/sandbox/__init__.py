"""Sandbox / exec-container providers for SWE-bench-style benchmarks.

A "sandbox" is a per-task swe-rex server (SWE-agent's `remote` deployment
type) that the agent talks to over HTTP. An "exec container" is a plain
pod from an official instance image supporting file upload + command
execution, used by the harness evaluator to grade a submitted patch.

See teasbench.sandbox.k8s for the EIDF k8s implementations, selected by
AgentCAP via a dotted path (e.g. `--sandbox-provider
teasbench.sandbox.k8s:InClusterK8sProvider`). AgentCAP duck-types these
classes; nothing here imports agent_cap.
"""

from teasbench.sandbox.k8s import InClusterK8sProvider, PortForwardK8sProvider

__all__ = ["InClusterK8sProvider", "PortForwardK8sProvider"]
