"""K8s pod providers: the deployment side of the AgentCAP<->TEAS interface.

AgentCAP owns benchmark semantics: agent strategy, call limits, streaming
metrics, official grading, dataset loading, TEAS output schema.

TEASBench owns deployment scenario: which platform, which hardware, which
engine, how containers are provisioned, where results go.

The two only ever talk to each other through endpoints. This package
supplies the pods behind those endpoints for SWE-bench-style benchmarks: a
"sandbox" is a per-task swe-rex server (SWE-agent's `remote` deployment
type) that the agent talks to over HTTP, and an "exec container" is a plain
pod from an official instance image supporting file upload + command
execution, used by the harness evaluator to grade a submitted patch.

AgentCAP loads the classes below by dotted path, e.g.
``--sandbox-provider k8s_pod_providers:InClusterK8sProvider``. This package
has no import-time dependency on ``agent_cap`` and never will, since
AgentCAP duck-types the provider interface rather than importing a shared
base class.

Nothing here is specific to any one cluster: namespace, queue and pod
resources all come from ``TEASBENCH_K8S_*`` environment variables (see
``providers``), which the pipeline sets from the site profile in
``pipeline/configs/sites/``.
"""

from k8s_pod_providers.providers import (
    InClusterK8sProvider,
    PortForwardK8sProvider,
)

__all__ = ["InClusterK8sProvider", "PortForwardK8sProvider"]
