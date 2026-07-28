"""TEASBench: the deployment side of the AgentCAP<->TEAS interface.

AgentCAP owns benchmark semantics: agent strategy, call limits, streaming
metrics, official grading, dataset loading, TEAS output schema.

TEASBench owns deployment scenario: which platform, which hardware, which
engine, how containers are provisioned, where results go.

The two only ever talk to each other through endpoints. See
``teasbench.sandbox`` for the sandbox/exec-container providers AgentCAP
loads by dotted path (``teasbench.sandbox.k8s:InClusterK8sProvider`` etc.)
- this package has no import-time dependency on ``agent_cap`` and never
will, since AgentCAP duck-types the provider interface rather than
importing a shared base class.
"""
