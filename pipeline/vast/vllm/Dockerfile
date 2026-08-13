# Build with the following, from the repo root directory (note the build context is pipeline/,
# so that template.py, utils.py and configs/config.yaml can be copied into the image):
# docker build --platform=linux/amd64 --build-arg TEASBENCH_COMMIT=$(git rev-parse --short HEAD) -t ghcr.io/teas-project/vllm-bench:latest -f pipeline/vast/vllm/Dockerfile pipeline/
# (or podman)

# Base the image on vLLM v0.16.0, as on K8s.
FROM docker.io/vllm/vllm-openai:v0.16.0

ENV DEBIAN_FRONTEND=noninteractive

# Passing the current commit of TEASBench through is a bit clunky. I've added an extra argument here, which
# should be used at build time (see above).
ARG TEASBENCH_COMMIT=unknown
ENV TEASBENCH_COMMIT=${TEASBENCH_COMMIT}

ENV ALLOWED_ENGINE=vllm

# Install git and jq, then clean up apt lists to keep the image small.
RUN apt-get -o Acquire::Retries=3 -o Acquire::http::Timeout="30" update && \
    apt-get -o Acquire::Retries=3 -o Acquire::http::Timeout="30" install -y --no-install-recommends git curl jq ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Clone MoE-CAP into /opt at build time so it's available in the image. Although we'll be using it in
# /dev/shm the runtime may mount it as empty, so instead we clone it into /opt and the entrypoint script
# will copy it into the location we'll actually use it from.
# For now, clone whichever commit is most recent, but I imagine we may want to pin it in the future.
RUN git clone https://github.com/Auto-CAP/MoE-CAP.git /opt/MoE-CAP

# Set global git config used by automated pushes from the container
RUN git config --global user.name "Pipeline Autopush" && \
    git config --global user.email ""

# Copy the run_benchmarks.sh and entrypoint.sh scripts into the image.
COPY vast/run_benchmarks.sh /root/run_benchmarks.sh
COPY vast/vllm/entrypoint.sh /root/entrypoint.sh
RUN chmod +x /root/run_benchmarks.sh /root/entrypoint.sh

# Also copy in the MoE-CAP command resolution infrastructure.
COPY template.py utils.py /opt/teasbench/pipeline/
COPY configs/config.yaml /opt/teasbench/pipeline/configs/config.yaml
COPY configs/sites/vastai.yaml /opt/teasbench/pipeline/configs/sites/vastai.yaml
COPY vast/resolve_commands.py /opt/teasbench/pipeline/vast/resolve_commands.py

WORKDIR /root

# The entrypoint.sh script ultimately runs run_benchmarks.sh, but it's where we do
# anything specific for vLLM (here), and any potential runtime setup we might need
# in the future. For now, it just copies the MoE-CAP repo into /dev/shm.
ENTRYPOINT ["/root/entrypoint.sh"]
