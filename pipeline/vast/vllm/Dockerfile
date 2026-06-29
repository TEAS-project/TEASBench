# Base the image on vLLM v0.16.0, as on the EIDF.
FROM --platform=linux/amd64 vllm/vllm-openai:v0.16.0

ENV DEBIAN_FRONTEND=noninteractive

# Install git and jq, then clean up apt lists to keep the image small.
RUN apt-get -o Acquire::Retries=3 -o Acquire::http::Timeout="30" update \
 && apt-get -o Acquire::Retries=3 -o Acquire::http::Timeout="30" install -y --no-install-recommends \
	git jq ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Clone MoE-CAP into /opt at build time so it's available in the image. Although we'll be using it in
# /dev/shm the runtime may mount it as empty, so instead we clone it into /opt and the entrypoint script
# will copy it into the location we'll actually use it from.
# For now, clone whichever commit is most recent, but I imagine we may want to pin it in the future.
RUN git clone https://github.com/Auto-CAP/MoE-CAP.git /opt/MoE-CAP

# Set global git config used by automated pushes from the container
RUN git config --global user.name "Pipeline Autopush" \
 && git config --global user.email ""

# Copy the run_benchmarks.sh and entrypoint.sh scripts into the image.
COPY ../run_benchmarks.sh /root/run_benchmarks.sh
COPY entrypoint.sh /root/entrypoint.sh
RUN chmod +x /root/run_benchmarks.sh /root/entrypoint.sh

WORKDIR /root

# The entrypoint.sh script ultimately runs run_benchmarks.sh, but it's where we do
# anything specific for vLLM (here), and any potential runtime setup we might need
# in the future. For now, it just copies the MoE-CAP repo into /dev/shm.
ENTRYPOINT ["/root/entrypoint.sh"]
