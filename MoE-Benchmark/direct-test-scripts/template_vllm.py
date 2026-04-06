#!/bin/python3

import os
from utils import get_run_name

class Template:
    def __init__(self):
        return

    def get(self, \
        model_name: str, \
        tensor_parallel_size: int, \
        dataset: str, \
        target_input_tokens: int, \
        target_output_tokens: int, \
        num_samples: int, \
        batch_size: int, \
        num_gpu: int, \
        gpu_product: str):

        #timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        model_name_clean=model_name.split("/")[1].replace(".", "-")
        gpu=gpu_product.split("-")[1] 

        run_name = get_run_name("vllm", model_name, gpu_product, num_gpu, target_input_tokens, target_output_tokens, batch_size, dataset)
        run_name_lower = run_name.replace("_", "-").lower()
       
        with open(os.path.expanduser("yaml_templates/template_vllm.yaml"), "r") as f:
          template = f.read()

        template = template.replace("@model_name@", str(model_name))
        template = template.replace("@tensor_parallel_size@", str(tensor_parallel_size))
        template = template.replace("@dataset@", str(dataset))
        template = template.replace("@num_gpu@", str(num_gpu))
        template = template.replace("@target_input_tokens@", str(target_input_tokens))
        template = template.replace("@target_output_tokens@", str(target_output_tokens))
        template = template.replace("@num_samples@", str(num_samples))
        template = template.replace("@gpu_product@", str(gpu_product))
        template = template.replace("@batch_size@", str(batch_size))
        template = template.replace("@run_name@", str(run_name))

        return template

"""
apiVersion: batch/v1
kind: Job
metadata:
  generateName: vllm-moe-cap-
  #generateName: vllm-moe-cap-{run_name.replace("_", "-").lower()}-
  labels:
    kueue.x-k8s.io/queue-name:  eidf230ns-user-queue
spec:
  completions: 1
  backoffLimit: 0
  ttlSecondsAfterFinished: 1800
  template:
    metadata:
      name: job-vllm-moe-cap
      #name: job-vllm-moe-cap-{run_name.replace("_", "-").lower()}
    spec:
      containers:
      - name: sglang-server
        image: vllm/vllm-openai:v0.18.0-cu130
        imagePullPolicy: IfNotPresent
        env:
          - name: HF_HOME
            value: /mnt/input/hf_cache
          - name: HUGGINGFACE_HUB_CACHE
            value: /mnt/input/hf_cache/hub
          - name: TRANSFORMERS_CACHE
            value: /mnt/input/hf_cache/transformers
          - name: HF_DATASETS_CACHE
            value: /mnt/input/hf_cache/datasets
          - name: TEAS_OUTPUT_DIR
            value: /mnt/develop/outputs
          - name: GIT_TOKEN
            valueFrom:
              secretKeyRef:
                name: teas-develop-results-private-jr
                key: git_teas_develop_results_private
        command: ["/bin/bash", "-c"]
        args:
          - |
            apt-get update
            apt-get -y install git
            git clone https://github.com/Auto-CAP/MoE-CAP.git /dev/shm/MoE-CAP
            cd /dev/shm/MoE-CAP
            pip install -e .
            pip install gputil

            timestamp=$( date +%Y%m%d-%H%M )

            # Start server
            python -m moe_cap.systems.vllm \\
              --model-path {model_name} \\
              --port 30000 \\
              --host 0.0.0.0 \\
              --tensor-parallel-size {tensor_parallel_size} \\
              --reasoning-parser deepseek_r1 \\ ### ??? model specific: yes
              --enable-expert-distribution-metrics \\
              --max-num-batched-tokens 131072 \\ ### unstable suggestion by vllm developers leave for now
              &> /dev/shm/{run_name}_$timestamp.server_log &
            SERVER_PID=$!

            # Wait until the /health endpoint returns HTTP 200
            echo "Waiting for vllm server to be ready..."

            until curl -s -f http://localhost:30000/health > /dev/null; do
              echo -n "."
              sleep 2
            done

            echo "vllm server is ready!"
            echo "Starting to serve bench (sending http requests)..."
            
            mkdir -p /dev/shm/{run_name}
            python -m moe_cap.runner.openai_api_profile \\
              --model_name {model_name} \\
              --datasets {dataset} \\
              --input-tokens {target_input_tokens} \\
              --output-tokens {target_output_tokens} \\
              --num-samples {num_samples} \\
              --config-file configs/stub.yaml \\
              --api-url http://localhost:30000/v1/completions \\
              --backend vllm \\
              --ignore-eos \\
              --server-batch-size {batch_size} \\
              --output_dir /dev/shm/{run_name} \\
              &> /dev/shm/{run_name}_$timestamp.client_log

            echo "Starting to serve bench (sending http requests)... done!"
            echo "Benchmark finished, shutting down server..."

            kill $SERVER_PID
            wait $SERVER_PID
            
            echo "Server stopped. Copying files to pvc..."
            
            RUN_OUTPUT_DIR=$TEAS_OUPUT_DIR/vLLM/

            mkdir -p $RUN_OUTPUT_DIR

            cp -R /dev/shm/{run_name} $RUN_OUPUT_DIR/
            cp /dev/shm/{run_name}_$timestamp* $RUN_OUPUT_DIR/
            
            echo "Files copied to pvc at $RUN_OUTPUT_DIR"

            # Commit data to github

            echo "update to github here"


            # End of benchmark message

            echo "Finalising container"

        ports:
          - containerPort: 30000 
        resources:
          requests:
            cpu: 10
            memory: '100Gi'
          limits:
            cpu: 10
            memory: '100Gi'
            nvidia.com/gpu: {num_gpu}
        volumeMounts:
          - mountPath: /mnt/develop
            name: develop
          - mountPath: /mnt/input
            name: inputs
          - mountPath: /dev/shm
            name: dshm
      restartPolicy: Never
      volumes:
        - name: inputs
          persistentVolumeClaim:
            claimName: inputs-pvc
        - name: develop
          persistentVolumeClaim:
            claimName: develop-pvc
        - name: dshm
          emptyDir:
            medium: Memory
            sizeLimit: 16Gi
      nodeSelector:
        nvidia.com/gpu.product: {gpu_product}
"""
