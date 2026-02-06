#!/bin/python3

from datetime import datetime
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

        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        model_name_clean=model_name.split("/")[1].replace(".", "-")
        gpu=gpu_product.split("-")[1] 

        run_name = get_run_name(model_name, gpu_product, num_gpu, target_input_tokens, target_output_tokens, batch_size, dataset)
        run_name_lower = run_name.replace("_", "-").lower()
        
        return f"""
apiVersion: batch/v1
kind: Job
metadata:
  generateName: sglang-moe-cap-
  #generateName: sglang-moe-cap-{run_name.replace("_", "-").lower()}-
  labels:
    kueue.x-k8s.io/queue-name:  eidf230ns-user-queue
spec:
  completions: 1
  backoffLimit: 0
  ttlSecondsAfterFinished: 1800
  template:
    metadata:
      name: job-sglang-moe-cap
      #name: job-sglang-moe-cap-{run_name.replace("_", "-").lower()}
    spec:
      containers:
      - name: sglang-server
        image: lmsysorg/sglang:latest
        imagePullPolicy: IfNotPresent
        env:
          - name: SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR
            value: "/dev/shm/sglang_expert_distribution_recorder"
        command: ["/bin/bash", "-c"]
        args:
          - |
            apt-get update
            apt-get -y install git
            git clone https://github.com/markxio/MoE-CAP.git /dev/shm/MoE-CAP
            cd /dev/shm/MoE-CAP
            pip install -e .
            pip install gputil

            # Start server
            python -m moe_cap.systems.sglang \\
              --model-path {model_name} \\
              --port 30000 \\
              --expert-distribution-recorder-mode stat \\
              --tp-size {tensor_parallel_size} \\
              &> /dev/shm/{run_name}_{timestamp}.server_log &
            SERVER_PID=$!

            # Wait until the /health endpoint returns HTTP 200
            echo "Waiting for SGLang server to be ready..."

            until curl -s -f http://localhost:30000/health > /dev/null; do
              echo -n "."
              sleep 2
            done

            echo "SGLang server is ready!"
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
              --backend sglang \\
              --ignore-eos \\
              --server-batch-size {batch_size} \\
              --output_dir /dev/shm/{run_name} \\
              &> /dev/shm/{run_name}_{timestamp}.client_log

            echo "Starting to serve bench (sending http requests)... done!"
            echo "Benchmark finished, shutting down server..."

            kill $SERVER_PID
            wait $SERVER_PID
            
            echo "Server stopped. Copying files to pvc..."
            
            mkdir -p /mnt/ceph/tmp/MoE-CAP-outputs
            cp -R /dev/shm/{run_name} /mnt/ceph/tmp/MoE-CAP-outputs/num_samples_256/
            cp /dev/shm/{run_name}_{timestamp}* /mnt/ceph/tmp/MoE-CAP-outputs/num_samples_256/
            
            echo "Files copied, exiting container"
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
          - mountPath: /mnt/ceph
            name: volume
          - mountPath: /dev/shm
            name: dshm
      restartPolicy: Never
      volumes:
        - name: volume
          persistentVolumeClaim:
            claimName: client-ceph-pvc
        - name: dshm
          emptyDir:
            medium: Memory
            sizeLimit: 16Gi
      nodeSelector:
        nvidia.com/gpu.product: {gpu_product}
               """