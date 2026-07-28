#!/bin/bash

# This runs outside the container.

# Encode CSV contents with base64.
csv_file="../../../experiments/smoke_tests_vllm.csv"
CONTAINER_CSV=$(base64 -w 0 $csv_file)
echo "Contents of $csv_file:"
echo "$CONTAINER_CSV"

# Now heading into container.

# Configuration
REQUIRED_HEADERS=("inference_engine" "model" "dataset" "num_samples" "gpu" "num_gpu" "batch_size")
ALLOWED_ENGINE="vllm" # Change to SGLang for that container, or whatever else we may have in the future.

# Decode the CSV and grab the header line
IFS=',' read -r -a HEADERS < <(echo "$CONTAINER_CSV" | base64 -d | head -n 1)

# Verify that we have all the column headers we're expecting.
for req in "${REQUIRED_HEADERS[@]}"; do
  if [[ ! " ${HEADERS[@]} " =~ " ${req} " ]]; then
    echo "ERROR: Missing required column header: '$req'" >&2
    exit 1
  fi
done

declare -A row
line_number=1

# Stream the raws from the encoded CSV, skipping the first line (headers).
while IFS=',' read -r -a VALUES; do
  ((line_number++))

  # Verify that we have the same number of entries as headers.
  if [ "${#VALUES[@]}" -ne "${#HEADERS[@]}" ]; then
    echo "ERROR on line $line_number: Column count mismatch" >&2
    exit 1
  fi

  # Clear array from previous line
  row=()

  # Map raw values to headers.
  for i in "${!VALUES[@]}"; do
    row["${HEADERS[$i]}"]="${VALUES[$i]}"
  done

  # Clean up any potential trailing carriage returns or spaces in the value
  current_engine=$(echo "${row[inference_engine]}" | xargs | tr -d '\r')
  # then check it's the engine we've said is allowed.
  if [[ "$current_engine" != "$ALLOWED_ENGINE" ]]; then
    echo "ERROR on line $line_number: Invalid inference_engine '$current_engine'." >&2
    echo "   This container only accepts benchmarks for '$ALLOWED_ENGINE'." >&2
    exit 1
  fi

  # Also raise error if any value in the row is a blank.
  for req in "${REQUIRED_HEADERS[@]}"; do
    if [[ -z "${row[$req]}" ]]; then
      echo "ERROR on line $line_number: Required field '$req' is empty." >&2
      exit 1
    fi
  done

  # Everything looks good, so start the server and run the benchmark.
  echo "Running ${row[inference_engine]} ${row[model]} ${row[dataset]} ${row[num_samples]} ${row[gpu]} ${row[num_gpu]} ${row[batch_size]}"

done < <(echo "$CONTAINER_CSV" | base64 -d | tail -n +2)

echo "Done!"
