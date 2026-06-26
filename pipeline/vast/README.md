# Pipeline execution on Vast.ai

## Contents

This directory contains the resources needed to run the pipeline through Vast.ai.

We will provide containers derived from the same vLLM and SGLang images used on the EIDF, with a runscript and MoE-CAP baked in. Environment variables passed through the Vast.ai interface parameterise the benchmarks to be run and provide necessary secrets. The pipeline script will then run through the set of benchmarks, run them one-by-one and push the results to GitHub.

A script [run_vllm_gpt_oss_smoke.sh](vllm/run_vllm_gpt_oss_smoke.sh) can be found in the vllm directory. This runs through three datasets for the gpt-oss-20b benchmark. There is also [encode-csv-keyed-errors.sh](vllm/encode-csv-keyed-errors.sh); this is a test script which shows the contents of a CSV file being turned into a variable which can be passed through Vast.ai's interface into the container, then used to loop through the benchmark parameters on each line, error checking along the way. These two scripts now need to be combined.

Alternatively, allow an inclusive CSV within the container but only run those benchmarks which match current hardware and inference engine. It's a question of whether to do this selection outside or inside the container. I'm of the opinion at the moment that it would be easier to do outside in the Python layer.

## Planned workflow

1. Run the pipeline `generate.py` script locally, likely with a new `--vast` option to tell it to generate commands for Vast.ai and not the EIDF.
2. This generates a bash script or scripts containing the required commands to reserve the requested resource on Vast.ai. Likely multiple scripts, separated by hardware and inference engine (since we need to ask Vast.ai to reserve given instances running given images). They will include the tokenised CSV file and the ability to retrieve and pass through as environment variables any secrets.
3. Run the bash script to submit a 'job' to Vast.ai. Needs to be some decision-making about how to determine whether to reserve a given hardware option (given price in particular).
4. The container will go through the entrypoint script, running all the benchmarks for this GPU/inference engine combination, pushing to GitHub as it goes.

## Further work

Currently, the entrypoint scripts living inside the container are decoupled from the EIDF pipeline, in particular the config.yaml options. This means any changes made to one pipeline (EIDF/VAST) need to be manually ported to the other. It's also rather ad hoc at the moment; doing something more programmatic would be much preferred.

This isn't ideal. Longer term, it may be a good idea to turn container entrypoint scripts into templates to be modified much in the same way as the EIDF YAMLs with the correct option sets for different benchmarks.