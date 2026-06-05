# Pipeline execution on Vast.ai

This directory contains the resources needed to run the pipeline through Vast.ai.

We will provide containers derived from the same vLLM and SGLang images used on the EIDF, with a runscript and MoE-CAP baked in. Environment variables passed through the Vast.ai interface parameterise the benchmarks to be run and provide necessary secrets. The pipeline script will then run through the set of benchmarks, run them one-by-one and push the results to GitHub.

An in-development version of the runscript can be found in the `vllm` directory.