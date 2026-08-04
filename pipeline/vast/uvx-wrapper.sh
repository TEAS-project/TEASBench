#!/bin/bash
export UV_CONSTRAINT=/opt/teasbench/pipeline/vast/mcp-atlas-constraints.txt
exec /root/.local/bin/uvx "$@"
