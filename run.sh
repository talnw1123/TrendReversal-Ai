#!/bin/bash

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# ใช้ uv run เพื่อหลีกเลี่ยงปัญหา Conda แทรกแซง numpy
uv run workflows/llm_hybrid_agent.py
