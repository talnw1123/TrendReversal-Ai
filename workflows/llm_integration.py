import sys
import os

# Ensure workflows/ directory is on sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

if __name__ == "__main__":
    # Run llm_hybrid_agent directly in-process (no subprocess) to avoid numpy path issues
    exec(open(os.path.join(script_dir, "llm_hybrid_agent.py")).read())
