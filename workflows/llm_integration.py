import sys
import os
import subprocess

# Ensure we can import from the same directory
script_dir = os.path.dirname(os.path.abspath(__file__))
target_script = os.path.join(script_dir, "llm_hybrid_agent.py")

if __name__ == "__main__":
    if not os.path.exists(target_script):
        print(f"❌ Error: Target script '{target_script}' not found.")
        sys.exit(1)
    
    print(f"🚀 Starting LLM Integration (via {os.path.basename(target_script)})...")
    
    try:
        # Run the hybrid agent script
        # We use sys.executable to ensure we use the same python interpreter
        subprocess.run([sys.executable, target_script], check=True)
    except KeyboardInterrupt:
        print("\n👋 LLM Integration closed.")
    except Exception as e:
        print(f"❌ Error during LLM Integration: {e}")
        sys.exit(1)
