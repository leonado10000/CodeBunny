import os
import sys
from dotenv import load_dotenv

from pathlib import Path
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from src.optimizer import optimize_payload_for_ai
from src.agent import _get_file_summary, get_strategic_summary

# Load API Keys
load_dotenv()

def run_local_simulation():
    print("🚀 --- CodeBunny Local Logic Simulation --- 🚀\n")

    # 1. API KEY CHECK
    if not os.getenv("GROQ_API_KEY"):
        print("❌ ERROR: GROQ_API_KEY not found in .env file.")
        return

    # 2. MOCK INPUTS (Simulating what main.py fetches from GitHub)
    filepath = "sample.py"
    
    # Realistic source code (The "New" version of the file)
    mock_full_content = """
import os

def calculate_discount(price, discount):
    if discount > 1:
        raise ValueError("Discount too high")
    return price * (1 - discount)

async def get_user_profile(user_id):
    \"\"\"Fetches user data from the DB.\"\"\"
    # Simulating a logic change here
    print(f"Fetching user {user_id}...")
    user = {"id": user_id, "active": True}
    return user
"""

    # Realistic Diff (Simulating the PR changes)
    mock_diff = """--- a/sample.py
+++ b/sample.py
@@ -3,2 +3,4 @@
 def calculate_discount(price, discount):
+    if discount > 1:
+        raise ValueError("Discount too high")
     return price * (1 - discount)
@@ -8,2 +10,3 @@
     \"\"\"Fetches user data from the DB.\"\"\"
+    print(f"Fetching user {user_id}...")
     user = {"id": user_id, "active": True}
"""

    mock_repo_tree = "src/main.py\nsrc/agent_groq.py\nsample.py\nrequirements.txt"

    # ==========================================
    # STEP 1: TEST THE OPTIMIZER (The Token Squeezer)
    # ==========================================
    print("🛠️  [STEP 1] Testing Optimizer Inputs/Outputs...")
    print(f"INPUT: filepath='{filepath}', content_len={len(mock_full_content)}, diff_len={len(mock_diff)}")
    
    optimized_payload = optimize_payload_for_ai(filepath, mock_full_content, mock_diff)
    
    print("\nOUTPUT (Optimized Payload for AI):")
    print("-" * 30)
    print(optimized_payload)
    print("-" * 30 + "\n")

    # ==========================================
    # STEP 2: TEST THE MAP AGENT (File-Level Review)
    # ==========================================
    print("🤖 [STEP 2] Testing Map Agent (Groq AI)...")
    
    # This simulates the loop in main.py
    file_summary = _get_file_summary(filepath, mock_full_content, mock_diff)
    
    print("\nOUTPUT (Groq File Summary):")
    print("-" * 30)
    print(file_summary)
    print("-" * 30 + "\n")

    # ==========================================
    # STEP 3: TEST THE REDUCE AGENT (Strategic Summary)
    # ==========================================
    print("🧠 [STEP 3] Testing Reduce Agent (Strategic Synthesis)...")
    
    file_summaries = [file_summary] # Simulate a list of summaries
    final_report = get_strategic_summary(file_summaries, mock_repo_tree)
    
    print("\nOUTPUT (Final Strategic Report):")
    print("-" * 60)
    print(final_report)
    print("-" * 60 + "\n")

if __name__ == "__main__":
    run_local_simulation()