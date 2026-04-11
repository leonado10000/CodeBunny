import sys
import os

from pathlib import Path
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from src.optimizer import optimize_payload_for_ai


def run_optimizer_grader():
    print("🚀 Starting External Token Optimizer Test & Grader...\n")

    # ==========================================
    # 1. SETUP: LOAD EXTERNAL FILES
    # ==========================================
    sample_py_path = "sample.py"
    sample_diff_path = "sample2.diff"

    print(f"📂 Loading target file: {sample_py_path}")
    with open(sample_py_path, 'r', encoding='utf-8') as f:
        mock_full_code = f.read()

    print(f"📂 Loading diff file: {sample_diff_path}")
    with open(sample_diff_path, 'r', encoding='utf-8') as f:
        mock_diff = f.read()

    # ==========================================
    # 2. RUN OPTIMIZER
    # ==========================================
    print("⚙️  Running optimize_payload_for_ai() with realistic files...")
    result = optimize_payload_for_ai(sample_py_path, mock_full_code, mock_diff)
    
    print("\n" + "="*60)
    print("OUTPUT FROM OPTIMIZER (What the AI will see):")
    print("="*60)
    print(result)
    print("="*60 + "\n")

    # ==========================================
    # 3. GRADING / EVALUATION
    # ==========================================
    print("📊 GRADING RESULTS:\n")

    # These evaluate if the AST successfully translated the new realistic files
    expected_findings = [
        {"desc": "Finds `calculate_discount`", "str": "Function: `def calculate_discount`"},
        {"desc": "Flags Missing Docstring", "str": "⚠️ FLAG: MISSING_DOCSTRING"},
        {"desc": "Identifies Minor Fix", "str": "Change Type: Minor Fix"},
        {"desc": "Finds `get_user_profile` (Async)", "str": "Function: `async def get_user_profile`"},
        {"desc": "Extracts Decorators", "str": "Decorators: @rate_limit(...)"},
        {"desc": "Extracts existing Docstring", "str": "Docstring: \"\"\"Fetches a user profile"},
        {"desc": "Identifies Major Refactor", "str": "Change Type: Major Refactor/Addition"},
        {"desc": "Finds `generate_monthly_report`", "str": "Function: `def generate_monthly_report`"},
        {"desc": "Flags Large Function (> 100 lines)", "str": "⚠️ FLAG: LARGE_FUNCTION (>100 lines)"},
        {"desc": "Simulates Usage Scan (Cross-Codebase)", "str": "Usage Scan: `calculate_discount`"},
        {"desc": "Cleans the Diff (Truncation/Stripping)", "str": "--- CODE CHANGES (Small Fix Context) ---"}
    ]

    score = 0
    total = len(expected_findings)

    for item in expected_findings:
        if item["str"] in result:
            print(f"✅ PASS: {item['desc']}")
            score += 1
        else:
            print(f"❌ FAIL: {item['desc']}")
            print(f"   -> Expected to find substring: '{item['str']}'")

    # ==========================================
    # 4. FINAL SCORE
    # ==========================================
    percentage = (score / total) * 100
    print("\n" + "="*30)
    print(f"🏆 FINAL SCORE: {score} / {total} ({percentage:.1f}%)")
    
    if score == total:
        print("🌟 PERFECT! The optimizer successfully extracted metadata from the realistic files!")
    else:
        print("⚠️ SOME FLAGS MISSED. Check the AST parser logic or line tracking.")
    print("="*30 + "\n")

if __name__ == "__main__":
    run_optimizer_grader()