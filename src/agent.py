import os
import dotenv
from groq import Groq

# Import our deterministic token optimizer
from src.optimizer import optimize_payload_for_ai

# Load environment variables
dotenv.load_dotenv()

# Initialize Groq Client
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)

def _get_file_summary(filepath: str, full_file_content: str, file_diff: str) -> str:
    """
    (Map Step) Blunt, technical audit of a single file.
    """
    optimized_payload = optimize_payload_for_ai(filepath, full_file_content, file_diff)
    
    system_prompt = """
    Persona: Principal Engineer. Blunt, authoritative, zero fluff. 
    Task: Audit the AST-mapped diff. 
    Rules:
    1. Identify only the most critical logic or structural issue.
    2. Ignore trivialities (style, minor logs).
    3. Use format:
    
    ### [Function/Module]
    - **Issue:** [Short, blunt description]
    - **Fix:** `[One-line code example]`
    - **Risk:** [Low/Med/High]
    """

    SYSTEM_PROMPT_LEN = len(system_prompt)
    TOKEN_LIMIT = 12000
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": optimized_payload[:TOKEN_LIMIT - SYSTEM_PROMPT_LEN]},
            ],
            temperature=0.0,
            max_tokens=512
        )
        
        summary = completion.choices[0].message.content
        return f"**File: {filepath}**\n{summary}"
        
    except Exception as e:
        return f"Error: {str(e)}"

def get_strategic_summary(file_summaries: list[str], repo_tree: str) -> str:
    """
    (Reduce Step) High-level architectural directives. 
    Focuses strictly on improvements and required actions.
    """
    system_prompt = f"""
    Persona: Principal Engineer. Guardian of the codebase. 
    Task: Review file audits and provide collective guidance.
    
    Rules:
    1. No summaries. No "I've reviewed". No respectful filler.
    2. Focus only on what MUST be improved.
    3. Be extremely brief.

    --- REPO TREE ---
    {repo_tree}

    """
    
    combined_summaries = "\n\n".join(file_summaries)
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Audits:\n{combined_summaries}"},
            ],
            temperature=0.0,
            max_tokens=1024
        )
        
        synthesis = completion.choices[0].message.content
        
        # Combined output for PR: Guidance first, then raw file audits
        return f"""
# 🐰 Principal Review

{synthesis}

---
### 🔍 Technical Findings
{combined_summaries}
        """.strip()
        
    except Exception as e:
        return f"Error in synthesis: {str(e)}"