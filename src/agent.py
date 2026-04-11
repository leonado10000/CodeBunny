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
    (The Map Step)
    Pre-processes the file using AST-based mapping to strip noise, 
    then uses Llama-3.3 to summarize the engineering impact.
    """
    # 1. Deterministic Token Squeezing
    optimized_payload = optimize_payload_for_ai(filepath, full_file_content, file_diff)
    
    system_prompt = """
    Persona: You are a Principal Software Engineer. You are authoritative, concise, and corrective.
    
    Task: Review the deterministic AST-mapped diff. 
    Rules:
    1. Focus on "The Main Thing" (impactful architectural/logic concerns).
    2. Prioritize 'Pylint Flags' if present.
    3. Provide a 1-line code example for corrections.
    4. Ignore minor low-stack issues.
    
    Format:
    ### [Function/Module Name]
    - **Issue:** [The critical logic or pattern concern]
    - **Correction:** [Example code or specific directive]
    - **Impact:** [High/Med/Low] - [Risk description]
    """
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": optimized_payload},
            ],
            temperature=0.0,
            max_tokens=1024
        )
        
        summary = completion.choices[0].message.content
        return f"#### 📄 File: {filepath}\n{summary}"
        
    except Exception as e:
        return f"Error analyzing {filepath}: {str(e)}"

def get_strategic_summary(file_summaries: list[str], repo_tree: str) -> str:
    """
    (The Reduce Step)
    Synthesizes the collection of file-level summaries into a Principal-level report.
    This version includes both the raw problems and the high-level guidance.
    """
    system_prompt = f"""
    Persona: Principal Engineer. You are a mentor and a guardian of the codebase. 
    You are conscious of your words, collective in your outlook, and focused on growth.
    
    Task: Synthesize file reviews into high-level engineering guidance. 
    Do NOT summarize. Instead, provide directives and mentorship.

    --- REPOSITORY CONTEXT ---
    {repo_tree}
    --------------------------

    Instructions:
    - Use the following sections: '## 🎯 Engineering Directives', '## 💡 Mentorship & Guidance', and '## 🏗️ Architectural Outlook'.
    - Engineering Directives: Absolute "must-fixes" for this PR to meet quality standards.
    - Mentorship & Guidance: Explain "why" certain patterns are better, helping the dev level up.
    - Architectural Outlook: How this change shifts the structural integrity of the project.
    - Be authoritative but constructive. Focus only on what needs improvement.
    """
    
    combined_summaries = "\n\n".join(file_summaries)
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Summaries to synthesize:\n{combined_summaries}"},
            ],
            temperature=0.1,
            max_tokens=2048
        )
        
        synthesis = completion.choices[0].message.content
        
        # We return both the detailed problems and the strategic guidance for the PR
        final_report = f"""
# 🐰 CodeBunny Principal Review

{synthesis}

---

## 🔍 Detailed File-Level Audits
Below are the specific technical findings for each modified component.

{combined_summaries}
        """
        return final_report.strip()
        
    except Exception as e:
        return f"Error in strategic synthesis: {str(e)}"