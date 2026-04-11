import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI Client
# Note: If you encounter proxy errors again, use: 
# import httpx
# client = OpenAI(api_key=..., http_client=httpx.Client(proxy=None))
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY")
)

def _get_file_summary(file_diff: str) -> str:
    """
    (The "Map" Step) Generates a high-density summary for a single file's diff.
    """
    system_prompt = "You are a code analysis bot. Summarize the changes in this diff file in a Code specialist, specific to a major code change, technical, bullet-point format."
    
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": file_diff},
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error summarizing file: {e}"

def get_strategic_summary(file_summaries: list[str], repo_tree: str) -> str:
    """
    (The "Reduce" Step) Takes file summaries AND repo structure to synthesize a high-level overview.
    """
    # We inject the repo_tree into the system prompt for context
    system_prompt = f"""
    You are a principal engineer reviewing a pull request. 
    You have received summaries of changes from your junior engineers for each file. 
    
    Your task is to synthesize these summaries into a single, high-level strategic overview.
    
    --- REPOSITORY STRUCTURE CONTEXT ---
    {repo_tree}
    ------------------------------------

    Write in single line points, Make it readable, keep it short. Write File names and key notes. 
    Focus on the overall goal, the architectural impact, and any potential risks (especially if changes touch sensitive files in the repo structure).
    Structure your output with the 'Three-Pillar Analysis': ## Summary, ## Rationale, and ## Consequence.
    Dont be overly friendly or humble or supportive, be critical, be concise, dont write more than what is needed.
    Write in a way which forces reader to ponder on your words.
    """
    
    combined_summaries = "\n".join(file_summaries)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Here are the file summaries:\n{combined_summaries}"},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating strategic summary: {e}"