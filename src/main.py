import os
import json
import requests
import uvicorn
import re
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

# Use absolute imports for agent logic
from src.auth import get_installation_access_token
from src.agent_groq import _get_file_summary, get_strategic_summary

app = FastAPI()

# --- RATE LIMITING (In-Memory) ---
USAGE_COUNTS = {}
USAGE_LIMIT = 10 

# --- Helper Functions ---

def get_pr_diff(token: str, diff_url: str) -> str:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3.diff"
    }
    try:
        response = requests.get(diff_url, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"FATAL: Failed to fetch PR diff: {e}")
        raise

def get_file_content(token: str, repo_full_name: str, filepath: str, sha: str) -> str:
    """Fetches raw content of a file for AST analysis."""
    url = f"https://api.github.com/repos/{repo_full_name}/contents/{filepath}?ref={sha}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3.raw"
    }
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 404:
            return "" # Likely a new file
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Warning: Could not fetch content for {filepath}: {e}")
        return ""

def get_repo_tree(token: str, repo_full_name: str, main_branch: str) -> str:
    url = f"https://api.github.com/repos/{repo_full_name}/git/trees/{main_branch}?recursive=1"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        tree_data = response.json()
        paths = [item['path'] for item in tree_data['tree'] if item['type'] == 'blob']
        return "\n".join(paths[:1500]) # Token-safe limit
    except Exception as e:
        print(f"Warning: Could not fetch repo tree: {e}")
        return "Repository file tree unavailable."

def post_pr_comment(token: str, repo_full_name: str, pr_number: int, body: str):
    url = f"https://api.github.com/repos/{repo_full_name}/issues/{pr_number}/comments"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    try:
        requests.post(url, headers=headers, json={"body": body}).raise_for_status()
        print(f"✅ Posted comment to {repo_full_name} PR #{pr_number}.")
    except Exception as e:
        print(f"❌ Failed to post comment: {e}")

def parse_diff_into_files(full_diff: str) -> list[str]:
    file_diffs = re.split(r'(?=diff --git a/)', full_diff)
    return [diff.strip() for diff in file_diffs if diff.strip()]

def extract_filepath(file_diff: str) -> str:
    match = re.search(r'b/(\S+)', file_diff)
    return match.group(1) if match else "unknown"

def run_analysis_task(payload: dict):
    """
    Background task to handle long-running AI analysis.
    This allows the webhook to return immediately.
    """
    try:
        # 1. Parse Metadata
        if isinstance(payload.get('repository'), dict):
            repo_name = payload['repository']['full_name']
            pr_data = payload.get('pull_request')
        else:
            repo_name = payload['repository']
            pr_data = payload.get('event', {}).get('pull_request')
            
        head_sha = pr_data['head']['sha']
        diff_url = pr_data['diff_url']
        pr_number = pr_data['number']
        main_branch = pr_data['base']['ref']

        # 2. Auth
        token = get_installation_access_token(payload)

        # 3. Fetch Context
        full_diff = get_pr_diff(token, diff_url)
        repo_tree = get_repo_tree(token, repo_name, main_branch)

        # 4. Map Step (File Audits)
        file_diff_blocks = parse_diff_into_files(full_diff)
        file_diff_blocks = sorted(file_diff_blocks, key=lambda x: len(x), reverse=True)[:8]
        
        file_summaries = []
        for block in file_diff_blocks:
            path = extract_filepath(block)
            content = get_file_content(token, repo_name, path, head_sha)
            summary = _get_file_summary(path, content, block)
            file_summaries.append(summary)

        # 5. Reduce Step (Strategic Mentorship)
        final_report = get_strategic_summary(file_summaries, repo_tree)
        
        # 6. Post Result
        post_pr_comment(token, repo_name, pr_number, final_report)
        print(f"Analysis complete for {repo_name} PR #{pr_number}.")

    except Exception as e:
        print(f"Background Task Error: {e}")

# --- API Endpoints ---

@app.get("/")
async def health_check():
    """
    Standard 200 OK endpoint for GitHub Actions.
    Ensures the 'curl' check in the workflow passes reliably.
    """
    return JSONResponse(
        content={
            "status": "ready",
            "agent": "CodeBunny",
            "info": "https://rahul-jangra-leonado10000.vercel.app/projects/codebunny"
        },
        status_code=200
    )

@app.post("/webhook/github")
async def handle_github_webhook(request: Request, background_tasks: BackgroundTasks):
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # 1. IDENTIFY PAYLOAD SOURCE & FILTER
    try:
        if isinstance(payload.get('repository'), dict):
            repo_name = payload['repository']['full_name']
            pr_data = payload.get('pull_request')
            action = payload.get('action')
        else:
            repo_name = payload['repository']
            pr_data = payload.get('event', {}).get('pull_request')
            action = payload.get('event', {}).get('action')
    except (KeyError, AttributeError):
        return {"status": "ignored", "reason": "unrecognized_payload"}

    if not pr_data or action not in ['opened', 'reopened', 'synchronize']:
        return {"status": "ignored", "reason": "non_relevant_action"}

    # 2. RATE LIMIT
    count = USAGE_COUNTS.get(repo_name, 0)
    if count >= USAGE_LIMIT:
        return {"status": "ignored", "reason": "rate_limited"}
    USAGE_COUNTS[repo_name] = count + 1

    # 3. DISPATCH BACKGROUND TASK
    print(f"🐰 Dispatching background audit for {repo_name} PR #{pr_data['number']}...")
    background_tasks.add_task(run_analysis_task, payload)

    return {"status": "processing", "message": "Analysis dispatched to background task."}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)