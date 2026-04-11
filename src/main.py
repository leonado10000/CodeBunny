import os
import json
import requests
import uvicorn
import re
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import RedirectResponse

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

# --- API Endpoints ---

@app.get("/")
async def root_redirect():
    """
    Handles GET requests to the root URL.
    Redirects users to the project documentation/landing page.
    """
    return RedirectResponse(url="https://rahul-jangra-leonado10000.vercel.app/projects/codebunny")

@app.post("/webhook/github")
async def handle_github_webhook(request: Request):
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # 1. IDENTIFY PAYLOAD SOURCE
    try:
        if isinstance(payload.get('repository'), dict):
            # Direct GitHub Webhook
            repo_name = payload['repository']['full_name']
            pr_data = payload.get('pull_request')
            action = payload.get('action')
            head_sha = pr_data['head']['sha'] if pr_data else None
        else:
            # Action Context (Dispatch)
            repo_name = payload['repository']
            pr_data = payload.get('event', {}).get('pull_request')
            action = payload.get('event', {}).get('action')
            head_sha = pr_data['head']['sha'] if pr_data else None
    except (KeyError, AttributeError):
        return {"status": "ignored", "reason": "unrecognized_payload"}

    # 2. FILTER & RATE LIMIT
    if not pr_data or action not in ['opened', 'reopened', 'synchronize']:
        return {"status": "ignored", "reason": "non_relevant_action"}

    count = USAGE_COUNTS.get(repo_name, 0)
    if count >= USAGE_LIMIT:
        return {"status": "ignored", "reason": "rate_limited"}
    USAGE_COUNTS[repo_name] = count + 1

    print(f"🐰 CodeBunny auditing {repo_name} PR #{pr_data['number']}...")

    try:
        # 3. AUTH & CONTEXT
        token = get_installation_access_token(payload)
        diff_url = pr_data['diff_url']
        pr_number = pr_data['number']
        main_branch = pr_data['base']['ref']
        
        full_diff = get_pr_diff(token, diff_url)
        repo_tree = get_repo_tree(token, repo_name, main_branch)

        # 4. MAP STEP (File Audits)
        file_diff_blocks = parse_diff_into_files(full_diff)
        file_diff_blocks = sorted(file_diff_blocks, key=lambda x: len(x), reverse=True)[:8]
        
        file_summaries = []
        for block in file_diff_blocks:
            path = extract_filepath(block)
            content = get_file_content(token, repo_name, path, head_sha)
            summary = _get_file_summary(path, content, block)
            file_summaries.append(summary)

        # 5. REDUCE STEP (Strategic Mentorship)
        final_report = get_strategic_summary(file_summaries, repo_tree)
        
        # 6. POST TO GITHUB
        post_pr_comment(token, repo_name, pr_number, final_report)

        return {"status": "success"}

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)