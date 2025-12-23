import os
import json
import requests
import uvicorn
import re
from fastapi import FastAPI, Request, HTTPException

# Use absolute imports
from src.auth import get_installation_access_token
from src.agent import _get_file_summary, get_strategic_summary

app = FastAPI()

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
        print(f"FATAL: Failed to fetch PR diff. Status: {e.response.status_code}, Body: {e.response.text}")
        raise

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
        if len(paths) > 2000:
            paths = paths[:2000] + ["... (truncated)"]
        return "\n".join(paths)
    except Exception as e:
        print(f"Warning: Could not fetch repo tree. Proceeding without it. Error: {e}")
        return "Repository file tree was not available."

def post_pr_comment(token: str, repo_full_name: str, pr_number: int, body: str):
    url = f"https://api.github.com/repos/{repo_full_name}/issues/{pr_number}/comments"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {"body": body}
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        print(f"Successfully posted comment to PR #{pr_number}.")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"FATAL: Failed to post comment. Status: {e.response.status_code}, Body: {e.response.text}")
        raise

def parse_diff_into_files(full_diff: str) -> list[str]:
    file_diffs = re.split(r'(?=diff --git a/)', full_diff)
    return [diff.strip() for diff in file_diffs if diff.strip()]

# --- The Main API Endpoint ---

@app.post("/webhook/github")
async def handle_github_webhook(request: Request):
    try:
        github_context = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    print("PR_AGENT analysis initiated.")
    try:
        # --- FIX: Handle repository string correctly ---
        # The Actions context provides 'repository' as a string "owner/repo"
        repo_full_name = github_context['repository']
        
        # We dig into the event payload for PR details
        pr_number = github_context['event']['pull_request']['number']
        diff_url = github_context['event']['pull_request']['diff_url']
        main_branch = github_context['event']['pull_request']['base']['ref']
        # -----------------------------------------------
        
        # 2. Authenticate
        print("Authenticating as PR_AGENT...")
        access_token = get_installation_access_token(github_context)
        print("Authentication successful.")

        # 3. Get All Context
        print("Fetching PR diff...")
        full_diff_text = get_pr_diff(access_token, diff_url)
        
        print("Fetching repo tree...")
        repo_tree = get_repo_tree(access_token, repo_full_name, main_branch)

        # 4. Run the "Two-Pass Brain"
        print("Parsing diff into individual files...")
        file_diffs = parse_diff_into_files(full_diff_text)
        
        print(f"Generating file-level summaries for {len(file_diffs)} files...")
        file_summaries = [_get_file_summary(diff) for diff in file_diffs]

        print("Generating strategic summary...")
        final_summary = get_strategic_summary(file_summaries, repo_tree)
        
        # 5. Post the result
        print(f"Posting final summary to {repo_full_name} PR #{pr_number}...")
        post_pr_comment(access_token, repo_full_name, pr_number, final_summary)

        print("PR_AGENT analysis complete.")
        return {"status": "success"}

    except KeyError as e:
        print(f"KeyError: Missing expected key in GitHub payload. Key: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid GitHub webhook payload: Missing key {e}")
    except Exception as e:
        print(f"An unhandled error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))