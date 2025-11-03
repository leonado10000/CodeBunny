import os
import json
import time
from github import Github, GithubException
from tqdm import tqdm
from dotenv import load_dotenv

# --- CONFIGURATION ---

# Repositories to scrape for high-quality review comments
# Format: "owner/repo"
TARGET_REPOS = [
    "rust-lang/rust",
    "microsoft/vscode",
    "kubernetes/kubernetes",
    "pytorch/pytorch"
]

# How many recent, merged PRs to check from each repo?
# GitHub's API limits this, but we'll paginate.
PRS_TO_CHECK_PER_REPO = 500

# Minimum reactions on a comment to be considered "gold"
MIN_REACTIONS = 2

# Output file
OUTPUT_FILE = "training_data.jsonl"

# --- HELPER FUNCTION ---

def create_training_example(system_prompt, user_content, assistant_content):
    """Formats the data into the JSONL structure required by OpenAI."""
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }

# --- MAIN SCRIPT ---

def main():
    load_dotenv()
    github_token = os.getenv("GITHUB_PAT")
    if not github_token:
        print("FATAL: GITHUB_PAT (Personal Access Token) not found in .env file.")
        print("Please create a PAT with 'public_repo' scope and add it to your .env.")
        return

    try:
        g = Github(github_token)
        # Test authentication
        print(f"Authenticated as: {g.get_user().login}")
    except Exception as e:
        print(f"Error authenticating with GitHub: {e}")
        return

    system_prompt = "You are a code analysis bot. Summarize the changes in this diff file in a concise, technical, bullet-point format."
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        total_examples_found = 0
        for repo_name in TARGET_REPOS:
            try:
                repo = g.get_repo(repo_name)
                print(f"\n--- Scraping Repository: {repo_name} ---")
                
                # Get recent merged PRs
                prs = repo.get_pulls(state="closed", sort="updated", direction="desc")
                
                # tqdm creates a smart progress bar
                for pr in tqdm(prs.get_page(0)[:PRS_TO_CHECK_PER_REPO], total=PRS_TO_CHECK_PER_REPO, desc=f"Scanning {repo_name} PRs"):
                    if not pr.merged:
                        continue
                        
                    pr_author = pr.user.login
                    
                    try:
                        # We are interested in review comments, which are tied to code
                        comments = pr.get_review_comments()
                        for comment in comments:
                            # 1. Filter for "gold" comments
                            is_high_quality = (
                                comment.body and               # Has a body
                                len(comment.body) > 50 and     # Is not just "LGTM"
                                comment.user.login != pr_author and # Is not from the author
                                (comment.reactions.total_count >= MIN_REACTIONS or
                                 comment.reactions.plus_one >= MIN_REACTIONS) # Has reactions
                            )
                            
                            if is_high_quality:
                                # 2. We found one. Get the context.
                                user_content = comment.diff_hunk
                                assistant_content = comment.body
                                
                                # 3. Create the training example
                                example = create_training_example(
                                    system_prompt,
                                    user_content,
                                    assistant_content
                                )
                                
                                # 4. Write to file
                                f.write(json.dumps(example) + "\n")
                                total_examples_found += 1

                    except GithubException as ge:
                        if ge.status == 404:
                            print(f"Skipping PR (comments not found, likely deleted): {pr.number}")
                        else:
                            print(f"GitHub API error on PR {pr.number}: {ge}")
                    except Exception as e:
                        print(f"An error occurred processing PR {pr.number}: {e}")
                    
                    # Be nice to the API
                    time.sleep(0.1) 

            except GithubException as ge:
                if ge.status == 404:
                    print(f"Repository not found or access denied: {repo_name}")
                else:
                    print(f"GitHub API error on repo {repo_name}: {ge}")
            except Exception as e:
                print(f"An error occurred processing repo {repo_name}: {e}")

    print(f"\n--- Curation Complete ---")
    print(f"Total high-quality examples found: {total_examples_found}")
    print(f"Dataset saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()