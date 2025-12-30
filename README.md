<div align="center">

🐰 CodeBunny

The AI-Powered Code Review Agent

CodeBunny is a serverless, event-driven AI agent that acts as a Principal Engineer for your pull requests.

</div>

🚀 Overview

Most AI coding tools just summarize diffs. CodeBunny understands context.

It uses a novel Two-Pass "Map-Reduce" Architecture to analyze massive Pull Requests that would choke standard LLM wrappers. It doesn't just say what changed; it analyzes the repository structure to tell you why it matters and what the risks are.

Key Features

🧠 Two-Pass Brain: * Phase 1 (The Map): Parallelized analysis of individual files using fast models (GPT-4o mini).

Phase 2 (The Reduce): Strategic synthesis using high-reasoning models (GPT-4o mini/Gemini Pro) injected with full repository tree context.

🛡️ Enterprise-Grade Security: Authenticates as a GitHub App using RS256-signed JWTs. No personal access tokens required.

⚡ Event-Driven & Serverless: Deploys as a stateless Docker container on Render/Cloud Run. Triggered instantly by GitHub Webhooks.

🔍 Repo-Aware: Fetches and analyzes the file tree to detect high-risk changes (e.g., changes to auth/ or billing/ directories).

🏗️ Architecture

CodeBunny operates as a microservice triggered by GitHub Actions.

<img src="https://raw.githubusercontent.com/leonado10000/CodeBunny/refs/heads/main/data/flow.png" width="500" alt="Description of image" />



🛠️ Installation (For Your Repo)

You can add CodeBunny to any repository in 3 steps.

1. Deploy the Brain

Click the "Deploy to Render" button above, or deploy the Docker container manually.

Environment Variables Required:

PR_AGENT_APP_ID: Your GitHub App ID.

PR_AGENT_PRIVATE_KEY: Your App's Private Key (PEM format).

OPENAI_API_KEY: Your LLM Provider Key.

2. Connect the Action

In your repository, go to Settings > Secrets > Actions and add:

CODEBUNNY_SERVICE_URL: The URL of your deployed service (e.g., https://codebunny.onrender.com).

3. Add the Workflow

Create .github/workflows/codebunny.yml:

name: '🐰 CodeBunny Agent'

on:
  pull_request:
    types: [opened, reopened, synchronize]

jobs:
  dispatch_analysis:
    name: 🥕 Dispatch to HQ
    runs-on: ubuntu-latest
    steps:
      - name: 📡 Contacting CodeBunny Brain
        run: |
          curl -X POST "${{ secrets.CODEBUNNY_SERVICE_URL }}/webhook/github" \
          -H "Content-Type: application/json" \
          -d '${{ toJSON(github) }}' \
          --fail


🧪 Local Development

Clone the repo

git clone [https://github.com/yourusername/CodeBunny.git](https://github.com/yourusername/CodeBunny.git)
cd CodeBunny


Install Dependencies

pip install -r requirements-dev.txt


Run Tests
We use pytest-mock to simulate GitHub payloads locally.

python local_run.py


📜 License

MIT License. Built with ❤️ by [Your Name].

