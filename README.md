<div align="center">  
<img src="[https://raw.githubusercontent.com/leonado10000/CodeBunny/main/data/logo.png](https://www.google.com/url?sa=E&source=gmail&q=https://raw.githubusercontent.com/leonado10000/CodeBunny/main/data/logo.png)" alt="CodeBunny Logo" width="120" height="120">

# **🐰 CodeBunny**

### **The Blunt AI Code Reviewer**

CodeBunny is a high-speed AI agent that acts as a **Principal Engineer** for your Pull Requests. It skips the fluff and tells you exactly what needs to be fixed to meet professional standards.  
[**Install App**](https://www.google.com/search?q=https://github.com/apps/codebunny) • [**View Demo**](https://www.google.com/search?q=https://rahul-jangra-leonado10000.vercel.app/projects/codebunny) • [**Report Bug**](https://www.google.com/search?q=https://github.com/leonado10000/CodeBunny/issues)  
</div>

## **🚀 Overview**

Most AI tools just summarize code changes. **CodeBunny audits them.** By deconstructing your code structure before analysis, it focuses strictly on logic, architectural risk, and best practices. It behaves like a senior developer who has seen it all and wants your code to be production-ready—now.

## **🛠️ Key Features**

| Feature | Description |
| :---- | :---- |
| 👔 **Principal Persona** | Provides blunt, corrective feedback and one-line code examples. |
| 🏗️ **Structural Awareness** | Understands function boundaries and logic flow via AST mapping. |
| ⚡ **Instant Reviews** | Powered by Groq LPUs for near-instant feedback on your PRs. |
| 🚥 **Quality Gates** | Automatically flags missing docs, complex functions, and naming violations. |

## **📦 Tech Stack**

* **Brain:** Llama 3.3 (70B) via **Groq** for high-speed inference.  
* **Analysis:** Python AST (Abstract Syntax Tree) for deterministic code mapping.  
* **Linter:** Pylint integration for automated code quality scoring.  
* **Backend:** FastAPI & Uvicorn.  
* **Infrastructure:** GitHub Actions & Render.

## **📥 How to Use**

Add CodeBunny to your repo in **2 minutes**:

### **1️⃣ Install the App**

### **Grant the [CodeBunny GitHub App](https://www.google.com/search?q=https://github.com/apps/codebunny) access to your repository.**

### **2️⃣ Add the Workflow**

Create .github/workflows/codebunny.yml in your repo:  
```
name: '🐰 CodeBunny Agent'  
on:  
  pull\_request:  
    types: \[opened, reopened, synchronize\]

jobs:  
  review:  
    runs-on: ubuntu-latest  
    permissions: { pull-requests: write }  
    steps:  
      \- name: 🔍 Wellness Check  
        id: health  
        run: |  
          STATUS=$(curl \-s \-o /dev/null \-w "%{http\_code}" "https://codebunny-5o9o.onrender.com/")  
          if \[ "$STATUS" \-eq 200 \]; then echo "status=online" \>\> $GITHUB\_OUTPUT; fi

      \- name: 📡 Dispatch Analysis  
        if: steps.health.outputs.status \== 'online'  
        run: |  
          curl \-X POST "https://codebunny-5o9o.onrender.com/webhook/github" \\  
          \-H "Content-Type: application/json" \-d '${{ toJSON(github) }}' \--fail \--max-time 30
```
## **🧪 Local Testing**

Test the logic on your machine:

1. **Clone & Install**  
```
   git clone https://github.com/leonado10000/CodeBunny.git  
   cd CodeBunny  
   pip install \-r requirements.txt
```
3. **Run Audit Simulation**  
```
   python local\_run.py
```
<div align="center">  
Built with ❤️ by [Rahul Jangra](https://www.google.com/search?q=https://rahul-jangra-leonado10000.vercel.app/projects/codebunny)  
\</div\>
