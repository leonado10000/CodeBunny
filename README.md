<div align="center">

<!-- Replace with your own logo later! -->

<h1>CodeBunny 🐰</h1>

<p>
<strong>Your new AI-powered code review assistant.</strong>
</p>
<p>
CodeBunny helps you review pull requests faster, smarter, and with more context—so you can merge with confidence.
</p>

<!-- Badges - Add your own links later -->

<p>
<img src="https://www.google.com/search?q=https://img.shields.io/badge/build-passing-brightgreen" alt="Build Status">
<img src="https://www.google.com/search?q=https://img.shields.io/badge/license-MIT-blue" alt="License">
<img src="https://www.google.com/search?q=https://img.shields.io/github/stars/your-username/codebunny%3Fstyle%3Dsocial" alt="GitHub Stars">
</p>

</div>

What is CodeBunny?

Hello there! 👋 CodeBunny is a GitHub App that acts as a friendly, super-intelligent assistant on your team. When you open a pull request, CodeBunny hops in, reads the changes, and posts a comprehensive analysis right in the comments.

It's not just another "summary" bot. CodeBunny is built on a powerful multi-AI architecture to provide a deep, three-pillar analysis of every PR:

✨ The Summary (The "What"): A clean, high-level overview of what changed.

🤔 Inferred Rationale (The "Why"): An intelligent guess at the business or technical reason behind the change.

🚨 Consequence Analysis (The "What Next"): A breakdown of potential risks, side effects, and items to double-check before merging.

[IMAGE-PLACEHOLDER: A large, high-quality GIF showing CodeBunny posting its full three-pillar analysis on a new Pull Request.]
<p align="center">CodeBunny delivering a full review on a new PR.</p>

</div>

Core Features

Three-Pillar Analysis: Get the "What," "Why," and "What Next" for every PR.

🧠 Handles Huge PRs: CodeBunny uses a special Two-Pass Brain to analyze pull requests of any size—even 10,000+ lines—that make other bots choke.

🤖 Multi-Provider AI: It intelligently uses a mix of the best models (like GPT-4o mini for speed and Gemini Pro for reasoning) to give you the highest quality analysis at the lowest cost.

✅ One-Click Install: No complex setup or CI/CD configuration. Just install the GitHub App, and you're done.

(Coming Soon) Deep Repo Context: CodeBunny will understand your entire file structure, not just the diff, to warn you about changes in sensitive areas (like /src/core/security/).

Why CodeBunny?

We've all been there. A teammate opens a massive pull request. Reviewing it feels like a chore, and generic AI summaries don't really help you understand the risk.

CodeBunny is different by design.

It Won't Choke: Most review bots fail on large PRs. CodeBunny's "Two-Pass Brain" first uses a team of "Analyst" AIs to summarize each file in parallel. Then, it sends those summaries to a "Strategist" AI to synthesize the final, high-level report. It's built for real-world complexity.

It Provides Insight, Not Just Info: The "Consequence Analysis" is a game-changer. It's like having a senior engineer tap you on the shoulder and say, "This looks good, but did you check if the database migration is backward-compatible?"

[IMAGE-PLACEHOLDER: A GIF showing CodeBunny's analysis of a very large PR, proving it can handle it.]
<p align="center">CodeBunny handling a 10,000+ line PR with ease.</p>

</div>




Getting Started

Getting CodeBunny on your team is a breeze. We're packaging it as a one-click GitHub App (coming soon!).

Go to the CodeBunny GitHub Marketplace Page [Link to be added]

Click 'Install' and choose the repositories you want CodeBunny to watch.

...That's it. 🎉

The next time you open a pull request, CodeBunny will be there to help!

We're Just Getting Started!

CodeBunny is a new project, and we're buzzing with ideas. Our roadmap includes:

Custom fine-tuned models to give CodeBunny a unique, expert personality.

A/B testing and MLOps integration to publicly prove our model's quality.

Deeper repo context to understand the full impact of a change.

Contributing

We'd love your help! This is an open-source project for developers, by developers. Whether you're a prompt engineer, a front-end dev, or an MLOps guru, there's a place for you.

Check out our CONTRIBUTING.md file to get started. PRs are always welcome!

<p align="center">
Made with ❤️ by Rahul
</p>
<p align="center">
Licensed under the <a href="LICENSE">MIT License</a>.

</p>
