import os
import uuid
from git import Repo
import requests
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import re

# ----------------- CONFIG -----------------
repo_dir = os.getcwd()
main_branch = "main"
main_file = "main.py"
changes_file = "changes.py"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GH_PAT = os.getenv("GH_PAT")
repo_owner, repo_name = os.getenv("GITHUB_REPOSITORY").split("/")
issue_title = os.getenv("ISSUE_TITLE")
issue_body = os.getenv("ISSUE_BODY")

# ----------------- GIT SETUP -----------------
repo = Repo(repo_dir)
repo.git.config("user.name", "github-actions[bot]")
repo.git.config("user.email", "github-actions[bot]@users.noreply.github.com")
repo.git.checkout(main_branch)

# ----------------- READ FILES -----------------
with open(main_file, "r") as f:
    main_code = f.read()

# Ensure changes.py exists
if not os.path.exists(changes_file):
    with open(changes_file, "w") as f:
        f.write("")

# Skip merge if changes.py is empty
if os.path.getsize(changes_file) == 0:
    print("⚠️ changes.py is empty. Nothing to merge.")
    exit(0)

with open(changes_file, "r") as f:
    changes_code = f.read()

# ----------------- PREPARE GPT PROMPT -----------------
gpt_prompt = f"""
You are an expert Python developer.

Merge the following changes into main.py intelligently.

--- main.py ---
{main_code}

--- changes.py ---
{changes_code}

Instructions:
- Produce the final merged main.py with all changes applied.
- Preserve all existing functionality.
- Return ONLY the final Python code inside triple backticks.
- Do NOT include any explanations or extra text.
"""

# ----------------- CALL GPT -----------------
try:
    chat_model = ChatOpenAI(
        temperature=0,
        model="gpt-4o-mini",
        openai_api_key=OPENAI_API_KEY,
        max_tokens=3000
    )

    response = chat_model([
        HumanMessage(content=gpt_prompt)
    ])

    merged_text = response.content.strip()

except Exception as e:
    print(f"❌ GPT merge failed: {e}")
    exit(1)

# ----------------- EXTRACT FINAL CODE -----------------
code_blocks = re.findall(r"```(?:python)?(.*?)```", merged_text, flags=re.S)
if not code_blocks:
    print("❌ No code block returned by GPT.")
    exit(1)

final_code = code_blocks[0].strip()

with open(main_file, "w") as f:
    f.write(final_code)

print("✅ main.py successfully merged with changes.py")

# ----------------- CREATE NEW BRANCH -----------------
branch_name = f"issue-{uuid.uuid4().hex[:8]}"
repo.git.checkout("-b", branch_name)

# ----------------- COMMIT & PUSH -----------------
repo.git.add(all=True)
repo.git.commit("-m", f"Auto-merge changes for issue: {issue_title}")
repo.git.push("origin", branch_name)

# ----------------- CREATE PULL REQUEST -----------------
pr_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/pulls"
headers = {
    "Authorization": f"token {GH_PAT}",
    "Accept": "application/vnd.github+json"
}
pr_data = {
    "title": f"Fix: {issue_title}",
    "head": branch_name,
    "base": main_branch,
    "body": f"Auto-generated merge for issue:\n\n{issue_body}"
}

r = requests.post(pr_url, headers=headers, json=pr_data)
if r.status_code == 201:
    print(f"✅ Pull request created: {r.json()['html_url']}")
else:
    print(f"❌ Failed to create PR: {r.status_code} {r.text}")
