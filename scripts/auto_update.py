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

# ----------------- READ main.py -----------------
with open(main_file, "r") as f:
    main_code = f.read()

# ----------------- CALL GPT TO GENERATE CHANGES -----------------
generate_prompt = f"""
You are a Python developer.

Analyze the following main.py file and generate necessary code modifications
based on this issue:

Issue Title: {issue_title}
Issue Body: {issue_body}

Return only the modifications as valid Python code. Preferably use triple backticks
with python, but if not, just the code is fine.
"""

try:
    chat_model = ChatOpenAI(
        temperature=0,
        model="gpt-4o-mini",
        openai_api_key=OPENAI_API_KEY,
        max_tokens=1500
    )

    response = chat_model.invoke([HumanMessage(content=generate_prompt)])
    modifications = response.content.strip()

except Exception as e:
    print(f"❌ GPT generation failed: {e}")
    exit(1)

# ----------------- WRITE modifications TO changes.py -----------------
with open(changes_file, "w") as f:
    f.write(modifications + "\n")

# ----------------- MERGE changes.py INTO main.py -----------------
with open(changes_file, "r") as f:
    changes_code = f.read()

if not changes_code.strip():
    print("⚠️ changes.py is empty. Nothing to merge.")
    exit(0)

merge_prompt = f"""
You are a Python developer.

Merge the following changes into main.py intelligently.

--- main.py ---
{main_code}

--- changes.py ---
{changes_code}

Instructions:
- Integrate the changes into the appropriate place(s) in main.py.
- Preserve all existing functionality.
- Return ONLY the final Python code. Preferably inside triple backticks.
"""

try:
    merge_response = chat_model.invoke([HumanMessage(content=merge_prompt)])
    merged_text = merge_response.content.strip()

except Exception as e:
    print(f"❌ GPT merge failed: {e}")
    exit(1)

# ----------------- EXTRACT FINAL CODE -----------------
# Try triple backticks first, fallback to entire response
code_blocks = re.findall(r"```(?:python)?(.*?)```", merged_text, flags=re.S)
if not code_blocks:
    code_blocks = [merged_text]

final_code = code_blocks[0].strip()

with open(main_file, "w") as f:
    f.write(final_code)

print("✅ main.py successfully merged with changes.py")

# ----------------- CREATE NEW BRANCH -----------------
branch_name = f"issue-{uuid.uuid4().hex[:8]}"
repo.git.checkout(main_branch)
repo.git.checkout("-b", branch_name)

# ----------------- COMMIT & PUSH -----------------
repo.git.add(all=True)
repo.git.commit("-m", f"Auto-merge changes for issue: {issue_title}")
repo.git.push("--set-upstream", "origin", branch_name)

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
    "body": f"Auto-generated update for issue:\n\n{issue_body}"
}

r = requests.post(pr_url, headers=headers, json=pr_data)
if r.status_code == 201:
    print(f"✅ Pull request created: {r.json()['html_url']}")
else:
    print(f"❌ Failed to create PR: {r.status_code} {r.text}")
