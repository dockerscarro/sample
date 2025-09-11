import os
import uuid
from git import Repo
import requests
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from difflib import SequenceMatcher
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

# ----------------- PREPARE GPT PROMPT -----------------
gpt_prompt = f"""
Issue: {issue_title}
Description: {issue_body}

Provide ONLY valid Python code inside triple backticks.
Do NOT return full main.py. Only the code that should be updated or added.
"""

# ----------------- CALL GPT VIA LangChain -----------------
try:
    chat_model = ChatOpenAI(
        temperature=0,
        model="gpt-4o-mini",
        openai_api_key=OPENAI_API_KEY,
        max_tokens=1500
    )

    response = chat_model([
        HumanMessage(content="You are a Python developer. Update or add only the necessary code."),
        HumanMessage(content=gpt_prompt)
    ])

    updated_text = response.content.strip()

except Exception as e:
    print(f"❌ GPT analysis failed: {e}")
    exit(1)

# ----------------- WRITE changes.py -----------------
with open(changes_file, "w") as f:
    f.write(updated_text + "\n")

# ----------------- MERGE CHANGES INTO main.py -----------------
# Extract Python code block(s) from GPT response
code_blocks = re.findall(r"```(?:python)?(.*?)```", updated_text, flags=re.S)
if not code_blocks:
    print("❌ No code block found in GPT output.")
    exit(1)

changes_lines = code_blocks[0].strip().splitlines()
main_lines = main_code.splitlines()
matcher = SequenceMatcher(None, main_lines, changes_lines)

merged_lines = []
for tag, i1, i2, j1, j2 in matcher.get_opcodes():
    if tag == 'equal':
        merged_lines.extend(main_lines[i1:i2])
    elif tag in ('replace', 'delete', 'insert'):
        # Replace or insert with GPT lines
        merged_lines.extend(changes_lines[j1:j2])

# Save merged main.py
main_code = "\n".join(merged_lines) + "\n"
with open(main_file, "w") as f:
    f.write(main_code)

# ----------------- CREATE NEW BRANCH -----------------
branch_name = f"issue-{uuid.uuid4().hex[:8]}"
repo.git.checkout("-b", branch_name)

# ----------------- COMMIT & PUSH -----------------
repo.git.add(all=True)
repo.git.commit("-m", f"Auto-update for issue: {issue_title}")
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
    "body": f"Auto-generated update for issue:\n\n{issue_body}"
}

r = requests.post(pr_url, headers=headers, json=pr_data)
if r.status_code == 201:
    print(f"✅ Pull request created: {r.json()['html_url']}")
else:
    print(f"❌ Failed to create PR: {r.status_code} {r.text}")
