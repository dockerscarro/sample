import os
import re
import uuid
from git import Repo
import requests
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

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

# ----------------- DETECT INSERTION POINTS -----------------
# 1️⃣ Imports
imports_end = 0
for match in re.finditer(r'^(import .*|from .+ import .+)', main_code, flags=re.M):
    imports_end = match.end()

# 2️⃣ Functions
functions_end = imports_end
func_matches = list(re.finditer(r'^def .+:\s*$', main_code, flags=re.M))
if func_matches:
    last_func = func_matches[-1]
    # Find end of function body by detecting indentation change or next def
    func_body = main_code[last_func.start():]
    next_def = re.search(r'^\S', func_body, flags=re.M)
    if next_def:
        functions_end = last_func.start() + next_def.start()
    else:
        functions_end = len(main_code)

# ----------------- PREPARE PROMPT -----------------
gpt_prompt = f"""
Issue: {issue_title}
Description: {issue_body}

Only provide updates for main.py content.
- Do NOT touch imports at the top.
- Do NOT remove existing functions.
- If new functions are needed, place after existing functions.
- If modifying main Streamlit logic, place after functions.
- Return ONLY valid Python code inside triple backticks.
"""

# Send current main.py content so GPT can know where to merge
gpt_prompt += f"\n\nCURRENT CODE:\n```\n{main_code}\n```"

# ----------------- CALL GPT -----------------
try:
    chat_model = ChatOpenAI(
        temperature=0,
        model="gpt-4o-mini",
        openai_api_key=OPENAI_API_KEY,
        max_tokens=2000
    )

    response = chat_model([
        HumanMessage(content="You are a Python developer. Update main.py according to the issue without breaking its structure."),
        HumanMessage(content=gpt_prompt)
    ])

    updated_text = response.content.strip()

except Exception as e:
    print(f"❌ GPT analysis failed: {e}")
    exit(1)

# ----------------- EXTRACT CODE -----------------
code_blocks = re.findall(r"```(?:python)?(.*?)```", updated_text, flags=re.S)
if not code_blocks:
    print("❌ No code block found in GPT output.")
    exit(1)

new_code = code_blocks[0].strip()

# ----------------- WRITE changes.py -----------------
with open(changes_file, "w") as f:
    f.write(new_code + "\n")

# ----------------- OVERWRITE main.py -----------------
with open(main_file, "w") as f:
    f.write(new_code + "\n")

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
