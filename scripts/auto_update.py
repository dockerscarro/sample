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

MAX_RETRIES = 5  # Retry GPT merge until fully valid

# ----------------- GIT SETUP -----------------
repo = Repo(repo_dir)
repo.git.config("user.name", "github-actions[bot]")
repo.git.config("user.email", "github-actions[bot]@users.noreply.github.com")
repo.git.checkout(main_branch)

# ----------------- READ main.py -----------------
with open(main_file, "r") as f:
    main_code = f.read()

# ----------------- STEP 1: GENERATE CHANGES -----------------
generate_changes_prompt = f"""
You are an expert Python developer.

Current main.py:

--- main.py ---
{main_code}

Issue to solve:
Title: {issue_title}
Description: {issue_body}

Generate ONLY the modifications needed to address this issue.
- Do not rewrite the entire file.
- Return ONLY valid Python code inside triple backticks (no explanations).
"""

chat_model = ChatOpenAI(
    temperature=0,
    model="gpt-4o-mini",
    openai_api_key=OPENAI_API_KEY,
    max_tokens=1500
)

try:
    response = chat_model.invoke([HumanMessage(content=generate_changes_prompt)])
    changes_code = response.content.strip()
except Exception as e:
    print(f"❌ GPT failed to generate changes: {e}")
    exit(1)

# ----------------- WRITE changes.py -----------------
with open(changes_file, "w") as f:
    f.write(changes_code + "\n")

if not changes_code.strip():
    print("⚠️ changes.py is empty. Nothing to merge.")
    exit(0)

print("✅ changes.py created successfully.")

# ----------------- STEP 2: MERGE INTO COMPLETE main.py -----------------
retry_count = 0
merged_successfully = False

while retry_count < MAX_RETRIES:
    merge_prompt = f"""
You are an expert Python developer.

Current main.py:

--- main.py ---
{main_code}

Generated changes.py:

--- changes.py ---
{changes_code}

Apply the changes.py modifications to main.py:
- Return the full, complete, and fully valid Python code of main.py.
- Make all changes, remove or replace old code if needed.
- Ensure the final main.py is executable and contains nothing missing.
- Return ONLY the complete code inside triple backticks.
- Do NOT include explanations.
"""
    try:
        merged_response = chat_model.invoke([HumanMessage(content=merge_prompt)])
        merged_text = merged_response.content.strip()
    except Exception as e:
        print(f"❌ GPT failed to merge changes: {e}")
        exit(1)

    # Extract code from triple backticks
    code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", merged_text, flags=re.S)
    if code_blocks:
        final_code = code_blocks[0].strip()
    else:
        # fallback: remove any leading/trailing backticks
        final_code = "\n".join(
            line for line in merged_text.splitlines() if not line.strip().startswith("```")
        ).strip()

    # Remove any invisible/non-ASCII characters
    final_code = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]+", "", final_code)

    # Syntax check
    try:
        compile(final_code, main_file, "exec")
        merged_successfully = True
        print(f"✅ main.py merged successfully on attempt {retry_count + 1}")
        break
    except SyntaxError as e:
        print(f"⚠️ Syntax error detected: {e}. Retrying GPT merge ({retry_count + 1}/{MAX_RETRIES})...")
        retry_count += 1

if not merged_successfully:
    print("❌ Failed to generate fully valid main.py after multiple retries.")
    with open("failed_gpt_output.py", "w") as f:
        f.write(final_code)
    exit(1)

# ----------------- WRITE UPDATED main.py -----------------
with open(main_file, "w") as f:
    f.write(final_code)

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
    "title": f"Fix/Update: {issue_title}",
    "head": branch_name,
    "base": main_branch,
    "body": f"Auto-generated update for issue:\n\n{issue_body}"
}

r = requests.post(pr_url, headers=headers, json=pr_data)
if r.status_code == 201:
    print(f"✅ Pull request created: {r.json()['html_url']}")
else:
    print(f"❌ Failed to create PR: {r.status_code} {r.text}")
