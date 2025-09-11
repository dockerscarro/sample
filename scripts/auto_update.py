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

# ----------------- CALL GPT TO UPDATE main.py -----------------
update_prompt = f"""
You are an expert Python developer.

The following is the current main.py code:

--- main.py ---
{main_code}

Issue:
Title: {issue_title}
Description: {issue_body}

Update main.py to implement the requested changes.
- Preserve existing functionality.
- Integrate new code in the most appropriate place.
- Return ONLY the full updated main.py code inside triple backticks.
- Do NOT return explanations, only code.
"""

try:
    chat_model = ChatOpenAI(
        temperature=0,
        model="gpt-4o-mini",
        openai_api_key=OPENAI_API_KEY,
        max_tokens=4000
    )

    response = chat_model.invoke([HumanMessage(content=update_prompt)])
    merged_text = response.content.strip()

except Exception as e:
    print(f"❌ GPT update failed: {e}")
    exit(1)

# ----------------- EXTRACT FINAL UPDATED CODE -----------------
# Try to extract code inside triple backticks
code_blocks = re.findall(r"```(?:python)?(.*?)```", merged_text, flags=re.S)
if not code_blocks:
    code_blocks = [merged_text]  # fallback if GPT didn't use backticks

final_code = code_blocks[0].strip()

# ----------------- WRITE UPDATED main.py -----------------
with open(main_file, "w") as f:
    f.write(final_code)

print("✅ main.py successfully updated with issue changes.")

# ----------------- CREATE NEW BRANCH -----------------
branch_name = f"issue-{uuid.uuid4().hex[:8]}"
repo.git.checkout(main_branch)
repo.git.checkout("-b", branch_name)

# ----------------- COMMIT & PUSH -----------------
repo.git.add(all=True)
repo.git.commit("-m", f"Auto-update main.py for issue: {issue_title}")
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
