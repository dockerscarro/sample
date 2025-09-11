import os
import openai
from git import Repo
import uuid
import requests
import re

# ----------------- CONFIG -----------------
repo_dir = os.getcwd()
main_branch = "main"
main_file = "main.py"
changes_file = "changes.py"

openai.api_key = os.getenv("OPENAI_API_KEY")
GH_PAT = os.getenv("GH_PAT")
repo_owner, repo_name = os.getenv("GITHUB_REPOSITORY").split("/")
issue_title = os.getenv("ISSUE_TITLE")
issue_body = os.getenv("ISSUE_BODY")

# ----------------- GIT SETUP -----------------
repo = Repo(repo_dir)
repo.git.config("user.name", "github-actions[bot]")
repo.git.config("user.email", "github-actions[bot]@users.noreply.github.com")
repo.git.checkout(main_branch)

# ----------------- READ MAIN FILE -----------------
with open(main_file, "r") as f:
    main_code = f.read()

# ----------------- CREATE OPENAI PROMPT -----------------
prompt = f"""
Issue: {issue_title}
Description: {issue_body}

You are given the following main.py code:

{main_code}

Instructions:
- Only return the Python code parts that need to be changed or added.
- Do NOT return the entire 600+ lines.
- Do NOT include explanations or markdown.
- Mark updated sections with:
### UPDATED START
<updated code here>
### UPDATED END
"""

# ----------------- CALL OPENAI -----------------
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}],
    temperature=0
)

changes_text = response.choices[0].message.content.strip()

# Exit if no changes
if not changes_text:
    print("No changes detected. Exiting.")
    exit(0)

# ----------------- WRITE CHANGES -----------------
with open(changes_file, "w") as f:
    f.write(changes_text)

# ----------------- MERGE CHANGES INTO MAIN -----------------
updated_sections = re.findall(r"### UPDATED START(.*?)### UPDATED END", changes_text, flags=re.S)
merged_code = main_code
for section in updated_sections:
    merged_code = re.sub(r"### UPDATED START.*?### UPDATED END", section.strip(), merged_code, flags=re.S)

# Write updated main.py in new branch
branch_name = f"issue-{uuid.uuid4().hex[:8]}"
repo.git.checkout("-b", branch_name)

with open(main_file, "w") as f:
    f.write(merged_code)

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
