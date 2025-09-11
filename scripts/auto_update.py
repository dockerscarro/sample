import os
import openai
from git import Repo
import uuid
import requests
import re

# ----------------- CONFIG -----------------
repo_dir = os.getcwd()
main_branch = "main"
target_file = "main.py"           # only main.py exists
changes_file = "changes.py"       # will store OpenAI updates

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
with open(target_file, "r") as f:
    main_code = f.read()

# ----------------- CREATE OPENAI PROMPT -----------------
prompt = f"""
Issue: {issue_title}
Description: {issue_body}

Below is the current code in main.py:

{main_code}

Instructions:
1. Identify only the minimal code changes needed to address the issue.
2. Return the updated parts **inside markers**:
   ### UPDATED START ###
   <your code changes>
   ### UPDATED END ###
3. Do not modify unrelated code.
4. Preserve formatting and indentation.
5. Only return Python code, no explanations.
"""

# ----------------- CALL OPENAI -----------------
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}],
    temperature=0
)

updated_text = response.choices[0].message.content.strip()

# ----------------- SAVE CHANGES WITH MARKERS -----------------
with open(changes_file, "w") as f:
    f.write(updated_text)

# ----------------- REMOVE MARKERS -----------------
def remove_markers(code_with_markers):
    pattern = r"### UPDATED START ###[\s\S]*?### UPDATED END ###"
    def repl(match):
        # Keep everything between markers, remove marker lines
        inner = match.group(0).split("\n")[1:-1]
        return "\n".join(inner)
    return re.sub(pattern, repl, code_with_markers)

cleaned_update = remove_markers(updated_text)

# ----------------- MERGE INTO MAIN.PY -----------------
# If main.py already has old updated section, replace it
pattern = r"### UPDATED START ###[\s\S]*?### UPDATED END ###"
if re.search(pattern, main_code):
    main_code_new = re.sub(pattern, cleaned_update, main_code)
else:
    main_code_new = main_code + "\n\n" + cleaned_update

with open(target_file, "w") as f:
    f.write(main_code_new)

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
