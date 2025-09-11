import os
import openai
from git import Repo
import uuid
import requests
import re

# ----------------- CONFIG -----------------
repo_dir = os.getcwd()
main_branch = "main"
target_files = ["main1.py", "main2.py", "main3.py", "main4.py", "main5.py"]
merged_file = "main.py"  # merged code, only in new branch

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
repo.git.fetch("origin", main_branch)

# ----------------- DETECT CHANGED FILES -----------------
changed_files = []
for f in target_files:
    diff = repo.git.diff(f"origin/{main_branch}", f)
    if diff.strip():  # if there is a diff
        changed_files.append(f)

if not changed_files:
    print("No changes detected in target files. Exiting.")
    exit(0)

files_content = {}
for f in changed_files:
    with open(f, "r") as file:
        files_content[f] = file.read()

print(f"Changed files sent to OpenAI: {changed_files}")

# ----------------- CREATE OPENAI PROMPT -----------------
prompt = f"""
Issue: {issue_title}
Description: {issue_body}

Below are multiple Python files that together make up the project. 
Please update ONLY the changed files as needed to resolve the issue. 
Important rules:
- Always return FULL UPDATED FILES, not snippets.
- Keep the file headers (### FILE: <filename>) exactly as given.
- Do NOT include Markdown markers like ```.

"""

for filename, content in files_content.items():
    prompt += f"\n### FILE: {filename}\n{content}\n"

# ----------------- CALL OPENAI -----------------
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}],
    temperature=0
)

updated_text = response.choices[0].message.content.strip()

# ----------------- PARSE UPDATED FILES -----------------
updated_files = {}
pattern = r"### FILE: ([^\n]+)\n([\s\S]*?)(?=(?:\n### FILE:|\Z))"
matches = re.findall(pattern, updated_text)

for filename, code in matches:
    updated_files[filename.strip()] = code.strip()

# Fallback: keep original if model missed a file
for f in changed_files:
    if f not in updated_files:
        updated_files[f] = files_content[f]

# ----------------- CREATE NEW BRANCH -----------------
branch_name = f"issue-{uuid.uuid4().hex[:8]}"
repo.git.checkout("-b", branch_name)

# ----------------- WRITE UPDATED FILES -----------------
for filename, code in updated_files.items():
    with open(filename, "w") as f:
        f.write(code)

# ----------------- COMBINE INTO main.py (only in new branch) -----------------
with open(merged_file, "w") as f:
    for fpart in target_files:
        with open(fpart, "r") as part_file:
            f.write(part_file.read() + "\n")

# ----------------- VERIFY ALL FILES -----------------
missing_files = [f for f in target_files if not os.path.exists(f)]
if missing_files:
    raise Exception(f"Missing files after update: {missing_files}")

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
