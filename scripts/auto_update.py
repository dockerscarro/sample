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
merged_file = "main.py"  # full combined code in new branch

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

# ----------------- READ ALL SPLIT FILES -----------------
files_content = {}
for f in target_files:
    with open(f, "r") as file:
        files_content[f] = file.read()

# ----------------- DETERMINE FILES THAT NEED CHANGES -----------------
prompt_detect = f"""
Issue: {issue_title}
Description: {issue_body}

Below are all Python files in the project. 
Please identify which files need updates to fix the issue. 
Return a list of filenames only.
Even if only a single line needs a change, include that file.
Do NOT include explanations or code yet.
"""

for filename, content in files_content.items():
    prompt_detect += f"\n### FILE: {filename}\n{content}\n"

# Call OpenAI to detect changed files
response_detect = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt_detect}],
    temperature=0
)

changed_files_raw = response_detect.choices[0].message.content.strip().splitlines()
changed_files = [f.strip() for f in changed_files_raw if f.strip() in target_files]

# ----------------- FALLBACK: use all files if none detected -----------------
if not changed_files:
    print("⚠️ No files detected by OpenAI, defaulting to all files.")
    changed_files = target_files

print(f"Files to be updated by OpenAI: {changed_files}")

# ----------------- CREATE PROMPT FOR UPDATING FILES -----------------
prompt_update = f"""
Issue: {issue_title}
Description: {issue_body}

Please update ONLY the following Python files to resolve the issue. 
Return the full updated content for each file.
Keep file headers exactly as ### FILE: <filename>.
Do NOT include markdown formatting.
"""

for f in changed_files:
    prompt_update += f"\n### FILE: {f}\n{files_content[f]}\n"

# Call OpenAI to get updated code
response_update = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt_update}],
    temperature=0
)

updated_text = response_update.choices[0].message.content.strip()

# ----------------- PARSE UPDATED FILES -----------------
updated_files = {}
pattern = r"### FILE: ([^\n]+)\n([\s\S]*?)(?=(?:\n### FILE:|\Z))"
matches = re.findall(pattern, updated_text)

for filename, code in matches:
    updated_files[filename.strip()] = code.strip()

# Fallback: keep original content if OpenAI missed a file
for f in changed_files:
    if f not in updated_files:
        updated_files[f] = files_content[f]

# ----------------- CREATE NEW BRANCH -----------------
branch_name = f"issue-{uuid.uuid4().hex[:8]}"
print(f"Creating new branch: {branch_name}")
repo.git.checkout("-b", branch_name)

# ----------------- WRITE UPDATED FILES -----------------
for filename in target_files:
    content_to_write = updated_files.get(filename, files_content[filename])
    with open(filename, "w") as f:
        f.write(content_to_write)

# ----------------- MERGE ALL FILES INTO main.py -----------------
with open(merged_file, "w") as f:
    for fpart in target_files:
        with open(fpart, "r") as part_file:
            f.write(part_file.read() + "\n")

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
