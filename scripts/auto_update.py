import os
import openai
from git import Repo
import uuid
import requests
import re

# ----------------- CONFIG -----------------
repo_dir = os.getcwd()
main_branch = "main"
main_file = "main.py"  # single large file
changes_file = "changes.py"  # only changed blocks returned by OpenAI

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

# ----------------- READ main.py -----------------
with open(main_file, "r") as f:
    main_content = f.read()

# ----------------- CREATE OPENAI PROMPT -----------------
prompt = f"""
Issue: {issue_title}
Description: {issue_body}

You are given a large Python file (main.py). Only suggest the minimal code changes needed 
to resolve the issue. Do NOT return the full file. 

Return code ONLY with ### UPDATED START and ### UPDATED END markers.
Strip all explanations, comments, or text. Only code inside the markers should be returned.
"""

prompt += f"\n\n# Current main.py snippet:\n{main_content[:2000]}...\n"

# ----------------- CALL OPENAI -----------------
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}],
    temperature=0
)

changes_text = response.choices[0].message.content.strip()

if not changes_text:
    print("No changes suggested by OpenAI. Exiting.")
    exit(0)

# ----------------- KEEP ONLY CODE BLOCKS -----------------
blocks = re.findall(r"### UPDATED START([\s\S]*?)### UPDATED END", changes_text)
code_only_changes = "\n".join(blocks)

with open(changes_file, "w") as f:
    f.write(code_only_changes)

# ----------------- MERGE CHANGES INTO main.py -----------------
def merge_changes(original, changes):
    """
    Replace existing ### UPDATED START/END blocks with changes or append at end.
    """
    original_blocks = re.findall(r"### UPDATED START[\s\S]*?### UPDATED END", original)
    merged = original
    for block in re.findall(r"### UPDATED START[\s\S]*?### UPDATED END", changes):
        if original_blocks:
            merged = re.sub(r"### UPDATED START[\s\S]*?### UPDATED END", block, merged, count=1)
        else:
            merged += "\n\n" + block
    return merged

merged_content = merge_changes(main_content, code_only_changes)

# ----------------- CREATE NEW BRANCH -----------------
branch_name = f"issue-{uuid.uuid4().hex[:8]}"
repo.git.checkout("-b", branch_name)

# ----------------- WRITE UPDATED main.py -----------------
with open(main_file, "w") as f:
    f.write(merged_content)

# ----------------- COMMIT & PUSH -----------------
repo.git.add(all=True)
repo.git.commit("-m", f"Auto-update main.py for issue: {issue_title}")
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
