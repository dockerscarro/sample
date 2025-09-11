import os
import openai
from git import Repo
import uuid
import requests

# ----------------- CONFIG -----------------
repo_dir = os.getcwd()
main_branch = "main"
main_file = "main.py"  # full 600+ lines
changes_file = "changes.py"  # only the modifications
GH_PAT = os.getenv("GH_PAT")
openai.api_key = os.getenv("OPENAI_API_KEY")
repo_owner, repo_name = os.getenv("GITHUB_REPOSITORY").split("/")
issue_title = os.getenv("ISSUE_TITLE")
issue_body = os.getenv("ISSUE_BODY")

# ----------------- GIT SETUP -----------------
repo = Repo(repo_dir)
repo.git.config("user.name", "github-actions[bot]")
repo.git.config("user.email", "github-actions[bot]@users.noreply.github.com")
repo.git.checkout(main_branch)
repo.git.fetch("origin", main_branch)

# ----------------- READ FULL main.py -----------------
with open(main_file, "r") as f:
    main_code = f.read()

# ----------------- CREATE OPENAI PROMPT -----------------
prompt = f"""
Issue: {issue_title}
Description: {issue_body}

Below is the current main.py code (full 600+ lines):

{main_code}

Instructions:
- Return only the code that needs to be modified or added to fix the issue.
- Include enough context so changes can be merged safely.
- Use comments ### UPDATED START and ### UPDATED END around each modified block.
- Do NOT return the entire main.py.
"""

# ----------------- CALL OPENAI -----------------
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}],
    temperature=0
)

modifications = response.choices[0].message.content.strip()
if not modifications:
    print("No changes detected by OpenAI. Exiting.")
    exit(0)

# Save the modifications separately for review
with open(changes_file, "w") as f:
    f.write(modifications)

print(f"✅ Changes written to {changes_file}")

# ----------------- MERGE MODIFICATIONS INTO main.py -----------------
def merge_updates(original_code, updates):
    """
    Simple approach: append modified blocks at the end.
    Can be enhanced to replace functions if needed.
    """
    merged_code = original_code + "\n\n# --- AUTO-UPDATE BLOCKS ---\n" + updates
    return merged_code

merged_code = merge_updates(main_code, modifications)

with open(main_file, "w") as f:
    f.write(merged_code)

print(f"✅ main.py updated with modifications")

# ----------------- CREATE NEW BRANCH -----------------
branch_name = f"issue-{uuid.uuid4().hex[:8]}"
repo.git.checkout("-b", branch_name)

# ----------------- COMMIT & PUSH -----------------
repo.git.add([main_file, changes_file])
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
    "body": f"Auto-generated update for issue:\n\n{issue_body}\n\nSee changes.py for modified code blocks."
}

r = requests.post(pr_url, headers=headers, json=pr_data)
if r.status_code == 201:
    print(f"✅ Pull request created: {r.json()['html_url']}")
else:
    print(f"❌ Failed to create PR: {r.status_code} {r.text}")
