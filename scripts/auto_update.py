import os
import uuid
import re
from git import Repo
import requests
from langchain.chat_models import ChatOpenAI
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

# ----------------- AUTO MARK SECTIONS -----------------
def mark_section(code, pattern, description="placeholder"):
    section_id = uuid.uuid4().hex[:8]
    marker_start = f"### UPDATED START {section_id} ###"
    marker_end = f"### UPDATED END {section_id} ###"

    if re.search(pattern, code, flags=re.S):
        code = re.sub(
            pattern,
            f"{marker_start}\n# {description}\n\\1\n{marker_end}",
            code,
            flags=re.S
        )
    else:
        # Append at the end if pattern not found
        code += f"\n{marker_start}\n# {description}\n{marker_end}\n"

    return code, section_id

# Patterns to mark automatically
patterns_to_mark = [
    (r"(uploaded_file\s*=\s*st\.file_uploader\(.*\))", "File uploader placeholder"),
    (r"(def color_log_message\(.*?\):.*?return message\s*)", "Color log function"),
    (r"(for log in logs:.*?unsafe_allow_html=True\))", "Log display loop")
]

section_ids = []

for pat, desc in patterns_to_mark:
    main_code, sid = mark_section(main_code, pat, desc)
    section_ids.append(sid)

with open(main_file, "w") as f:
    f.write(main_code)

# ----------------- EXTRACT SECTIONS -----------------
sections_to_update = {}
for sid in section_ids:
    pattern = rf"### UPDATED START {sid} ###(.*?)### UPDATED END {sid} ###"
    matches = re.findall(pattern, main_code, flags=re.S)
    if matches:
        sections_to_update[sid] = matches[0]

if not sections_to_update:
    print("No sections found to update. Exiting.")
    exit(0)

# ----------------- PREPARE GPT PROMPT -----------------
gpt_prompt = f"""
Issue: {issue_title}
Description: {issue_body}

Update the marked sections below. Return code only inside markers.
Do NOT return full main.py.
"""

for sid, code_block in sections_to_update.items():
    gpt_prompt += f"\n### SECTION {sid} ###\n{code_block.strip()}\n"

# ----------------- CALL GPT -----------------
try:
    chat_model = ChatOpenAI(
        temperature=0,
        model="gpt-4o-mini",
        openai_api_key=OPENAI_API_KEY,
        max_tokens=2000
    )

    response = chat_model([
        HumanMessage(content="You are a Python developer. Update the marked code sections only."),
        HumanMessage(content=gpt_prompt)
    ])

    updated_text = response.content.strip()

except Exception as e:
    print(f"❌ GPT analysis failed: {e}")
    exit(1)

# ----------------- WRITE changes.py -----------------
with open(changes_file, "w") as f:
    f.write(updated_text + "\n")

# ----------------- CREATE NEW BRANCH -----------------
branch_name = f"issue-{uuid.uuid4().hex[:8]}"
repo.git.checkout("-b", branch_name)

# ----------------- MERGE UPDATES -----------------
for sid in section_ids:
    pattern = rf"### UPDATED START {sid} ###(.*?)### UPDATED END {sid} ###"
    updated_sections = re.findall(pattern, updated_text, flags=re.S)
    if updated_sections:
        main_code = re.sub(
            pattern,
            f"### UPDATED START {sid} ###\n{updated_sections[0].strip()}\n### UPDATED END {sid} ###",
            main_code,
            flags=re.S
        )

with open(main_file, "w") as f:
    f.write(main_code)

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
