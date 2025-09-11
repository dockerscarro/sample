import os
import re
import uuid
import time
import requests
from git import Repo
import openai
from openai.error import APIConnectionError, RateLimitError, InvalidRequestError

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

# ----------------- READ main.py -----------------
with open(main_file, "r") as f:
    main_code = f.read()

# ----------------- INSERT UNIQUE PLACEHOLDER -----------------
def insert_unique_placeholder(code, description="placeholder"):
    section_id = uuid.uuid4().hex[:8]
    marker_start = f"### UPDATED START {section_id} ###"
    marker_end = f"### UPDATED END {section_id} ###"

    pattern = r"(uploaded_file\s*=\s*st\.file_uploader\(.*\))"
    if re.search(pattern, code):
        code = re.sub(
            pattern,
            f"{marker_start}\n# {description}\n\\1\n{marker_end}",
            code
        )
    else:
        code += f"\n{marker_start}\n# {description}\n{marker_end}\n"
    
    return code, section_id

main_code, section_id = insert_unique_placeholder(main_code)

with open(main_file, "w") as f:
    f.write(main_code)

# ----------------- EXTRACT SECTION TO UPDATE -----------------
pattern = rf"### UPDATED START {section_id} ###(.*?)### UPDATED END {section_id} ###"
sections_to_update = re.findall(pattern, main_code, flags=re.S)
if not sections_to_update:
    print("No sections found to update. Exiting.")
    exit(0)

prompt = f"""
Issue: {issue_title}
Description: {issue_body}

Only provide updates for the marked section below.
Return the updated code exactly in the same format with markers.
Do NOT return full main.py.
"""

for section in sections_to_update:
    prompt += f"\n### SECTION ###\n{section.strip()}\n"

# ----------------- CALL OPENAI WITH RETRIES -----------------
max_retries = 3
for attempt in range(max_retries):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        updated_text = response.choices[0].message.content.strip()
        break
    except (APIConnectionError, RateLimitError) as e:
        print(f"OpenAI API error, retrying ({attempt+1}/{max_retries}): {e}")
        time.sleep(2 + attempt * 3)
    except InvalidRequestError as e:
        print(f"Invalid request: {e}")
        exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        exit(1)
else:
    raise RuntimeError("Failed to call OpenAI API after retries")

# ----------------- WRITE changes.py -----------------
with open(changes_file, "w") as f:
    f.write(updated_text + "\n")

# ----------------- CREATE NEW BRANCH -----------------
branch_name = f"issue-{uuid.uuid4().hex[:8]}"
repo.git.checkout("-b", branch_name)

# ----------------- MERGE UPDATES INTO main.py -----------------
updated_sections = re.findall(pattern, updated_text, flags=re.S)
for updated_section in updated_sections:
    main_code = re.sub(
        pattern,
        f"### UPDATED START {section_id} ###\n{updated_section.strip()}\n### UPDATED END {section_id} ###",
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
