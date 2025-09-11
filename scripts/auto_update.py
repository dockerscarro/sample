import os
import uuid
import ast
import astor
from git import Repo
import requests
from langchain_openai import ChatOpenAI
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

# ----------------- READ main.py AND changes.py -----------------
with open(main_file, "r") as f:
    main_code = f.read()

with open(changes_file, "r") as f:
    changes_code = f.read()

# ----------------- AST MERGE FUNCTION -----------------
def merge_changes_into_main(main_code, changes_code):
    """
    Merge changes from changes.py into main.py intelligently using AST.
    Functions and classes with the same name are replaced. New ones are added.
    Other code in main.py is preserved.
    """
    try:
        main_tree = ast.parse(main_code)
        changes_tree = ast.parse(changes_code)
    except Exception as e:
        print(f"❌ AST parsing failed: {e}")
        return main_code

    # Map top-level names in main.py
    main_nodes = {node.name: node for node in main_tree.body
                  if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}

    for node in changes_tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            main_nodes[node.name] = node  # Replace or add

    # Reconstruct merged tree
    merged_tree = ast.Module(
        body=list(main_nodes.values()) + 
             [n for n in main_tree.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))],
        type_ignores=[]
    )
    return astor.to_source(merged_tree)

# ----------------- MERGE CHANGES -----------------
main_code = merge_changes_into_main(main_code, changes_code)

with open(main_file, "w") as f:
    f.write(main_code)

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
