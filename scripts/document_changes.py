import os
import subprocess
import datetime
import re
import sys


def run_git(*args):
    """Helper to run git commands and capture output."""
    result = subprocess.run(("git",) + args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Git command failed: git {' '.join(args)}\n{result.stderr}")
    return result.stdout


def get_commit_details(commit_hash):
    """Get commit metadata: short hash, author, date, message."""
    out = run_git("show", "-s", "--format=%h|%an|%ad|%s", commit_hash)
    return tuple(out.strip().split("|", 3))


def get_changed_files(commit_hash):
    """Return a list of (status, path) tuples for a commit."""
    out = run_git("show", "--name-status", "--format=", commit_hash)
    files = []
    for line in out.splitlines():
        if not line.strip():
            continue
        status, path = line.split("\t", 1)
        files.append((status, path))
    return files


def get_file_diff(commit_hash, file_path):
    """Get full unified diff for a single file in a commit."""
    # Use git diff-tree to avoid ambiguous args
    return run_git("diff", f"{commit_hash}~", commit_hash, "--", file_path)


def identify_functions(diff_text, file_path):
    """Extract function/method names from diff by language-specific patterns."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.py':
        pat = r'^\+{0,1}def\s+([A-Za-z0-9_]+)\s*\('  # added or context
    elif ext in ('.js', '.ts'):
        pat = r'^(?:\+{0,1})(?:function\s+([A-Za-z0-9_]+)\s*\(|([A-Za-z0-9_]+)\s*=\s*\([^)]*\)\s*=>)'
    else:
        pat = r'^(?:\+{0,1})(?:[A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\s*\([^)]*\)'
    funcs = set()
    for line in diff_text.splitlines():
        m = re.match(pat, line)
        if m:
            name = m.group(1) or m.group(2)
            if name:
                funcs.add(name)
    return sorted(funcs)


def parse_diff_hunks(diff_text):
    """Extract changed hunk contexts (line ranges)."""
    hunks = []
    for line in diff_text.splitlines():
        if line.startswith('@@'):
            m = re.search(r'@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@', line)
            if m:
                start = int(m.group(1))
                count = int(m.group(2) or '1')
                hunks.append((start, start + count - 1))
    return hunks


def get_recent_changes(limit=10):
    """Gather detailed info for the last `limit` commits."""
    hashes = run_git("log", f"-n{limit}", "--format=%H").splitlines()
    changes = []
    for full_hash in hashes:
        short, author, date, msg = get_commit_details(full_hash)
        files = []
        for status, path in get_changed_files(full_hash):
            diff = get_file_diff(full_hash, path)
            hunks = parse_diff_hunks(diff)
            funcs = identify_functions(diff, path)
            files.append({
                'status': status,
                'path': path,
                'hunks': hunks,
                'functions': funcs,
                'diff': diff.strip().splitlines()
            })
        changes.append({
            'hash': short,
            'full_hash': full_hash,
            'author': author,
            'date': date,
            'message': msg,
            'files': files
        })
    return changes


def generate_markdown(changes):
    """Render the full changes into a Markdown document."""
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    md = [f"# Code Changes - {today}\n"]
    for c in changes:
        md.append(f"## {c['message']} \n")
        md.append(f"**Commit:** `{c['hash']}` by {c['author']} on {c['date']}\n")
        md.append("### Files Changed\n")
        for f in c['files']:
            status = {'M':'Modified','A':'Added','D':'Deleted','R':'Renamed'}.get(f['status'], f['status'])
            md.append(f"- **{status}** `{f['path']}`\n")

            if f['functions']:
                md.append("  - **Affected functions:**\n")
                for fn in f['functions']:
                    md.append(f"    - `{fn}`\n")

            if f['hunks']:
                md.append("  - **Changed ranges:**\n")
                for start, end in f['hunks']:
                    md.append(f"    - Lines {start}-{end}\n")

            md.append("  - **Diff:**\n")
            md.append("  ```diff\n")
            for line in f['diff']:
                md.append(f"{line}" if line.startswith(('+','-','@@')) else f" {line}")
            md.append("```\n")
            md.append("\n")
        md.append("---\n")
        md.append("\n")
    return "\n".join(md)


def main(limit=10):
    repo_dir = "/home/triniborrell/home/projects/TOTEM_for_EEG_code"
    docs_dir = os.path.join(repo_dir, 'docs', 'changes')
    os.makedirs(docs_dir, exist_ok=True)

    changes = get_recent_changes(limit)
    markdown = generate_markdown(changes)

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    output = os.path.join(docs_dir, f"changelog-{today}.md")
    with open(output, 'w') as f:
        f.write(markdown)

    latest = os.path.join(docs_dir, 'latest.md')
    with open(latest, 'w') as f:
        f.write(markdown)

    print(f"Changelog written to {output} and {latest}")

    if not os.environ.get('GIT_HOOK_RUNNING'):
        try:
            run_git('add', output, latest)
            print("Staged changelog files for commit.")
        except RuntimeError as e:
            print(f"Warning: {e}")


if __name__ == '__main__':
    try:
        n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
        main(n)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
