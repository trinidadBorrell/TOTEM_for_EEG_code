import os
import subprocess
import datetime
import re
import sys

def get_commit_details(commit_hash):
    """Get detailed information about a specific commit"""
    # Get the commit message and metadata
    result = subprocess.run(
        ["git", "show", "-s", "--format=%h|%an|%ad|%s", commit_hash],
        capture_output=True, text=True
    )
    return result.stdout.strip()

def get_file_diff(commit_hash, file_path):
    """Get the detailed diff for a specific file in a commit"""
    result = subprocess.run(
        ["git", "show", "--format=", "--unified=3", f"{commit_hash}:{file_path}"],
        capture_output=True, text=True
    )
    return result.stdout

def identify_functions(diff_text, file_path):
    """Extract function names and contexts from diff text"""
    functions_affected = []
    
    # Adjust pattern based on file extension for different languages
    ext = os.path.splitext(file_path)[1].lower()
    
    # Define patterns for different languages
    if ext in ['.py']:
        # Python function pattern
        pattern = r'(def\s+([a-zA-Z0-9_]+)\(.*?\):)'
    elif ext in ['.js', '.ts']:
        # JavaScript/TypeScript function pattern
        pattern = r'(function\s+([a-zA-Z0-9_]+)\(.*?\)|([a-zA-Z0-9_]+)\s*=\s*function\(.*?\)|([a-zA-Z0-9_]+)\s*:\s*function\(.*?\))'
    elif ext in ['.java', '.c', '.cpp', '.h', '.hpp']:
        # C/C++/Java pattern
        pattern = r'(([a-zA-Z0-9_]+)\s+([a-zA-Z0-9_]+)\s*\(.*?\)\s*\{)'
    else:
        # Generic fallback
        pattern = r'(([a-zA-Z0-9_]+)\s*\(.*?\))'
    
    # Find all matches
    matches = re.findall(pattern, diff_text, re.DOTALL)
    if matches:
        for match in matches:
            if isinstance(match, tuple):
                func_name = match[1] if len(match) > 1 else match[0]
                functions_affected.append(func_name)
    
    return list(set(functions_affected))  # Remove duplicates

def get_recent_changes():
    """Get detailed information about recent changes"""
    # Get recent commit hashes
    result = subprocess.run(
        ["git", "log", "--format=%H", "-n", "10"],
        capture_output=True, text=True
    )
    commit_hashes = result.stdout.strip().split('\n')
    
    changes = []
    
    for commit_hash in commit_hashes:
        commit_hash = commit_hash.strip()
        if not commit_hash:
            continue
            
        # Get basic commit details
        details = get_commit_details(commit_hash)
        if '|' not in details:
            continue
            
        hash, author, date, message = details.split('|', 3)
        
        # Get list of changed files
        result = subprocess.run(
            ["git", "show", "--name-status", "--format=", commit_hash],
            capture_output=True, text=True
        )
        changed_files = []
        
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
                
            parts = line.split('\t')
            if len(parts) < 2:
                continue
                
            status, file_path = parts[0], parts[1]
            
            # Get detailed diff for this file
            diff_output = subprocess.run(
                ["git", "show", "--format=", "--unified=3", f"{commit_hash} -- {file_path}"],
                capture_output=True, text=True
            )
            diff_text = diff_output.stdout
            
            # Extract line numbers changed
            line_changes = []
            for diff_line in diff_text.split('\n'):
                if diff_line.startswith('@@'):
                    match = re.search(r'@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@', diff_line)
                    if match:
                        line_changes.append(f"Lines around {match.group(2)}")
            
            # Identify affected functions
            affected_functions = identify_functions(diff_text, file_path)
            
            file_info = {
                'status': status,
                'path': file_path,
                'line_changes': line_changes,
                'affected_functions': affected_functions,
                'diff': diff_text.split('\n')[:20]  # First 20 lines of diff for context
            }
            changed_files.append(file_info)
        
        change_info = {
            'hash': hash,
            'author': author,
            'date': date,
            'message': message,
            'files': changed_files
        }
        changes.append(change_info)
    
    return changes

def generate_markdown(changes):
    """Generate detailed markdown documentation from changes"""
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    
    md = f"# Code Changes - {today}\n\n"
    
    for commit in changes:
        md += f"## {commit['message']}\n"
        md += f"**Commit:** {commit['hash']} by {commit['author']} on {commit['date']}\n\n"
        
        md += "### Files Changed\n"
        for file in commit['files']:
            status_map = {'M': 'Modified', 'A': 'Added', 'D': 'Deleted', 'R': 'Renamed'}
            status = status_map.get(file['status'], file['status'])
            md += f"- **{status}**: `{file['path']}`\n"
            
            # Add affected functions
            if file['affected_functions']:
                md += "  - **Affected functions/methods:**\n"
                for func in file['affected_functions']:
                    md += f"    - `{func}`\n"
            
            # Add line changes
            if file['line_changes']:
                md += "  - **Changed sections:**\n"
                for line_change in file['line_changes']:
                    md += f"    - {line_change}\n"
            
            # Add diff snippet
            if 'diff' in file and file['diff']:
                md += "  - **Change preview:**\n"
                md += "    ```diff\n"
                for line in file['diff'][:10]:  # First 10 lines
                    if line.startswith('+'):
                        md += f"    {line}\n"
                    elif line.startswith('-'):
                        md += f"    {line}\n"
                    else:
                        md += f"    {line}\n"
                md += "    ```\n"
        
        md += "\n---\n\n"
    
    return md

def main():
    """Generate changelog and write to file"""
    repo_dir = "/home/triniborrell/home/projects/TOTEM_for_EEG_code"
    os.makedirs(f"{repo_dir}/docs/changes", exist_ok=True)
    
    changes = get_recent_changes()
    markdown = generate_markdown(changes)
    
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    output_path = f"{repo_dir}/docs/changes/changelog-{today}.md"
    with open(output_path, "w") as f:
        f.write(markdown)
    
    print(f"Generated detailed changelog at {output_path}")
    
    # Also update the latest.md file
    latest_path = f"{repo_dir}/docs/changes/latest.md"
    with open(latest_path, "w") as f:
        f.write(markdown)
    
    # Add the generated files to git if this is not run from a hook
    if not os.environ.get('GIT_HOOK_RUNNING'):
        try:
            subprocess.run(["git", "add", output_path, latest_path], check=False)
            print("Added changelog files to git staging")
        except Exception as e:
            print(f"Note: Could not add files to git: {e}")

if __name__ == "__main__":
    main()