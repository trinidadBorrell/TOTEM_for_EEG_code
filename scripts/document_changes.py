import os
import subprocess
import datetime

def get_recent_changes():
    # Get git log with details of recent changes
    result = subprocess.run(
        ["git", "log", "--name-status", "--pretty=format:%h|%an|%ad|%s", "-n", "10"],
        capture_output=True, text=True
    )
    return result.stdout

def parse_changes(git_log):
    # Parse git log into structured data
    changes = []
    current_commit = None
    
    for line in git_log.split('\n'):
        if '|' in line:  # This is a commit line
            hash, author, date, message = line.split('|', 3)
            current_commit = {
                'hash': hash,
                'author': author,
                'date': date,
                'message': message,
                'files': []
            }
            changes.append(current_commit)
        elif line.strip() and current_commit is not None:
            # This is a file change line
            status, file = line.split('\t', 1) if '\t' in line else (line[0], line[1:].strip())
            current_commit['files'].append({'status': status, 'file': file})
    
    return changes

def generate_markdown(changes):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    
    md = f"# Code Changes - {today}\n\n"
    
    for commit in changes:
        md += f"## {commit['message']}\n"
        md += f"**Commit:** {commit['hash']} by {commit['author']} on {commit['date']}\n\n"
        
        md += "### Files Changed\n"
        for file in commit['files']:
            status_map = {'M': 'Modified', 'A': 'Added', 'D': 'Deleted', 'R': 'Renamed'}
            status = status_map.get(file['status'], file['status'])
            md += f"- **{status}**: `{file['file']}`\n"
        
        md += "\n---\n\n"
    
    return md

def main():
    os.makedirs("/home/triniborrell/home/projects/TOTEM_for_EEG_code/docs/changes", exist_ok=True)
    
    changes = get_recent_changes()
    parsed_changes = parse_changes(changes)
    markdown = generate_markdown(parsed_changes)
    
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    with open(f"/home/triniborrell/home/projects/TOTEM_for_EEG_code/docs/changes/changelog-{today}.md", "w") as f:
        f.write(markdown)
    
    print(f"Generated changelog at docs/changes/changelog-{today}.md")

if __name__ == "__main__":
    main()