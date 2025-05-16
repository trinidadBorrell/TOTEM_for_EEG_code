# Code Changes - 2025-05-16

## works till step 3 with cpu 

**Commit:** `b9643ec` by Trinidad Borrell on Fri May 16 19:03:51 2025 +0200

### Files Changed

- **Modified** `conf/logging/comet_template.yaml`

  - **Changed ranges:**

    - Lines 2-6

  - **Diff:**

  ```diff

 diff --git a/conf/logging/comet_template.yaml b/conf/logging/comet_template.yaml
 index c962617..968f4c3 100644
--- a/conf/logging/comet_template.yaml
+++ b/conf/logging/comet_template.yaml
@@ -2,5 +2,5 @@ comet:
    api_key: "CaCR03FkswI9b7gW1XuubXpbG"
    project_name: "totem_eeg"
    workspace: "trinidadborrell"
-  comet_tag: "first_run"
+  comet_tag: "second_run"
    comet_name: "totem_eeg"
 \ No newline at end of file
```



- **Modified** `scripts/document_changes.py`

  - **Affected functions:**

    - `get_file_diff`

  - **Changed ranges:**

    - Lines 30-60

  - **Diff:**

  ```diff

 diff --git a/scripts/document_changes.py b/scripts/document_changes.py
 index 15f2685..7993650 100644
--- a/scripts/document_changes.py
+++ b/scripts/document_changes.py
@@ -30,13 +30,31 @@ def get_changed_files(commit_hash):
          files.append((status, path))
      return files
  
+"""
  
  def get_file_diff(commit_hash, file_path):
-    """Get full unified diff for a single file in a commit."""
+    Get full unified diff for a single file in a commit.
      # Use git diff-tree to avoid ambiguous args
      return run_git("diff", f"{commit_hash}~", commit_hash, "--", file_path)
  
+"""
  
+def get_file_diff(commit_hash, file_path):
+    """Get full unified diff for a single file in a commit."""
+    try:
+        # Try to get diff between commit and its parent
+        return run_git("diff", f"{commit_hash}~", commit_hash, "--", file_path)
+    except RuntimeError:
+        # For initial commit or other special cases, get the full file content
+        try:
+            # Get the file content at this commit (for added files)
+            content = run_git("show", f"{commit_hash}:{file_path}")
+            # Format it like a diff with all lines added
+            return "\n".join([f"+{line}" for line in content.splitlines()])
+        except RuntimeError:
+            # If that fails too, just return empty diff
+            return ""
+        
  def identify_functions(diff_text, file_path):
      """Extract function/method names from diff by language-specific patterns."""
      ext = os.path.splitext(file_path)[1].lower()
```



- **Modified** `scripts/step2.sh`

  - **Changed ranges:**

    - Lines 7-11

  - **Diff:**

  ```diff

 diff --git a/scripts/step2.sh b/scripts/step2.sh
 index dc4b66e..cc1d785 100644
--- a/scripts/step2.sh
+++ b/scripts/step2.sh
@@ -7,4 +7,5 @@ python -m steps.STEP2_train_vqvae \
      ++exp.save_dir="${repo_dir}/pipeline/step2_train_vqvae/${dataset_name}" \
      ++exp.vqvae_config.dataset=${dataset_name} \
      ++exp.vqvae_config.dataset_base_path="${repo_dir}/pipeline/step1_revin_x_data" \
+    ++exp.vqvae_config.num_epochs=3 \
      +logging=comet 
 \ No newline at end of file
```



- **Modified** `steps/STEP1_save_revin_xdata_for_vqvae.py`

  - **Changed ranges:**

    - Lines 46-53

  - **Diff:**

  ```diff

 diff --git a/steps/STEP1_save_revin_xdata_for_vqvae.py b/steps/STEP1_save_revin_xdata_for_vqvae.py
 index 303c974..a31b603 100644
--- a/steps/STEP1_save_revin_xdata_for_vqvae.py
+++ b/steps/STEP1_save_revin_xdata_for_vqvae.py
@@ -46,6 +46,8 @@ class ExtractData:
      def __init__(self, args):
          self.args = args
          self.device = 'cuda:' + str(self.args.gpu)
+        print('-----------------Using device:', self.device)
+        self.device = torch.device(self.device if torch.cuda.is_available() else 'cpu')
          self.revin_layer_x = RevIN(num_features=self.args.enc_in, affine=False, subtract_last=False)
          self.revin_layer_y = RevIN(num_features=self.args.enc_in, affine=False, subtract_last=False)
```



- **Modified** `steps/STEP3_save_classification_data.py`

  - **Changed ranges:**

    - Lines 8-14

    - Lines 103-117

    - Lines 168-183

  - **Diff:**

  ```diff

 diff --git a/steps/STEP3_save_classification_data.py b/steps/STEP3_save_classification_data.py
 index fad001f..734c94d 100644
--- a/steps/STEP3_save_classification_data.py
+++ b/steps/STEP3_save_classification_data.py
@@ -8,6 +8,7 @@ import torch.nn as nn
  
  from data_provider.data_factory_vqvae_no_shuffle import data_provider_flexPath
  from lib.models.revin import RevIN
+from lib.models.vqvae import vqvae
  
  from omegaconf import DictConfig, OmegaConf
  import hydra
@@ -102,10 +103,15 @@ def main(cfg: DictConfig) -> None:
  class ExtractData:
      def __init__(self, args):
          self.args = args
-        self.device = 'cuda:' + str(self.args.gpu)
+        if not args.use_gpu or not torch.cuda.is_available():
+            self.device = 'cpu'
+        else:
+            self.device = 'cuda:' + str(self.args.gpu)
+    
+        print(f"Using device: {self.device}")
          self.revin_layer_x = RevIN(num_features=self.args.enc_in, affine=False, subtract_last=False)
-        self.revin_layer_y = RevIN(num_features=self.args.enc_in, affine=False, subtract_last=False)
-
+        self.revin_layer_y = RevIN(num_features=self.args.enc_in, affine=False, subtract_last=False)       
+        
      # def _get_data(self, flag):
      #     data_set, data_loader = data_provider(self.args, flag)
      #     return data_set, data_loader
@@ -162,9 +168,16 @@ class ExtractData:
          return data_dict
  
      def extract_data(self, root_path_name, data_path_name, save_data=True):
+        """
          vqvae_model = torch.load(self.args.trained_vqvae_model_path)
          vqvae_model.to(self.device)
          vqvae_model.eval()
+        """
+    #    torch.serialization.add_safe_globals([vqvae])
+    
+        vqvae_model = torch.load(self.args.trained_vqvae_model_path, weights_only=False)
+        vqvae_model.to(self.device)
+        vqvae_model.eval()
  
          _, train_loader = self._get_data(root_path_name, data_path_name, flag='train')
          _, vali_loader = self._get_data(root_path_name, data_path_name, flag='val')
```



---



## check documentation working 

**Commit:** `4914531` by Trinidad Borrell on Fri May 16 11:37:51 2025 +0200

### Files Changed

- **Modified** `scripts/document_changes.py`

  - **Changed ranges:**

    - Lines 132-138

  - **Diff:**

  ```diff

 diff --git a/scripts/document_changes.py b/scripts/document_changes.py
 index d74888c..15f2685 100644
--- a/scripts/document_changes.py
+++ b/scripts/document_changes.py
@@ -132,7 +132,7 @@ def generate_markdown(changes):
  
  
  def main(limit=10):
-    repo_dir = os.path.abspath(os.getcwd())
+    repo_dir = "/home/triniborrell/home/projects/TOTEM_for_EEG_code"
      docs_dir = os.path.join(repo_dir, 'docs', 'changes')
      os.makedirs(docs_dir, exist_ok=True)
```



---



## modify documentation 

**Commit:** `3261c4f` by Trinidad Borrell on Fri May 16 11:25:49 2025 +0200

### Files Changed

- **Modified** `scripts/document_changes.py`

  - **Affected functions:**

    - `get_changed_files`

    - `get_recent_changes`

    - `main`

    - `parse_diff_hunks`

    - `run_git`

  - **Changed ranges:**

    - Lines 4-167

  - **Diff:**

  ```diff

 diff --git a/scripts/document_changes.py b/scripts/document_changes.py
 index 004c791..d74888c 100644
--- a/scripts/document_changes.py
+++ b/scripts/document_changes.py
@@ -4,204 +4,164 @@ import datetime
  import re
  import sys
  
+
+def run_git(*args):
+    """Helper to run git commands and capture output."""
+    result = subprocess.run(("git",) + args, capture_output=True, text=True)
+    if result.returncode != 0:
+        raise RuntimeError(f"Git command failed: git {' '.join(args)}\n{result.stderr}")
+    return result.stdout
+
+
  def get_commit_details(commit_hash):
-    """Get detailed information about a specific commit"""
-    # Get the commit message and metadata
-    result = subprocess.run(
-        ["git", "show", "-s", "--format=%h|%an|%ad|%s", commit_hash],
-        capture_output=True, text=True
-    )
-    return result.stdout.strip()
+    """Get commit metadata: short hash, author, date, message."""
+    out = run_git("show", "-s", "--format=%h|%an|%ad|%s", commit_hash)
+    return tuple(out.strip().split("|", 3))
+
+
+def get_changed_files(commit_hash):
+    """Return a list of (status, path) tuples for a commit."""
+    out = run_git("show", "--name-status", "--format=", commit_hash)
+    files = []
+    for line in out.splitlines():
+        if not line.strip():
+            continue
+        status, path = line.split("\t", 1)
+        files.append((status, path))
+    return files
+
  
  def get_file_diff(commit_hash, file_path):
-    """Get the detailed diff for a specific file in a commit"""
-    result = subprocess.run(
-        ["git", "show", "--format=", "--unified=3", f"{commit_hash}:{file_path}"],
-        capture_output=True, text=True
-    )
-    return result.stdout
+    """Get full unified diff for a single file in a commit."""
+    # Use git diff-tree to avoid ambiguous args
+    return run_git("diff", f"{commit_hash}~", commit_hash, "--", file_path)
+
  
  def identify_functions(diff_text, file_path):
-    """Extract function names and contexts from diff text"""
-    functions_affected = []
-    
-    # Adjust pattern based on file extension for different languages
+    """Extract function/method names from diff by language-specific patterns."""
      ext = os.path.splitext(file_path)[1].lower()
-    
-    # Define patterns for different languages
-    if ext in ['.py']:
-        # Python function pattern
-        pattern = r'(def\s+([a-zA-Z0-9_]+)\(.*?\):)'
-    elif ext in ['.js', '.ts']:
-        # JavaScript/TypeScript function pattern
-        pattern = r'(function\s+([a-zA-Z0-9_]+)\(.*?\)|([a-zA-Z0-9_]+)\s*=\s*function\(.*?\)|([a-zA-Z0-9_]+)\s*:\s*function\(.*?\))'
-    elif ext in ['.java', '.c', '.cpp', '.h', '.hpp']:
-        # C/C++/Java pattern
-        pattern = r'(([a-zA-Z0-9_]+)\s+([a-zA-Z0-9_]+)\s*\(.*?\)\s*\{)'
+    if ext == '.py':
+        pat = r'^\+{0,1}def\s+([A-Za-z0-9_]+)\s*\('  # added or context
+    elif ext in ('.js', '.ts'):
+        pat = r'^(?:\+{0,1})(?:function\s+([A-Za-z0-9_]+)\s*\(|([A-Za-z0-9_]+)\s*=\s*\([^)]*\)\s*=>)'
      else:
-        # Generic fallback
-        pattern = r'(([a-zA-Z0-9_]+)\s*\(.*?\))'
-    
-    # Find all matches
-    matches = re.findall(pattern, diff_text, re.DOTALL)
-    if matches:
-        for match in matches:
-            if isinstance(match, tuple):
-                func_name = match[1] if len(match) > 1 else match[0]
-                functions_affected.append(func_name)
-    
-    return list(set(functions_affected))  # Remove duplicates
-
-def get_recent_changes():
-    """Get detailed information about recent changes"""
-    # Get recent commit hashes
-    result = subprocess.run(
-        ["git", "log", "--format=%H", "-n", "10"],
-        capture_output=True, text=True
-    )
-    commit_hashes = result.stdout.strip().split('\n')
-    
+        pat = r'^(?:\+{0,1})(?:[A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\s*\([^)]*\)'
+    funcs = set()
+    for line in diff_text.splitlines():
+        m = re.match(pat, line)
+        if m:
+            name = m.group(1) or m.group(2)
+            if name:
+                funcs.add(name)
+    return sorted(funcs)
+
+
+def parse_diff_hunks(diff_text):
+    """Extract changed hunk contexts (line ranges)."""
+    hunks = []
+    for line in diff_text.splitlines():
+        if line.startswith('@@'):
+            m = re.search(r'@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@', line)
+            if m:
+                start = int(m.group(1))
+                count = int(m.group(2) or '1')
+                hunks.append((start, start + count - 1))
+    return hunks
+
+
+def get_recent_changes(limit=10):
+    """Gather detailed info for the last `limit` commits."""
+    hashes = run_git("log", f"-n{limit}", "--format=%H").splitlines()
      changes = []
-    
-    for commit_hash in commit_hashes:
-        commit_hash = commit_hash.strip()
-        if not commit_hash:
-            continue
-            
-        # Get basic commit details
-        details = get_commit_details(commit_hash)
-        if '|' not in details:
-            continue
-            
-        hash, author, date, message = details.split('|', 3)
-        
-        # Get list of changed files
-        result = subprocess.run(
-            ["git", "show", "--name-status", "--format=", commit_hash],
-            capture_output=True, text=True
-        )
-        changed_files = []
-        
-        for line in result.stdout.strip().split('\n'):
-            if not line.strip():
-                continue
-                
-            parts = line.split('\t')
-            if len(parts) < 2:
-                continue
-                
-            status, file_path = parts[0], parts[1]
-            
-            # Get detailed diff for this file
-            diff_output = subprocess.run(
-                ["git", "show", "--format=", "--unified=3", f"{commit_hash} -- {file_path}"],
-                capture_output=True, text=True
-            )
-            diff_text = diff_output.stdout
-            
-            # Extract line numbers changed
-            line_changes = []
-            for diff_line in diff_text.split('\n'):
-                if diff_line.startswith('@@'):
-                    match = re.search(r'@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@', diff_line)
-                    if match:
-                        line_changes.append(f"Lines around {match.group(2)}")
-            
-            # Identify affected functions
-            affected_functions = identify_functions(diff_text, file_path)
-            
-            file_info = {
+    for full_hash in hashes:
+        short, author, date, msg = get_commit_details(full_hash)
+        files = []
+        for status, path in get_changed_files(full_hash):
+            diff = get_file_diff(full_hash, path)
+            hunks = parse_diff_hunks(diff)
+            funcs = identify_functions(diff, path)
+            files.append({
                  'status': status,
-                'path': file_path,
-                'line_changes': line_changes,
-                'affected_functions': affected_functions,
-                'diff': diff_text.split('\n')[:20]  # First 20 lines of diff for context
-            }
-            changed_files.append(file_info)
-        
-        change_info = {
-            'hash': hash,
+                'path': path,
+                'hunks': hunks,
+                'functions': funcs,
+                'diff': diff.strip().splitlines()
+            })
+        changes.append({
+            'hash': short,
+            'full_hash': full_hash,
              'author': author,
              'date': date,
-            'message': message,
-            'files': changed_files
-        }
-        changes.append(change_info)
-    
+            'message': msg,
+            'files': files
+        })
      return changes
  
+
  def generate_markdown(changes):
-    """Generate detailed markdown documentation from changes"""
+    """Render the full changes into a Markdown document."""
      today = datetime.datetime.now().strftime("%Y-%m-%d")
-    
-    md = f"# Code Changes - {today}\n\n"
-    
-    for commit in changes:
-        md += f"## {commit['message']}\n"
-        md += f"**Commit:** {commit['hash']} by {commit['author']} on {commit['date']}\n\n"
-        
-        md += "### Files Changed\n"
-        for file in commit['files']:
-            status_map = {'M': 'Modified', 'A': 'Added', 'D': 'Deleted', 'R': 'Renamed'}
-            status = status_map.get(file['status'], file['status'])
-            md += f"- **{status}**: `{file['path']}`\n"
-            
-            # Add affected functions
-            if file['affected_functions']:
-                md += "  - **Affected functions/methods:**\n"
-                for func in file['affected_functions']:
-                    md += f"    - `{func}`\n"
-            
-            # Add line changes
-            if file['line_changes']:
-                md += "  - **Changed sections:**\n"
-                for line_change in file['line_changes']:
-                    md += f"    - {line_change}\n"
-            
-            # Add diff snippet
-            if 'diff' in file and file['diff']:
-                md += "  - **Change preview:**\n"
-                md += "    ```diff\n"
-                for line in file['diff'][:10]:  # First 10 lines
-                    if line.startswith('+'):
-                        md += f"    {line}\n"
-                    elif line.startswith('-'):
-                        md += f"    {line}\n"
-                    else:
-                        md += f"    {line}\n"
-                md += "    ```\n"
-        
-        md += "\n---\n\n"
-    
-    return md
-
-def main():
-    """Generate changelog and write to file"""
-    repo_dir = "/home/triniborrell/home/projects/TOTEM_for_EEG_code"
-    os.makedirs(f"{repo_dir}/docs/changes", exist_ok=True)
-    
-    changes = get_recent_changes()
+    md = [f"# Code Changes - {today}\n"]
+    for c in changes:
+        md.append(f"## {c['message']} \n")
+        md.append(f"**Commit:** `{c['hash']}` by {c['author']} on {c['date']}\n")
+        md.append("### Files Changed\n")
+        for f in c['files']:
+            status = {'M':'Modified','A':'Added','D':'Deleted','R':'Renamed'}.get(f['status'], f['status'])
+            md.append(f"- **{status}** `{f['path']}`\n")
+
+            if f['functions']:
+                md.append("  - **Affected functions:**\n")
+                for fn in f['functions']:
+                    md.append(f"    - `{fn}`\n")
+
+            if f['hunks']:
+                md.append("  - **Changed ranges:**\n")
+                for start, end in f['hunks']:
+                    md.append(f"    - Lines {start}-{end}\n")
+
+            md.append("  - **Diff:**\n")
+            md.append("  ```diff\n")
+            for line in f['diff']:
+                md.append(f"{line}" if line.startswith(('+','-','@@')) else f" {line}")
+            md.append("```\n")
+            md.append("\n")
+        md.append("---\n")
+        md.append("\n")
+    return "\n".join(md)
+
+
+def main(limit=10):
+    repo_dir = os.path.abspath(os.getcwd())
+    docs_dir = os.path.join(repo_dir, 'docs', 'changes')
+    os.makedirs(docs_dir, exist_ok=True)
+
+    changes = get_recent_changes(limit)
      markdown = generate_markdown(changes)
-    
+
      today = datetime.datetime.now().strftime("%Y-%m-%d")
-    output_path = f"{repo_dir}/docs/changes/changelog-{today}.md"
-    with open(output_path, "w") as f:
+    output = os.path.join(docs_dir, f"changelog-{today}.md")
+    with open(output, 'w') as f:
          f.write(markdown)
-    
-    print(f"Generated detailed changelog at {output_path}")
-    
-    # Also update the latest.md file
-    latest_path = f"{repo_dir}/docs/changes/latest.md"
-    with open(latest_path, "w") as f:
+
+    latest = os.path.join(docs_dir, 'latest.md')
+    with open(latest, 'w') as f:
          f.write(markdown)
-    
-    # Add the generated files to git if this is not run from a hook
+
+    print(f"Changelog written to {output} and {latest}")
+
      if not os.environ.get('GIT_HOOK_RUNNING'):
          try:
-            subprocess.run(["git", "add", output_path, latest_path], check=False)
-            print("Added changelog files to git staging")
-        except Exception as e:
-            print(f"Note: Could not add files to git: {e}")
+            run_git('add', output, latest)
+            print("Staged changelog files for commit.")
+        except RuntimeError as e:
+            print(f"Warning: {e}")
+
  
-if __name__ == "__main__":
-    main()
 \ No newline at end of file
+if __name__ == '__main__':
+    try:
+        n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
+        main(n)
+    except Exception as e:
+        print(f"Error: {e}")
+        sys.exit(1)
```



---



## auto documentation and update data_loader 

**Commit:** `a0d9e5b` by Trinidad Borrell on Fri May 16 11:12:36 2025 +0200

### Files Changed

- **Modified** `data_provider/data_loader.py`

  - **Changed ranges:**

    - Lines 650-655

    - Lines 677-692

    - Lines 713-776

  - **Diff:**

  ```diff

 diff --git a/data_provider/data_loader.py b/data_provider/data_loader.py
 index 49d9798..e1971a1 100644
--- a/data_provider/data_loader.py
+++ b/data_provider/data_loader.py
@@ -650,8 +650,6 @@ class Dataset_EEG(Dataset):
              # Update self.eeg_columns with the filtered columns
              self.eeg_columns = filtered_columns
  
-
-
          
          if self.scale:
              self.scaler.fit(self.make_contiguous_x_data(df_raw, df_split, split='train')) 
@@ -679,10 +677,16 @@ class Dataset_EEG(Dataset):
              self.data_x_label = test_x_label
              self.data_y_label = test_y_label
  
+    """
+    Instead of considering each trial ends with a Rest event, we consider the last trial to end with the last event
+    (which is not Rest). 
+    Redo functions make_contiguous_x_data and make_full_x_y_data to consider this.
+    
+    
+
      def make_contiguous_x_data(self, df_raw, df_split, split):         
          data_x = []
          for trial_start_ind, r in df_split[df_split['split'] == split].iterrows():
-            print(df_raw.loc[trial_start_ind:][df_raw.loc[trial_start_ind:, 'STI'] == self.event_dict['Rest']].iloc[0].name)
              trial_end_ind = df_raw.loc[trial_start_ind:][df_raw.loc[trial_start_ind:, 'STI'] == self.event_dict['Rest']].iloc[0].name
              data_x.append(df_raw.loc[trial_start_ind:trial_end_ind, self.eeg_columns].values)
          return np.concatenate(data_x)
@@ -709,6 +713,64 @@ class Dataset_EEG(Dataset):
                  else:
                      break
          return data_x, data_y, data_x_label, data_y_label
+    """
+    def make_contiguous_x_data(self, df_raw, df_split, split):
+        data_x = []
+        # Get all trial start indices for this split
+        trial_starts = df_split[df_split['split'] == split].index.tolist()
+    
+        # Sort them to ensure they're in ascending order
+        trial_starts.sort()
+    
+        for i, trial_start_ind in enumerate(trial_starts):
+            # If this is not the last trial, use the next trial start as the end
+            if i < len(trial_starts) - 1:
+                trial_end_ind = trial_starts[i+1] - 1  # End just before next trial
+            else:
+                # For the last trial, go to the end of the dataset
+                trial_end_ind = df_raw.index[-1]
+            
+            data_x.append(df_raw.loc[trial_start_ind:trial_end_ind, self.eeg_columns].values)
+        
+        if not data_x:  # If no data was collected, return empty array with correct columns
+            return np.zeros((0, len(self.eeg_columns)))
+        return np.concatenate(data_x)
+
+    def make_full_x_y_data(self, df_raw, df_split, split):
+        data_x = []
+        data_y = []
+        data_x_label = []
+        data_y_label = []
+    
+        # Get all trial start indices for this split
+        trial_starts = df_split[df_split['split'] == split].index.tolist()
+        trial_starts.sort()
+    
+        for i, trial_start_ind in enumerate(trial_starts):
+            r = df_split.loc[trial_start_ind]
+        
+            # If this is not the last trial, use the next trial start as the end
+            if i < len(trial_starts) - 1:
+                trial_end_ind = trial_starts[i+1] - 1
+            else:
+                # For the last trial, go to the end of the dataset
+                trial_end_ind = df_raw.index[-1]
+            
+            for time in range(trial_start_ind, trial_end_ind, self.step_size):
+                s_begin = time
+                s_end = s_begin + self.seq_len
+                r_begin = s_end - self.label_len
+                r_end = r_begin + self.label_len + self.pred_len
+            
+                if r_end <= trial_end_ind:
+                    data_x.append(df_raw.loc[s_begin:s_end-1, self.eeg_columns].values)
+                    data_y.append(df_raw.loc[r_begin:r_end-1, self.eeg_columns].values)
+                    data_x_label.append(int(r['STI']))  # Keep original STI value (no -1)
+                    data_y_label.append(int(r['STI']))  # Keep original STI value
+                else:
+                    break
+                
+        return data_x, data_y, data_x_label, data_y_label
      
      def __getitem__(self, index):
          return self.data_x[index], self.data_y[index], self.data_x_label[index], self.data_y_label[index]
```



- **Modified** `scripts/document_changes.py`

  - **Affected functions:**

    - `get_commit_details`

    - `get_file_diff`

    - `get_recent_changes`

    - `identify_functions`

  - **Changed ranges:**

    - Lines 1-137

    - Lines 144-207

  - **Diff:**

  ```diff

 diff --git a/scripts/document_changes.py b/scripts/document_changes.py
 index e3de409..004c791 100644
--- a/scripts/document_changes.py
+++ b/scripts/document_changes.py
@@ -1,39 +1,137 @@
  import os
  import subprocess
  import datetime
+import re
+import sys
  
-def get_recent_changes():
-    # Get git log with details of recent changes
+def get_commit_details(commit_hash):
+    """Get detailed information about a specific commit"""
+    # Get the commit message and metadata
+    result = subprocess.run(
+        ["git", "show", "-s", "--format=%h|%an|%ad|%s", commit_hash],
+        capture_output=True, text=True
+    )
+    return result.stdout.strip()
+
+def get_file_diff(commit_hash, file_path):
+    """Get the detailed diff for a specific file in a commit"""
      result = subprocess.run(
-        ["git", "log", "--name-status", "--pretty=format:%h|%an|%ad|%s", "-n", "10"],
+        ["git", "show", "--format=", "--unified=3", f"{commit_hash}:{file_path}"],
          capture_output=True, text=True
      )
      return result.stdout
  
-def parse_changes(git_log):
-    # Parse git log into structured data
+def identify_functions(diff_text, file_path):
+    """Extract function names and contexts from diff text"""
+    functions_affected = []
+    
+    # Adjust pattern based on file extension for different languages
+    ext = os.path.splitext(file_path)[1].lower()
+    
+    # Define patterns for different languages
+    if ext in ['.py']:
+        # Python function pattern
+        pattern = r'(def\s+([a-zA-Z0-9_]+)\(.*?\):)'
+    elif ext in ['.js', '.ts']:
+        # JavaScript/TypeScript function pattern
+        pattern = r'(function\s+([a-zA-Z0-9_]+)\(.*?\)|([a-zA-Z0-9_]+)\s*=\s*function\(.*?\)|([a-zA-Z0-9_]+)\s*:\s*function\(.*?\))'
+    elif ext in ['.java', '.c', '.cpp', '.h', '.hpp']:
+        # C/C++/Java pattern
+        pattern = r'(([a-zA-Z0-9_]+)\s+([a-zA-Z0-9_]+)\s*\(.*?\)\s*\{)'
+    else:
+        # Generic fallback
+        pattern = r'(([a-zA-Z0-9_]+)\s*\(.*?\))'
+    
+    # Find all matches
+    matches = re.findall(pattern, diff_text, re.DOTALL)
+    if matches:
+        for match in matches:
+            if isinstance(match, tuple):
+                func_name = match[1] if len(match) > 1 else match[0]
+                functions_affected.append(func_name)
+    
+    return list(set(functions_affected))  # Remove duplicates
+
+def get_recent_changes():
+    """Get detailed information about recent changes"""
+    # Get recent commit hashes
+    result = subprocess.run(
+        ["git", "log", "--format=%H", "-n", "10"],
+        capture_output=True, text=True
+    )
+    commit_hashes = result.stdout.strip().split('\n')
+    
      changes = []
-    current_commit = None
      
-    for line in git_log.split('\n'):
-        if '|' in line:  # This is a commit line
-            hash, author, date, message = line.split('|', 3)
-            current_commit = {
-                'hash': hash,
-                'author': author,
-                'date': date,
-                'message': message,
-                'files': []
+    for commit_hash in commit_hashes:
+        commit_hash = commit_hash.strip()
+        if not commit_hash:
+            continue
+            
+        # Get basic commit details
+        details = get_commit_details(commit_hash)
+        if '|' not in details:
+            continue
+            
+        hash, author, date, message = details.split('|', 3)
+        
+        # Get list of changed files
+        result = subprocess.run(
+            ["git", "show", "--name-status", "--format=", commit_hash],
+            capture_output=True, text=True
+        )
+        changed_files = []
+        
+        for line in result.stdout.strip().split('\n'):
+            if not line.strip():
+                continue
+                
+            parts = line.split('\t')
+            if len(parts) < 2:
+                continue
+                
+            status, file_path = parts[0], parts[1]
+            
+            # Get detailed diff for this file
+            diff_output = subprocess.run(
+                ["git", "show", "--format=", "--unified=3", f"{commit_hash} -- {file_path}"],
+                capture_output=True, text=True
+            )
+            diff_text = diff_output.stdout
+            
+            # Extract line numbers changed
+            line_changes = []
+            for diff_line in diff_text.split('\n'):
+                if diff_line.startswith('@@'):
+                    match = re.search(r'@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@', diff_line)
+                    if match:
+                        line_changes.append(f"Lines around {match.group(2)}")
+            
+            # Identify affected functions
+            affected_functions = identify_functions(diff_text, file_path)
+            
+            file_info = {
+                'status': status,
+                'path': file_path,
+                'line_changes': line_changes,
+                'affected_functions': affected_functions,
+                'diff': diff_text.split('\n')[:20]  # First 20 lines of diff for context
              }
-            changes.append(current_commit)
-        elif line.strip() and current_commit is not None:
-            # This is a file change line
-            status, file = line.split('\t', 1) if '\t' in line else (line[0], line[1:].strip())
-            current_commit['files'].append({'status': status, 'file': file})
+            changed_files.append(file_info)
+        
+        change_info = {
+            'hash': hash,
+            'author': author,
+            'date': date,
+            'message': message,
+            'files': changed_files
+        }
+        changes.append(change_info)
      
      return changes
  
  def generate_markdown(changes):
+    """Generate detailed markdown documentation from changes"""
      today = datetime.datetime.now().strftime("%Y-%m-%d")
      
      md = f"# Code Changes - {today}\n\n"
@@ -46,24 +144,64 @@ def generate_markdown(changes):
          for file in commit['files']:
              status_map = {'M': 'Modified', 'A': 'Added', 'D': 'Deleted', 'R': 'Renamed'}
              status = status_map.get(file['status'], file['status'])
-            md += f"- **{status}**: `{file['file']}`\n"
+            md += f"- **{status}**: `{file['path']}`\n"
+            
+            # Add affected functions
+            if file['affected_functions']:
+                md += "  - **Affected functions/methods:**\n"
+                for func in file['affected_functions']:
+                    md += f"    - `{func}`\n"
+            
+            # Add line changes
+            if file['line_changes']:
+                md += "  - **Changed sections:**\n"
+                for line_change in file['line_changes']:
+                    md += f"    - {line_change}\n"
+            
+            # Add diff snippet
+            if 'diff' in file and file['diff']:
+                md += "  - **Change preview:**\n"
+                md += "    ```diff\n"
+                for line in file['diff'][:10]:  # First 10 lines
+                    if line.startswith('+'):
+                        md += f"    {line}\n"
+                    elif line.startswith('-'):
+                        md += f"    {line}\n"
+                    else:
+                        md += f"    {line}\n"
+                md += "    ```\n"
          
          md += "\n---\n\n"
      
      return md
  
  def main():
-    os.makedirs("/home/triniborrell/home/projects/TOTEM_for_EEG_code/docs/changes", exist_ok=True)
+    """Generate changelog and write to file"""
+    repo_dir = "/home/triniborrell/home/projects/TOTEM_for_EEG_code"
+    os.makedirs(f"{repo_dir}/docs/changes", exist_ok=True)
      
      changes = get_recent_changes()
-    parsed_changes = parse_changes(changes)
-    markdown = generate_markdown(parsed_changes)
+    markdown = generate_markdown(changes)
      
      today = datetime.datetime.now().strftime("%Y-%m-%d")
-    with open(f"/home/triniborrell/home/projects/TOTEM_for_EEG_code/docs/changes/changelog-{today}.md", "w") as f:
+    output_path = f"{repo_dir}/docs/changes/changelog-{today}.md"
+    with open(output_path, "w") as f:
+        f.write(markdown)
+    
+    print(f"Generated detailed changelog at {output_path}")
+    
+    # Also update the latest.md file
+    latest_path = f"{repo_dir}/docs/changes/latest.md"
+    with open(latest_path, "w") as f:
          f.write(markdown)
      
-    print(f"Generated changelog at docs/changes/changelog-{today}.md")
+    # Add the generated files to git if this is not run from a hook
+    if not os.environ.get('GIT_HOOK_RUNNING'):
+        try:
+            subprocess.run(["git", "add", output_path, latest_path], check=False)
+            print("Added changelog files to git staging")
+        except Exception as e:
+            print(f"Note: Could not add files to git: {e}")
  
  if __name__ == "__main__":
      main()
 \ No newline at end of file
```



- **Added** `scripts/setup_git_hook.sh`

  - **Changed ranges:**

    - Lines 1-23

  - **Diff:**

  ```diff

 diff --git a/scripts/setup_git_hook.sh b/scripts/setup_git_hook.sh
 new file mode 100755
 index 0000000..64880ea
--- /dev/null
+++ b/scripts/setup_git_hook.sh
@@ -0,0 +1,23 @@
+#!/bin/bash
+
+REPO_DIR="/home/triniborrell/home/projects/TOTEM_for_EEG_code"
+HOOK_DIR="$REPO_DIR/.git/hooks"
+HOOK_PATH="$HOOK_DIR/post-commit"
+
+# Create the post-commit hook
+cat > "$HOOK_PATH" << 'EOF'
+#!/bin/bash
+# Post-commit hook to update documentation
+
+export GIT_HOOK_RUNNING=true
+python /home/triniborrell/home/projects/TOTEM_for_EEG_code/scripts/document_changes.py
+
+# Don't commit the generated files in this hook to avoid infinite loop
+# They'll be committed in next commit
+EOF
+
+# Make the hook executable
+chmod +x "$HOOK_PATH"
+
+echo "Git post-commit hook installed successfully!"
+echo "Documentation will be automatically generated after each commit."
 \ No newline at end of file
```



---



## first iteration with UCI small database (UCI - sd) 

**Commit:** `8a729ee` by Trinidad Borrell on Fri May 16 10:41:22 2025 +0200

### Files Changed

- **Modified** `conf/exp/step2_train_vqvae.yaml`

  - **Changed ranges:**

    - Lines 1-4

    - Lines 17-22

  - **Diff:**

  ```diff

 diff --git a/conf/exp/step2_train_vqvae.yaml b/conf/exp/step2_train_vqvae.yaml
 index ca5ddc1..d3c0031 100644
--- a/conf/exp/step2_train_vqvae.yaml
+++ b/conf/exp/step2_train_vqvae.yaml
@@ -1,4 +1,4 @@
-save_dir: ???
+save_dir: "/home/triniborrell/home/projects/TOTEM_for_EEG_code/save/"
  gpu_id: 0
  data_init_cpu_or_gpu: "cpu"
  vqvae_config: 
@@ -17,5 +17,6 @@ vqvae_config:
    num_embeddings: 256
    commitment_cost: 0.25
    compression_factor: 4
-  dataset: ??? 
-  dataset_base_path: ??? 
 \ No newline at end of file
+  dataset: "a_dataset.csv" 
+  dataset_base_path: "/home/triniborrell/home/projects/TOTEM_for_EEG_code/data/"
+  
 \ No newline at end of file
```



- **Modified** `conf/exp/step4_train_xformer.yaml`

  - **Changed ranges:**

    - Lines 3-15

  - **Diff:**

  ```diff

 diff --git a/conf/exp/step4_train_xformer.yaml b/conf/exp/step4_train_xformer.yaml
 index 4d7fa91..d02e2c3 100644
--- a/conf/exp/step4_train_xformer.yaml
+++ b/conf/exp/step4_train_xformer.yaml
@@ -3,13 +3,13 @@ classifier_config:
    seed: 47
    model_type: "xformer"
    data_type: "eeg"
-  data_path: ???
+  data_path: "/home/triniborrell/home/projects/TOTEM_for_EEG_code/data/"
    codebook_size: 256
    compression: 4
    Tin: 512
    checkpoint: true 
-  checkpoint_path: ??? 
-  exp_name: ??? 
+  checkpoint_path: "/home/triniborrell/home/projects/TOTEM_for_EEG_code/pipeline/step4_train_xformer/" 
+  exp_name: "first_run" 
    optimizer: "adam" 
    scheduler: "step"
    baselr: 0.0001
```



- **Modified** `conf/logging/comet_template.yaml`

  - **Changed ranges:**

    - Lines 1-6

  - **Diff:**

  ```diff

 diff --git a/conf/logging/comet_template.yaml b/conf/logging/comet_template.yaml
 index de7fc1c..c962617 100644
--- a/conf/logging/comet_template.yaml
+++ b/conf/logging/comet_template.yaml
@@ -1,6 +1,6 @@
  comet:
-  api_key: ???
-  project_name: ???
-  workspace: ???
-  comet_tag: ???
-  comet_name: ???
 \ No newline at end of file
+  api_key: "CaCR03FkswI9b7gW1XuubXpbG"
+  project_name: "totem_eeg"
+  workspace: "trinidadborrell"
+  comet_tag: "first_run"
+  comet_name: "totem_eeg"
 \ No newline at end of file
```



- **Modified** `conf/preprocessing/step1_eeg.yaml`

  - **Changed ranges:**

    - Lines 1-7

    - Lines 10-16

  - **Diff:**

  ```diff

 diff --git a/conf/preprocessing/step1_eeg.yaml b/conf/preprocessing/step1_eeg.yaml
 index 1289c64..f775680 100644
--- a/conf/preprocessing/step1_eeg.yaml
+++ b/conf/preprocessing/step1_eeg.yaml
@@ -1,7 +1,7 @@
  random_seed: 2021
  data: "eeg" ## e.g. "eeg"
-root_paths: ??? ## e.g. ["./data/ETT/"]
-data_paths: ??? ## e.g. ["ETTh1.csv"]
+root_paths: "/home/triniborrell/home/projects/TOTEM_for_EEG_code/data/" ## e.g. ["./data/ETT/"]
+data_paths: "a_dataset.csv" ## e.g. ["ETTh1.csv"]
  seq_len: 96 ## input sequence length
  label_len: 0 ## start token length
  pred_len: 96 ## prediction sequence length
@@ -10,7 +10,7 @@ use_gpu: True
  gpu: 0
  use_multi_gpu: False
  devices: '0,1,2,3'
-save_path: ???
+save_path: "/home/triniborrell/home/projects/TOTEM_for_EEG_code/save/"
  num_workers: 10 # data loader num workers
  batch_size: 128 # batch size of train input data
  features: "M" # forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
```



- **Modified** `conf/preprocessing/step3_eeg.yaml`

  - **Changed ranges:**

    - Lines 1-8

    - Lines 16-21

  - **Diff:**

  ```diff

 diff --git a/conf/preprocessing/step3_eeg.yaml b/conf/preprocessing/step3_eeg.yaml
 index 9631be2..c13c929 100644
--- a/conf/preprocessing/step3_eeg.yaml
+++ b/conf/preprocessing/step3_eeg.yaml
@@ -1,8 +1,8 @@
  random_seed: 2021
  data: eeg
-train_root_paths: ???
+train_root_paths: "/home/triniborrell/home/projects/TOTEM_for_EEG_code/data/test/"
  train_data_paths: ???
-test_root_paths: ???
+test_root_paths: "/home/triniborrell/home/projects/TOTEM_for_EEG_code/data/test/"
  test_data_paths: ???
  features: "M" # options: [clean or other] if "clean" will clean the data using the specified bad_channels.csv in corresponding data_path.
  target: "half" # target feature in S or MS task (overriden as step_size in EEG dataset)
@@ -16,6 +16,6 @@ num_workers: 10
  batch_size: 128
  use_gpu: True 
  gpu: 0
-save_path: ???
+save_path: "/home/triniborrell/home/projects/TOTEM_for_EEG_code/save/"
  trained_vqvae_model_path: ???
  compression_factor: 4
```



- **Modified** `data_provider/data_loader.py`

  - **Changed ranges:**

    - Lines 594-599

    - Lines 619-627

    - Lines 682-688

  - **Diff:**

  ```diff

 diff --git a/data_provider/data_loader.py b/data_provider/data_loader.py
 index 933135b..49d9798 100644
--- a/data_provider/data_loader.py
+++ b/data_provider/data_loader.py
@@ -594,8 +594,6 @@ class Dataset_EEG(Dataset):
      def __init__(self, root_path, flag='train', size=None,
                   features=None, data_path='ETTh1.csv',
                   target='OT', scale=True, timeenc=0, freq='h'):
-        # size [seq_len, label_len, pred_len]
-        # info
          if size == None:
              self.seq_len = 24 * 4 * 4
              self.label_len = 24 * 4
@@ -621,7 +619,9 @@ class Dataset_EEG(Dataset):
          self.scale = scale
          self.timeenc = timeenc
          self.freq = freq
-        self.event_dict = {'Up': 1, 'Down': 2, 'Left': 3, 'Right': 4, 'Rest': 6}
+        # self.event_dict = {'Up': 1, 'Down': 2, 'Left': 3, 'Right': 4, 'Rest': 6}
+        # Edit this line in data_loader.py
+        self.event_dict = {'Rest': 0, 'Event1': 1, 'Event2': 2}
  
          self.root_path = root_path
          self.data_path = data_path
@@ -682,6 +682,7 @@ class Dataset_EEG(Dataset):
      def make_contiguous_x_data(self, df_raw, df_split, split):         
          data_x = []
          for trial_start_ind, r in df_split[df_split['split'] == split].iterrows():
+            print(df_raw.loc[trial_start_ind:][df_raw.loc[trial_start_ind:, 'STI'] == self.event_dict['Rest']].iloc[0].name)
              trial_end_ind = df_raw.loc[trial_start_ind:][df_raw.loc[trial_start_ind:, 'STI'] == self.event_dict['Rest']].iloc[0].name
              data_x.append(df_raw.loc[trial_start_ind:trial_end_ind, self.eeg_columns].values)
          return np.concatenate(data_x)
```



- **Modified** `env.yml`

  - **Changed ranges:**

    - Lines 1-7

  - **Diff:**

  ```diff

 diff --git a/env.yml b/env.yml
 index f5110c2..ef4bb09 100644
--- a/env.yml
+++ b/env.yml
@@ -1,5 +1,7 @@
+name: totem_eeg
  channels:
-  - defaults
+  - conda-forge
+  - pytorch
  dependencies:
    - pandas=1.4.2
    - matplotlib
```



- **Added** `scripts/document_changes.py`

  - **Affected functions:**

    - `generate_markdown`

    - `get_recent_changes`

    - `main`

    - `parse_changes`

  - **Changed ranges:**

    - Lines 1-69

  - **Diff:**

  ```diff

 diff --git a/scripts/document_changes.py b/scripts/document_changes.py
 new file mode 100644
 index 0000000..e3de409
--- /dev/null
+++ b/scripts/document_changes.py
@@ -0,0 +1,69 @@
+import os
+import subprocess
+import datetime
+
+def get_recent_changes():
+    # Get git log with details of recent changes
+    result = subprocess.run(
+        ["git", "log", "--name-status", "--pretty=format:%h|%an|%ad|%s", "-n", "10"],
+        capture_output=True, text=True
+    )
+    return result.stdout
+
+def parse_changes(git_log):
+    # Parse git log into structured data
+    changes = []
+    current_commit = None
+    
+    for line in git_log.split('\n'):
+        if '|' in line:  # This is a commit line
+            hash, author, date, message = line.split('|', 3)
+            current_commit = {
+                'hash': hash,
+                'author': author,
+                'date': date,
+                'message': message,
+                'files': []
+            }
+            changes.append(current_commit)
+        elif line.strip() and current_commit is not None:
+            # This is a file change line
+            status, file = line.split('\t', 1) if '\t' in line else (line[0], line[1:].strip())
+            current_commit['files'].append({'status': status, 'file': file})
+    
+    return changes
+
+def generate_markdown(changes):
+    today = datetime.datetime.now().strftime("%Y-%m-%d")
+    
+    md = f"# Code Changes - {today}\n\n"
+    
+    for commit in changes:
+        md += f"## {commit['message']}\n"
+        md += f"**Commit:** {commit['hash']} by {commit['author']} on {commit['date']}\n\n"
+        
+        md += "### Files Changed\n"
+        for file in commit['files']:
+            status_map = {'M': 'Modified', 'A': 'Added', 'D': 'Deleted', 'R': 'Renamed'}
+            status = status_map.get(file['status'], file['status'])
+            md += f"- **{status}**: `{file['file']}`\n"
+        
+        md += "\n---\n\n"
+    
+    return md
+
+def main():
+    os.makedirs("/home/triniborrell/home/projects/TOTEM_for_EEG_code/docs/changes", exist_ok=True)
+    
+    changes = get_recent_changes()
+    parsed_changes = parse_changes(changes)
+    markdown = generate_markdown(parsed_changes)
+    
+    today = datetime.datetime.now().strftime("%Y-%m-%d")
+    with open(f"/home/triniborrell/home/projects/TOTEM_for_EEG_code/docs/changes/changelog-{today}.md", "w") as f:
+        f.write(markdown)
+    
+    print(f"Generated changelog at docs/changes/changelog-{today}.md")
+
+if __name__ == "__main__":
+    main()
 \ No newline at end of file
```



- **Modified** `scripts/step1.sh`

  - **Changed ranges:**

    - Lines 1-6

  - **Diff:**

  ```diff

 diff --git a/scripts/step1.sh b/scripts/step1.sh
 index f2bee23..ff849cb 100644
--- a/scripts/step1.sh
+++ b/scripts/step1.sh
@@ -1,6 +1,6 @@
-repo_dir="/path/to/TOTEM_for_EEG_code"
+repo_dir="/home/triniborrell/home/projects/TOTEM_for_EEG_code"
  
-dataset_name="example"
+dataset_name="a_dataset"
  
  python -m steps.STEP1_save_revin_xdata_for_vqvae \
      +preprocessing=step1_eeg \
```



- **Modified** `scripts/step2.sh`

  - **Changed ranges:**

    - Lines 1-6

  - **Diff:**

  ```diff

 diff --git a/scripts/step2.sh b/scripts/step2.sh
 index e23ade4..dc4b66e 100644
--- a/scripts/step2.sh
+++ b/scripts/step2.sh
@@ -1,6 +1,6 @@
-repo_dir="/path/to/TOTEM_for_EEG_code"
+repo_dir="/home/triniborrell/home/projects/TOTEM_for_EEG_code"
  
-dataset_name="example"
+dataset_name="a_dataset"
  
  python -m steps.STEP2_train_vqvae \
      +exp=step2_train_vqvae \
```



- **Modified** `scripts/step3.sh`

  - **Changed ranges:**

    - Lines 1-6

  - **Diff:**

  ```diff

 diff --git a/scripts/step3.sh b/scripts/step3.sh
 index e6b6701..61bd880 100644
--- a/scripts/step3.sh
+++ b/scripts/step3.sh
@@ -1,6 +1,6 @@
-repo_dir="/path/to/TOTEM_for_EEG_code"
+repo_dir="/home/triniborrell/home/projects/TOTEM_for_EEG_code"
  
-dataset_name="example"
+dataset_name="a_dataset"
  
  python -m steps.STEP3_save_classification_data \
      +preprocessing=step3_eeg \
```



- **Modified** `scripts/step4.sh`

  - **Changed ranges:**

    - Lines 1-6

  - **Diff:**

  ```diff

 diff --git a/scripts/step4.sh b/scripts/step4.sh
 index ca3940b..fcdbc8f 100644
--- a/scripts/step4.sh
+++ b/scripts/step4.sh
@@ -1,6 +1,6 @@
-repo_dir="/path/to/TOTEM_for_EEG_code"
+repo_dir="/home/triniborrell/home/projects/TOTEM_for_EEG_code"
  
-dataset_name="example"
+dataset_name="a_dataset"
  
  python -m steps.STEP4_train_xformer \
      +exp=step4_train_xformer \
```



---



## Update README.md 

**Commit:** `5c4db33` by Geeling Chau on Fri Feb 21 09:59:05 2025 -0800

### Files Changed

- **Modified** `README.md`

  - **Changed ranges:**

    - Lines 76-89

  - **Diff:**

  ```diff

 diff --git a/README.md b/README.md
 index a047bb9..7175ba1 100644
--- a/README.md
+++ b/README.md
@@ -76,14 +76,14 @@ The pipeline has been tested with sampling rates 256-4096Hz, and is agnostic to
  ## Citation
  [TOTEM: TOkenized Time Series EMbeddings for General Time Series Analysis](https://arxiv.org/pdf/2402.16412)
  ```
-@article{talukder_totem_2024,
-	title = {TOTEM: TOkenized Time Series EMbeddings for General Time Series Analysis},
-	issn = {2835-8856},
-	shorttitle = {TOTEM},
-	url = {https://openreview.net/forum?id=QlTLkH6xRC},
-	journal = {Transactions on Machine Learning Research},
-	author = {Talukder, Sabera J. and Yue, Yisong and Gkioxari, Georgia},
-	year = {2024}
+@article{talukder2024totem,
+  title={{TOTEM}: {TO}kenized Time Series {EM}beddings for General Time Series Analysis},
+  author={Sabera J Talukder and Yisong Yue and Georgia Gkioxari},
+  journal={Transactions on Machine Learning Research},
+  issn={2835-8856},
+  year={2024},
+  url={https://openreview.net/forum?id=QlTLkH6xRC},
+  note={}
  }
  ```
```



---



## Update README.md 

**Commit:** `633f1b3` by Geeling Chau on Tue Jan 7 16:16:38 2025 -0800

### Files Changed

- **Modified** `README.md`

  - **Changed ranges:**

    - Lines 74-93

    - Lines 95-98

  - **Diff:**

  ```diff

 diff --git a/README.md b/README.md
 index c1fa653..a047bb9 100644
--- a/README.md
+++ b/README.md
@@ -74,17 +74,20 @@ The pipeline has been tested with sampling rates 256-4096Hz, and is agnostic to
  </details>
  
  ## Citation
+[TOTEM: TOkenized Time Series EMbeddings for General Time Series Analysis](https://arxiv.org/pdf/2402.16412)
  ```
-@misc{talukder2024totem,
-      title={TOTEM: TOkenized Time Series EMbeddings for General Time Series Analysis}, 
-      author={Sabera Talukder and Yisong Yue and Georgia Gkioxari},
-      year={2024},
-      eprint={2402.16412},
-      archivePrefix={arXiv},
-      primaryClass={cs.LG}
+@article{talukder_totem_2024,
+	title = {TOTEM: TOkenized Time Series EMbeddings for General Time Series Analysis},
+	issn = {2835-8856},
+	shorttitle = {TOTEM},
+	url = {https://openreview.net/forum?id=QlTLkH6xRC},
+	journal = {Transactions on Machine Learning Research},
+	author = {Talukder, Sabera J. and Yue, Yisong and Gkioxari, Georgia},
+	year = {2024}
  }
  ```
  
+[Generalizability Under Sensor Failure: Tokenization + Transformers Enable More Robust Latent Spaces](https://arxiv.org/abs/2402.18546)
  ```
  @article{chau2024generalizability,
    title={Generalizability Under Sensor Failure: Tokenization+ Transformers Enable More Robust Latent Spaces},
@@ -92,4 +95,4 @@ The pipeline has been tested with sampling rates 256-4096Hz, and is agnostic to
    journal={arXiv preprint arXiv:2402.18546},
    year={2024}
  }
-```
 \ No newline at end of file
+```
```



---



## Improving clarity 

**Commit:** `70b8b6c` by glchau on Tue Jan 7 15:42:57 2025 -0800

### Files Changed

- **Modified** `README.md`

  - **Changed ranges:**

    - Lines 3-95

  - **Diff:**

  ```diff

 diff --git a/README.md b/README.md
 index a8f185a..c1fa653 100644
--- a/README.md
+++ b/README.md
@@ -3,27 +3,93 @@
  Code and configs for the core model implementations used in the paper [Generalizability Under Sensor Failure: Tokenization + Transformers Enable More Robust Latent Spaces](https://arxiv.org/abs/2402.18546). Adapts the [original TOTEM implementation](https://arxiv.org/pdf/2402.16412) to EEG by implementing appropriate dataloaders and multivariate classification model. 
  
  ## Usage
-1. Create conda env with `env.yml`
-2. Obtain eeg data in the format in the [Data prep](#data-prep) section below. 
-3. Run scripts for steps 1-4 in order 
-    1. Step1 will take the multivariate EEG and convert it to normalized univariate EEG that are used as training trials for training the vq-vae. 
+1. Create a conda env with `env.yml`.
+2. Convert your eeg data the expected format detailed in [Data format](#data-format) and put into your `<repo_dir>/data/` folder.
+3. Edit the scripts with your `<repo_dir>` and `<dataset_name>`. 
+4. Edit the `conf/` files to adjust exp, data processing, and model configurations. 
+5. Run scripts for steps 1-4 in order using `bash ./scripts/stepX.sh`
+    1. Step1 will take the multivariate EEG and convert it to normalized univariate EEG (using the ReVIN module) that are used as training trials for training the vq-vae. 
      2. Step2 will train the vq-vae using the data from Step1. 
+        * You will need to set up comet for logging by copying `conf/logging/comet-template.yaml` to `conf/logging/comet.yaml` and adding your comet credentials.  
      3. Step3 will create the vq-vae tokenized multivariate EEG samples that will be used for downstream classification. 
-    4. Step4 will train a xformer (transformer) classifier 
-
-## Data prep
-* For EEG, make sure your data is formatted with columns as such: 
-    * `dataset.csv`:`,A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14,A15,A16,A17,A18,A19,A20,A21,A22,A23,A24,A25,A26,A27,A28,A29,A30,A31,A32,B1,B2,B3,B4,B5,B6,B7,B8,B9,B10,B11,B12,B13,B14,B15,B16,B17,B18,B19,B20,B21,B22,B23,B24,B25,B26,B27,B28,B29,B30,B31,B32,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15,C16,C17,C18,C19,C20,C21,C22,C23,C24,C25,C26,C27,C28,C29,C30,C31,C32,D1,D2,D3,D4,D5,D6,D7,D8,D9,D10,D11,D12,D13,D14,D15,D16,D17,D18,D19,D20,D21,D22,D23,D24,D25,D26,D27,D28,D29,D30,D31,D32,STI`
-        * First column is the index column which denote the timepoint in the recording.
-        * The current implementation of `Dataset_EEG` assumes 128 channels
-            * The example columns are for biosemi 128 channel device
-        * `STI` is the label column
-            * Values should be numbers representing the class as specified in `Dataset_EEG` `event_dict`
-        * Units of EEG columns are in `uV` and preprocessing is done as specified in the [paper](https://arxiv.org/abs/2402.18546). 
-    * `dataset-split.csv`: `,STI,split`
-        * First column is the index column which denote the timepoint in the recording.
-          * Only the timepoints which mark the beginning of a new trial are kept in this file. 
-        * `STI` is the label column
-            * Values should be numbers representing the class as specified in `Dataset_EEG` `event_dict`
-        * split is a column specifying the train test split assignments
-            * Possible values: {train, val, test} 
+    4. Step4 will train a xformer (transformer) classifier. 
+
+## Data format
+<details>
+<summary> Detailed description: </summary>
+
+* `dataset.csv`
+    * First column is the index column which denote the timepoint in the recording.
+    * The current implementation of `Dataset_EEG` assumes 128 channels
+        * The example columns are for biosemi 128 channel device
+    * `STI` is the label column
+        * Values should be numbers representing the class as specified in `Dataset_EEG` `event_dict`
+    * Units of EEG columns are in `uV` and preprocessing is done as specified in the [paper](https://arxiv.org/abs/2402.18546). 
+* `dataset-split.csv`
+    * First column is the index column which denote the timepoint in the recording.
+        * Only the timepoints which mark the beginning of a new trial are kept in this file. 
+    * `STI` is the label column
+        * Values should be numbers representing the class as specified in `Dataset_EEG` `event_dict`
+    * split is a column specifying the train test split assignments
+        * Possible values: {train, val, test} 
+</details>
+
+
+<details>
+<summary>Example data csv files:</summary>
+
+`dataset.csv`
+|  |    A1 |     A2 |    A3 |     A4 |     A5 |    A6 |    A7 |    A8 |     A9 |    A10 |    A11 |   A12 |    A13 |    A14 |    A15 |    A16 |    A17 |    A18 |    A19 |   A20 |    A21 |   A22 |   A23 |   A24 |   A25 |   A26 |   A27 |   A28 |   A29 |   A30 |   A31 |   A32 |    B1 |    B2 |    B3 |    B4 |    B5 |   B6 |    B7 |    B8 |    B9 |   B10 |   B11 |   B12 |   B13 |   B14 |   B15 |   B16 |   B17 |    B18 |   B19 |   B20 |    B21 |   B22 |    B23 |   B24 |   B25 |   B26 |    B27 |    B28 |    B29 |    B30 |   B31 |   B32 |    C1 |    C2 |    C3 |     C4 |     C5 |    C6 |    C7 |    C8 |    C9 |   C10 |   C11 |   C12 |   C13 |   C14 |   C15 |   C16 |    C17 |   C18 |   C19 |   C20 |   C21 |    C22 |   C23 |   C24 |    C25 |    C26 |   C27 |   C28 |    C29 |   C30 |   C31 |   C32 |     D1 |    D2 |     D3 |     D4 |     D5 |    D6 |    D7 |     D8 |     D9 |    D10 |    D11 |    D12 |    D13 |    D14 |    D15 |   D16 |    D17 |    D18 |    D19 |    D20 |    D21 |    D22 |    D23 |   D24 |    D25 |    D26 |    D27 |    D28 |   D29 |    D30 |   D31 |   D32 |   STI |
+|-------------:|------:|-------:|------:|-------:|-------:|------:|------:|------:|-------:|-------:|-------:|------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|------:|-------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|-----:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|-------:|------:|------:|-------:|------:|-------:|------:|------:|------:|-------:|-------:|-------:|-------:|------:|------:|------:|------:|------:|-------:|-------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|------:|------:|-------:|-------:|------:|------:|-------:|------:|------:|------:|-------:|------:|-------:|-------:|-------:|------:|------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|------:|-------:|-------:|-------:|-------:|------:|-------:|------:|------:|------:|
+|            0 |  0    |   0    | -0    |   0.01 |  -0    |  0    |  0    |  0    |   0    |   0    |   0    | -0    |  -0    |   0    |   0    |   0    |   0    |   0    |   0    |  0    |  -0    |  0.01 | -0    | -0    | -0.01 | -0    | -0    | -0    |  0.01 |  0.01 | -0    |  0    |  0    |  0    |  0    | -0    | -0    | 0    |  0    |  0    | -0    | -0    | -0    |  0    |  0    | -0    | -0    |  0    |  0    |   0    | -0    | -0    |   0    | -0    |   0    | -0    | -0.01 | -0.01 |   0    |   0    |   0    |  -0    |  0    | -0    |  0    | -0    |  0    |  -0    |  -0    |  0    |  0    |  0    | -0    | -0    |  0    |  0    |  0    |  0    |  0    |  0    |   0    |  0    | -0    | -0    |  0    |  -0    | -0    | -0    |   0    |   0    | -0    | -0    |   0    |  0    | -0    |  0    |   0.01 |  0    |  -0    |  -0    |  -0    | -0    | -0    |   0    |  -0.01 |  -0.01 |  -0    |  -0    |  -0    |   0    |   0    | -0    |   0    |   0    |  -0    |   0    |   0    |  -0    |  -0.01 | -0    |   0    |   0    |   0    |   0    |  0    |   0    |  0    | -0    |     0 |
+|            1 | -1.6  | -35.99 |  2.29 | -29.54 |  -7.82 | -3.52 | -4.2  | -7.48 |  -5.39 |  -1.24 |  -2.87 | -2.12 |  -3.67 |   0.61 |  -1.41 | -12.01 |  -9.75 |  -3.76 |  -9.29 | -7.56 | -61.4  | -6.41 | -1.24 | -2.71 |  4.58 | -9.75 | -0.3  | -6.72 | -4.77 | 41.88 | -1.95 | -3.13 | -0.41 | -4.08 |  4.14 |  1.47 | 25.73 | 0.08 |  2.66 | 11.76 | -4.17 | -7.05 |  6.64 | -0.7  | -1.98 | 10.43 | -2.96 | -0.01 |  3.77 |   0.08 |  3.95 |  3.21 |   2.28 |  6.41 |  10.73 |  7.61 | -0.74 | 18.04 | -17.57 | -13.97 |  35.06 |  18.59 |  9.04 |  6.6  | -4.74 |  4.94 |  6.82 |   6.31 |   5.01 | -4.5  |  5.69 | 21.68 | 41.79 |  4.74 |  2.95 |  0.03 |  7.14 |  7.53 | 15.66 | 14.73 |  -0.66 |  4.89 |  1.8  |  2.29 |  2.71 |  -2.43 |  2.07 |  0.22 | -38.28 | -11.08 |  3.03 | 11.77 | -15.7  | 30.52 |  8.4  | -2.69 | -33.4  | -7.47 |  -5.32 | -14.82 | -24.94 | 57.53 | 73.82 |  42.32 |  24.04 | -21.78 | -12.06 |  -7.91 |  -5.64 |  -7.72 | -13.47 |  2.75 |  -3.08 |  -2.26 | -13.08 | -17.2  | -22.2  |  -2.36 |  15.9  | -2.39 | -12.93 | -18.67 | -10.09 |  -4.68 | -7.15 | -12.41 | -0.46 | -1.87 |     0 |
+|            2 | -5.56 | -36.17 | -2.59 |  23.37 |  -9.78 | -8.41 | -8.28 | -8.91 |  -9.48 |   2.39 |   6.32 |  0.45 |  -1.05 |   0.66 |  -7.01 | -20.86 | -21.67 |  -8.31 |  -8.89 | -9.55 |   9.03 | 34.7  | -1.18 | -4.13 |  8.33 | -7.69 |  4.34 | -9.4  | -1.64 | 24.99 | -0.58 | -3.62 | -2.36 | -3.34 |  2.24 | -4.59 | 20.4  | 0.31 | -6.53 |  5.71 | -4.3  | -8.25 |  5.5  | -0.64 | -4.83 | 10.19 |  0.29 |  0.39 |  4.14 |   1.68 |  3.74 |  2.99 |   7.58 | 11.09 |  16.92 | 13.68 |  6.03 | 65.67 |   0.36 | -41.3  |  54.4  |  29.01 | 14.34 |  8.33 | -3.28 |  5.66 | 11.03 |  12.05 |   8.42 |  4.99 | -2.32 | 39.48 | 41.17 |  6.07 |  3.49 |  1.77 |  6.89 |  9.07 | 14.23 | 14.78 |  -0.04 |  4.22 |  5.14 |  4.45 |  4.9  |  -4.88 |  0.55 | -2.22 | -31.27 |  -7.15 |  7.76 |  3.98 | -23.93 | 15.23 | 24.37 |  9.81 | -32.04 | -8.79 | -10.91 | -18.49 | -33.9  | 71.95 | 44.52 |  -4.48 |  17.57 | -44.9  | -29.88 | -19.35 | -11.52 | -11.83 | -22.75 | -1.68 | -10.03 | -10.88 | -23.34 | -30.65 | -35.71 | -24.61 |  -1.32 | -6.48 | -18.47 | -23.99 | -15.5  |  22.84 | -8.37 | -11.33 |  1.95 | -0.75 |     0 |
+|            3 | -7.11 | -16.91 | -5.81 |  -0.97 | -12.2  | -7.62 | -5.75 | -5.68 |  -9.96 | -10.65 | -18.99 | -7.38 | -10.08 | -10.02 | -15.13 |  -5.39 |  -5.37 |  -8.93 | -15.64 | -2.06 | 100.66 | 69.33 | -2.35 | -4.1  |  1.69 | -7.12 | 14.09 | -5.68 | -3.96 | 91.12 | -4.78 | -6.83 | -5.79 | -9.89 | -4.05 | -3.73 | 19.81 | 1.61 |  3.42 | 14.57 | -3.41 | -1.16 | 11.24 | -5.14 | -9.73 |  8    |  5.82 | -5.75 | -2.06 |  -6.4  | -2.95 | -6.77 | -10.26 | -0.59 |  -5.21 |  1.96 | 16.33 | 52.02 |  57.2  |  58.43 | -25.45 | -12.58 | -8.27 | -8.23 | -6.22 | -3.19 | -6.6  | -10.39 | -17.86 | 12.27 | 14.9  |  9.39 | 51.4  |  3.41 | -6.08 | -8.93 | -0.72 |  2.55 |  1.93 | 15.03 | -12.41 | -5.9  | -4.1  | -2.94 | -5.61 | -11.54 | -2.15 | -5.79 |  -5.68 | -17.17 | -5.06 | -1.34 | -26.28 | -6.3  | -6.73 |  2.29 | -11.7  | -4.17 | -12.12 | -26.76 | -33.13 | 53.59 |  6.4  | -44.68 | -11.72 | -12.07 | -15.5  |  -8.91 |  -9.09 | -10.32 |  -5.33 |  9.45 |  -5.16 |  -3.34 |  -8.29 |  -4.59 |  14.07 | -10.43 | -39.93 | -4.36 |  -5.11 |  -1.79 |  -1.81 | 110.36 |  5.28 |  -2.66 | -5.6  | -8.14 |     0 |
+|            4 | -6.57 |  -5.34 | -6.3  |  26.85 | -11.39 | -8.28 | -5.12 | -8.08 | -14.77 |  -8.31 |  13.56 | -7.62 |  -9.57 |  -4.74 |  -6.45 | -15.6  | -18.74 | -13.65 | -14.41 | -7.22 |  56.43 | 60.44 | -6.56 | -4.48 | -5.32 | -9.21 | 58.9  | -7.63 | -1.76 | 77.02 | -8.79 | -5.93 | -5.33 | -8.28 | -7.49 | -6.32 | 14.79 | 0.82 |  1.13 |  9.1  | -8.88 |  1.05 | 12.49 | -6.5  | -9.31 |  1.14 |  7.53 | -5.74 | -5.32 | -10.57 | -1.3  | -5.14 |  -9.03 | -5.63 | -10.08 | -3.02 |  3.55 | 50.76 |  33.02 |  34.78 | -20.86 | -13.59 | -9.31 | -3.8  | -6.94 | -0.8  | -3.86 |  -6.41 |  -7.89 |  9.62 | 16.54 |  3.01 | 66.54 |  9.39 | -4    | -4.65 |  0.54 |  7.89 |  0.89 | 13.27 |  -7.71 | -2.83 |  1.84 |  0.79 | -0.67 |  -6.93 | -1.72 | -6.03 | -13.22 | -10.36 |  0.09 |  1.71 | -22.24 | -1.61 |  2.07 | 10.55 | -35.57 | -0.58 |  -9.58 | -16.55 | -21.11 | 57.74 | -2.71 | -28.81 |   6.32 | -17.83 |  -9.61 |  -8.4  |  -7.59 |  -8.58 |  -2.45 |  1.28 |  -5.55 |   0.5  |  -7.85 |  -3.75 |   5.8  | -23.28 | -35.7  | -5.49 | -12.6  |  -2.36 |  -4.14 |  93.6  | -3.58 |  -5.87 | -3.88 | -4.71 |     0 |
+|            5 | -3.25 | -15.77 | -7.52 |  45.73 | -13.16 | -8.64 | -3.79 | -7.87 | -19.53 | -13.22 |   1.68 | -6.54 |  -8.06 |  -5.58 |  -6.68 |   0.76 |  -1.42 | -11.03 | -16.23 | -8.85 |  55.46 | 85.37 | -8.1  | -5.76 | 10.28 | -7.81 | 33.75 | -1.06 |  8.36 | 45.86 | -2.84 | -1.96 | -4.43 | -9.47 | -4.1  | -3.31 | 11.25 | 3.32 | -3.2  |  7.94 | -1.95 | -0.54 |  9.16 | -1.31 | -6.27 | 16.58 | 12.04 |  2.44 | -2.08 |  -9.01 |  3.62 | -2.81 |   1.54 | -2.32 |  -0.33 |  3.53 |  1.53 | 41.14 |  18.85 | -10.15 |   3.47 |   1.74 | -0.56 | -1.05 | -4.58 | -1.6  |  3.32 |   1.92 |   0.88 | 13.58 | -7.47 |  7    | 32.84 |  8.13 | -4.23 | -3.04 |  2.49 |  3.13 | -6.68 | 18.92 |  -8.2  | -1.82 |  0.14 |  1.21 |  1.1  |  -5.52 | -2.81 | -5.13 |  -8.45 |  -8.15 | -0.36 |  4.58 | -29.45 | -3.43 |  8.95 |  9.27 | -20.29 | -2.12 |  -9.2  | -21.44 | -11.91 | 62.83 | 17.46 |  -8.79 |  36.64 | -11.79 | -19.74 |  -9.77 | -13.05 | -12.44 |  -4.29 |  4.74 | -10.37 |  -9.52 | -21.04 | -27.43 | -50.24 |   0.2  | -17.15 | -7.97 | -14.88 | -18.7  | -13.04 |  42.66 | -2.44 |  -5.42 | -7.46 | -8.45 |     0 |
+
+... and many more rows, one per timepoint. 
+
+`dataset-split.csv`
+|       | STI | split |
+|-------|-----|-------|
+| 10000 | 1.0 | test  |
+| 20240 | 2.0 | test  |
+| 30480 | 2.0 | train |
+| 40720 | 1.0 | train |
+| 50960 | 2.0 | train |
+| 61200 | 4.0 | train |
+| 71440 | 3.0 | val   |
+| 81680 | 1.0 | train |
+| 91920 | 1.0 | val   |
+
+... and more depending on number of trials you have
+</details>
+
+## FAQ
+<details>
+<summary>What sampling rate should I save my data in?</summary>
+
+The pipeline has been tested with sampling rates 256-4096Hz, and is agnostic to the underlying sampling rate. Some sampling rates may work better with the default window sizes (96 timepoints for VQVAE training, and 512 timepoints for classification modeling), depending on the nature of the task. Experimentation is encouraged! That said, it is important that the `dataset-split.csv` file is properly indexed to leverage the same sampling rate as `dataset.csv`.
+</details>
+
+## Citation
+```
+@misc{talukder2024totem,
+      title={TOTEM: TOkenized Time Series EMbeddings for General Time Series Analysis}, 
+      author={Sabera Talukder and Yisong Yue and Georgia Gkioxari},
+      year={2024},
+      eprint={2402.16412},
+      archivePrefix={arXiv},
+      primaryClass={cs.LG}
+}
+```
+
+```
+@article{chau2024generalizability,
+  title={Generalizability Under Sensor Failure: Tokenization+ Transformers Enable More Robust Latent Spaces},
+  author={Chau, Geeling and An, Yujin and Iqbal, Ahamed Raffey and Chung, Soon-Jo and Yue, Yisong and Talukder, Sabera},
+  journal={arXiv preprint arXiv:2402.18546},
+  year={2024}
+}
+```
 \ No newline at end of file
```



- **R092** `conf/exp/train_vqvae.yaml	conf/exp/step2_train_vqvae.yaml`

  - **Diff:**

  ```diff

```



- **R100** `conf/exp/train_xformer.yaml	conf/exp/step4_train_xformer.yaml`

  - **Diff:**

  ```diff

```



- **Modified** `scripts/step2.sh`

  - **Changed ranges:**

    - Lines 1-9

  - **Diff:**

  ```diff

 diff --git a/scripts/step2.sh b/scripts/step2.sh
 index 975bff9..e23ade4 100644
--- a/scripts/step2.sh
+++ b/scripts/step2.sh
@@ -1,8 +1,9 @@
  repo_dir="/path/to/TOTEM_for_EEG_code"
  
  dataset_name="example"
+
  python -m steps.STEP2_train_vqvae \
-    +exp=train_vqvae \
+    +exp=step2_train_vqvae \
      ++exp.save_dir="${repo_dir}/pipeline/step2_train_vqvae/${dataset_name}" \
      ++exp.vqvae_config.dataset=${dataset_name} \
      ++exp.vqvae_config.dataset_base_path="${repo_dir}/pipeline/step1_revin_x_data" \
```



- **Modified** `scripts/step3.sh`

  - **Changed ranges:**

    - Lines 1-8

  - **Diff:**

  ```diff

 diff --git a/scripts/step3.sh b/scripts/step3.sh
 index a026994..e6b6701 100644
--- a/scripts/step3.sh
+++ b/scripts/step3.sh
@@ -1,7 +1,8 @@
  repo_dir="/path/to/TOTEM_for_EEG_code"
  
  dataset_name="example"
-python -m pdb -m steps.STEP3_save_classification_data \
+
+python -m steps.STEP3_save_classification_data \
      +preprocessing=step3_eeg \
      "++preprocessing.train_root_paths=['${repo_dir}/data']" \
      "++preprocessing.train_data_paths=['${dataset_name}.csv']" \
```



- **Modified** `scripts/step4.sh`

  - **Changed ranges:**

    - Lines 3-9

  - **Diff:**

  ```diff

 diff --git a/scripts/step4.sh b/scripts/step4.sh
 index b5de2ea..ca3940b 100644
--- a/scripts/step4.sh
+++ b/scripts/step4.sh
@@ -3,7 +3,7 @@ repo_dir="/path/to/TOTEM_for_EEG_code"
  dataset_name="example"
  
  python -m steps.STEP4_train_xformer \
-    +exp=train_xformer \
+    +exp=step4_train_xformer \
      ++exp.classifier_config.data_path="${repo_dir}/pipeline/step3_classification_data/${dataset_name}" \
      ++exp.classifier_config.checkpoint_path="${repo_dir}/pipeline/step4_train_xformer" \
      ++exp.classifier_config.exp_name=${dataset_name} \
```



---



## Update README.md 

**Commit:** `827ed59` by Geeling Chau on Sun Dec 8 00:51:06 2024 -0800

### Files Changed

- **Modified** `README.md`

  - **Changed ranges:**

    - Lines 7-29

  - **Diff:**

  ```diff

 diff --git a/README.md b/README.md
 index 3a87cb0..a8f185a 100644
--- a/README.md
+++ b/README.md
@@ -7,18 +7,23 @@ Code and configs for the core model implementations used in the paper [Generaliz
  2. Obtain eeg data in the format in the [Data prep](#data-prep) section below. 
  3. Run scripts for steps 1-4 in order 
      1. Step1 will take the multivariate EEG and convert it to normalized univariate EEG that are used as training trials for training the vq-vae. 
-    2. Step2 will train the vqvae using the data from Step1. 
+    2. Step2 will train the vq-vae using the data from Step1. 
      3. Step3 will create the vq-vae tokenized multivariate EEG samples that will be used for downstream classification. 
-    4. Step4 will train a transformer classifier 
+    4. Step4 will train a xformer (transformer) classifier 
  
  ## Data prep
  * For EEG, make sure your data is formatted with columns as such: 
-    * dataset.csv:,A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14,A15,A16,A17,A18,A19,A20,A21,A22,A23,A24,A25,A26,A27,A28,A29,A30,A31,A32,B1,B2,B3,B4,B5,B6,B7,B8,B9,B10,B11,B12,B13,B14,B15,B16,B17,B18,B19,B20,B21,B22,B23,B24,B25,B26,B27,B28,B29,B30,B31,B32,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15,C16,C17,C18,C19,C20,C21,C22,C23,C24,C25,C26,C27,C28,C29,C30,C31,C32,D1,D2,D3,D4,D5,D6,D7,D8,D9,D10,D11,D12,D13,D14,D15,D16,D17,D18,D19,D20,D21,D22,D23,D24,D25,D26,D27,D28,D29,D30,D31,D32,STI
-        * First column is the index column which denote the timepoint in the recording
-        * The current implementation of Dataset_EEG assumes 128 channels (the current headings are for biosemi 128 channel device)
-        * STI is the label column (values should be numbers representing the class as specified in Dataset_EEG event_dict)
-        * Units are in uV and preprocessing is done as specified in the paper. 
-    * dataset-split.csv: ,STI,split
-        * First column is the index that corresponds to the beginning of this trial with label specified
-        * STI is the label column (values should be numbers representing the class as specified in Dataset_EEG event_dict)
-        * split is a column with these values: {train, val, test} 
+    * `dataset.csv`:`,A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14,A15,A16,A17,A18,A19,A20,A21,A22,A23,A24,A25,A26,A27,A28,A29,A30,A31,A32,B1,B2,B3,B4,B5,B6,B7,B8,B9,B10,B11,B12,B13,B14,B15,B16,B17,B18,B19,B20,B21,B22,B23,B24,B25,B26,B27,B28,B29,B30,B31,B32,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15,C16,C17,C18,C19,C20,C21,C22,C23,C24,C25,C26,C27,C28,C29,C30,C31,C32,D1,D2,D3,D4,D5,D6,D7,D8,D9,D10,D11,D12,D13,D14,D15,D16,D17,D18,D19,D20,D21,D22,D23,D24,D25,D26,D27,D28,D29,D30,D31,D32,STI`
+        * First column is the index column which denote the timepoint in the recording.
+        * The current implementation of `Dataset_EEG` assumes 128 channels
+            * The example columns are for biosemi 128 channel device
+        * `STI` is the label column
+            * Values should be numbers representing the class as specified in `Dataset_EEG` `event_dict`
+        * Units of EEG columns are in `uV` and preprocessing is done as specified in the [paper](https://arxiv.org/abs/2402.18546). 
+    * `dataset-split.csv`: `,STI,split`
+        * First column is the index column which denote the timepoint in the recording.
+          * Only the timepoints which mark the beginning of a new trial are kept in this file. 
+        * `STI` is the label column
+            * Values should be numbers representing the class as specified in `Dataset_EEG` `event_dict`
+        * split is a column specifying the train test split assignments
+            * Possible values: {train, val, test}
```



---



## Initial commit 

**Commit:** `6c59c8e` by glchau on Sun Dec 8 00:30:18 2024 -0800

### Files Changed

- **Added** `.gitignore`

  - **Diff:**

  ```diff

+**/__pycache__/**
+data/**
+outputs/**
+pipeline/**
+conf/logging/comet.yaml
```



- **Added** `README.md`

  - **Diff:**

  ```diff

+
+# TOTEM for EEG code
+Code and configs for the core model implementations used in the paper [Generalizability Under Sensor Failure: Tokenization + Transformers Enable More Robust Latent Spaces](https://arxiv.org/abs/2402.18546). Adapts the [original TOTEM implementation](https://arxiv.org/pdf/2402.16412) to EEG by implementing appropriate dataloaders and multivariate classification model. 
+
+## Usage
+1. Create conda env with `env.yml`
+2. Obtain eeg data in the format in the [Data prep](#data-prep) section below. 
+3. Run scripts for steps 1-4 in order 
+    1. Step1 will take the multivariate EEG and convert it to normalized univariate EEG that are used as training trials for training the vq-vae. 
+    2. Step2 will train the vqvae using the data from Step1. 
+    3. Step3 will create the vq-vae tokenized multivariate EEG samples that will be used for downstream classification. 
+    4. Step4 will train a transformer classifier 
+
+## Data prep
+* For EEG, make sure your data is formatted with columns as such: 
+    * dataset.csv:,A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14,A15,A16,A17,A18,A19,A20,A21,A22,A23,A24,A25,A26,A27,A28,A29,A30,A31,A32,B1,B2,B3,B4,B5,B6,B7,B8,B9,B10,B11,B12,B13,B14,B15,B16,B17,B18,B19,B20,B21,B22,B23,B24,B25,B26,B27,B28,B29,B30,B31,B32,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15,C16,C17,C18,C19,C20,C21,C22,C23,C24,C25,C26,C27,C28,C29,C30,C31,C32,D1,D2,D3,D4,D5,D6,D7,D8,D9,D10,D11,D12,D13,D14,D15,D16,D17,D18,D19,D20,D21,D22,D23,D24,D25,D26,D27,D28,D29,D30,D31,D32,STI
+        * First column is the index column which denote the timepoint in the recording
+        * The current implementation of Dataset_EEG assumes 128 channels (the current headings are for biosemi 128 channel device)
+        * STI is the label column (values should be numbers representing the class as specified in Dataset_EEG event_dict)
+        * Units are in uV and preprocessing is done as specified in the paper. 
+    * dataset-split.csv: ,STI,split
+        * First column is the index that corresponds to the beginning of this trial with label specified
+        * STI is the label column (values should be numbers representing the class as specified in Dataset_EEG event_dict)
+        * split is a column with these values: {train, val, test}
```



- **Added** `conf/exp/train_vqvae.yaml`

  - **Diff:**

  ```diff

+save_dir: ???
+gpu_id: 1
+data_init_cpu_or_gpu: "cpu"
+vqvae_config: 
+  general_seed: 47
+  model_name: "vqvae"
+  model_save_name: "vqvae_eeg_test"
+  data_name: "autoformer_vqvae"
+  pretrained: false
+  learning_rate: 0.001
+  num_epochs: 200
+  batch_size: 4096
+  block_hidden_size: 128
+  num_residual_layers: 2
+  res_hidden_size: 64
+  embedding_dim: 64
+  num_embeddings: 256
+  commitment_cost: 0.25
+  compression_factor: 4
+  dataset: ??? 
+  dataset_base_path: ???
```



- **Added** `conf/exp/train_xformer.yaml`

  - **Diff:**

  ```diff

+cuda_id: 0
+classifier_config: 
+  seed: 47
+  model_type: "xformer"
+  data_type: "eeg"
+  data_path: ???
+  codebook_size: 256
+  compression: 4
+  Tin: 512
+  checkpoint: true 
+  checkpoint_path: ??? 
+  exp_name: ??? 
+  optimizer: "adam" 
+  scheduler: "step"
+  baselr: 0.0001
+  epochs: 50
+  steps: 51
+  batchsize: 32
+  patience: 7
+  d_model: 64
+  d_hid: 128
+  nhead: 4
+  nlayers: 6
+  nsensors: 128 
+  nclasses: 4
+  onehot: false
+  scale: false
```



- **Added** `conf/logging/comet_template.yaml`

  - **Diff:**

  ```diff

+comet:
+  api_key: ???
+  project_name: ???
+  workspace: ???
+  comet_tag: ???
+  comet_name: ???
```



- **Added** `conf/preprocessing/step1_eeg.yaml`

  - **Diff:**

  ```diff

+random_seed: 2021
+data: "eeg" ## e.g. "eeg"
+root_paths: ??? ## e.g. ["./data/ETT/"]
+data_paths: ??? ## e.g. ["ETTh1.csv"]
+seq_len: 96 ## input sequence length
+label_len: 0 ## start token length
+pred_len: 96 ## prediction sequence length
+enc_in: 128 ## Not really needed bc revin affine=False
+use_gpu: True
+gpu: 0
+use_multi_gpu: False
+devices: '0,1,2,3'
+save_path: ???
+num_workers: 10 # data loader num workers
+batch_size: 128 # batch size of train input data
+features: "M" # forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
+target: "half" # target feature in S or MS task (overriden as step_size in EEG dataset)
+freq: "h" # freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h
+embed: "timeF" # time features encoding, options:[timeF, fixed, learned]; timeF:Time Features, fixed:learn a fixed representation, learned:let model learn the representation
```



- **Added** `conf/preprocessing/step3_eeg.yaml`

  - **Diff:**

  ```diff

+random_seed: 2021
+data: eeg
+train_root_paths: ???
+train_data_paths: ???
+test_root_paths: ???
+test_data_paths: ???
+features: "M" # options: [clean or other] if "clean" will clean the data using the specified bad_channels.csv in corresponding data_path.
+target: "half" # target feature in S or MS task (overriden as step_size in EEG dataset)
+freq: "h" # freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h
+seq_len: 512
+label_len: 0
+pred_len: 0
+enc_in: 128
+embed: "timeF" # time features encoding, options:[timeF, fixed, learned]
+num_workers: 10 
+batch_size: 128
+use_gpu: True 
+gpu: 0
+save_path: ???
+trained_vqvae_model_path: ???
+compression_factor: 4
```



- **Added** `data_provider/data_factory_vqvae_no_shuffle.py`

  - **Affected functions:**

    - `data_provider`

    - `data_provider_flexPath`

  - **Diff:**

  ```diff

+import pdb
+
+from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, Dataset_Neuro, Dataset_Earthquake, Dataset_EEG
+from torch.utils.data import DataLoader
+
+data_dict = {
+    'ETTh1': Dataset_ETT_hour,
+    'ETTh2': Dataset_ETT_hour,
+    'ETTm1': Dataset_ETT_minute,
+    'ETTm2': Dataset_ETT_minute,
+    'custom': Dataset_Custom,
+    'neuro': Dataset_Neuro,
+    'earthquake': Dataset_Earthquake,
+    'eeg': Dataset_EEG,
+}
+
+def data_provider(args, flag):
+    Data = data_dict[args.data]
+    timeenc = 0 if args.embed != 'timeF' else 1
+
+    if flag == 'test':
+        shuffle_flag = False
+        drop_last = False
+        batch_size = args.batch_size
+        freq = args.freq
+    elif flag == 'pred':
+        shuffle_flag = False
+        drop_last = False
+        batch_size = 1
+        freq = args.freq
+        Data = Dataset_Pred
+    else:
+        shuffle_flag = False
+        drop_last = False
+        batch_size = args.batch_size
+        freq = args.freq
+
+    data_set = Data(
+        root_path=args.root_path,
+        data_path=args.data_path,
+        flag=flag,
+        size=[args.seq_len, args.label_len, args.pred_len],
+        features=args.features,
+        target=args.target,
+        timeenc=timeenc,
+        freq=freq
+    )
+    print(flag, len(data_set))
+    data_loader = DataLoader(
+        data_set,
+        batch_size=batch_size,
+        shuffle=shuffle_flag,
+        num_workers=args.num_workers,
+        drop_last=drop_last)
+    return data_set, data_loader
+
+def data_provider_flexPath(args, root_path, data_path, flag):
+    Data = data_dict[args.data]
+    timeenc = 0 if args.embed != 'timeF' else 1
+
+    if flag == 'test':
+        shuffle_flag = False
+        drop_last = False
+        batch_size = args.batch_size
+        freq = args.freq
+    elif flag == 'pred':
+        shuffle_flag = False
+        drop_last = False
+        batch_size = 1
+        freq = args.freq
+        Data = Dataset_Pred
+    else:
+        shuffle_flag = False
+        drop_last = False
+        batch_size = args.batch_size
+        freq = args.freq
+
+    data_set = Data(
+        root_path=root_path,
+        data_path=data_path,
+        flag=flag,
+        size=[args.seq_len, args.label_len, args.pred_len],
+        features=args.features,
+        target=args.target,
+        timeenc=timeenc,
+        freq=freq
+    )
+    print(flag, len(data_set))
+    data_loader = DataLoader(
+        data_set,
+        batch_size=batch_size,
+        shuffle=shuffle_flag,
+        num_workers=args.num_workers,
+        drop_last=drop_last)
+    return data_set, data_loader
```



- **Added** `data_provider/data_loader.py`

  - **Diff:**

  ```diff

+import os
+import pdb
+
+import numpy as np
+import pandas as pd
+import os
+import torch
+from torch.utils.data import Dataset, DataLoader
+from sklearn.preprocessing import StandardScaler
+from lib.utils.timefeatures import time_features
+from pathlib import Path ## EEG dataset specific 
+import warnings
+
+warnings.filterwarnings('ignore')
+
+
+class Dataset_ETT_hour(Dataset):
+    def __init__(self, root_path, flag='train', size=None,
+                 features='S', data_path='ETTh1.csv',
+                 target='OT', scale=True, timeenc=0, freq='h'):
+        # size [seq_len, label_len, pred_len]
+        # info
+        if size == None:
+            self.seq_len = 24 * 4 * 4
+            self.label_len = 24 * 4
+            self.pred_len = 24 * 4
+        else:
+            self.seq_len = size[0]
+            self.label_len = size[1]
+            self.pred_len = size[2]
+        # init
+        assert flag in ['train', 'test', 'val']
+        type_map = {'train': 0, 'val': 1, 'test': 2}
+        self.set_type = type_map[flag]
+
+        self.features = features
+        self.target = target
+        self.scale = scale
+        self.timeenc = timeenc
+        self.freq = freq
+
+        self.root_path = root_path
+        self.data_path = data_path
+        self.__read_data__()
+
+    def __read_data__(self):
+        self.scaler = StandardScaler()
+        df_raw = pd.read_csv(os.path.join(self.root_path,
+                                          self.data_path))
+
+        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
+        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
+        border1 = border1s[self.set_type]
+        border2 = border2s[self.set_type]
+
+        if self.features == 'M' or self.features == 'MS':
+            cols_data = df_raw.columns[1:]
+            df_data = df_raw[cols_data]
+        elif self.features == 'S':
+            df_data = df_raw[[self.target]]
+
+        if self.scale:
+            train_data = df_data[border1s[0]:border2s[0]]
+            self.scaler.fit(train_data.values)
+            data = self.scaler.transform(df_data.values)
+        else:
+            data = df_data.values
+
+        df_stamp = df_raw[['date']][border1:border2]
+        df_stamp['date'] = pd.to_datetime(df_stamp.date)
+        if self.timeenc == 0:
+            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
+            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
+            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
+            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
+            data_stamp = df_stamp.drop(['date'], axis=1).values
+        elif self.timeenc == 1:
+            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
+            data_stamp = data_stamp.transpose(1, 0)
+
+        self.data_x = data[border1:border2]
+        self.data_y = data[border1:border2]
+        self.data_stamp = data_stamp
+
+    def __getitem__(self, index):
+        s_begin = index
+        s_end = s_begin + self.seq_len
+        r_begin = s_end - self.label_len
+        r_end = r_begin + self.label_len + self.pred_len
+
+        seq_x = self.data_x[s_begin:s_end]
+        seq_y = self.data_y[r_begin:r_end]
+        seq_x_mark = self.data_stamp[s_begin:s_end]
+        seq_y_mark = self.data_stamp[r_begin:r_end]
+
+        return seq_x, seq_y, seq_x_mark, seq_y_mark
+
+    def __len__(self):
+        return len(self.data_x) - self.seq_len - self.pred_len + 1
+
+    def inverse_transform(self, data):
+        return self.scaler.inverse_transform(data)
+
+
+class Dataset_ETT_minute(Dataset):
+    def __init__(self, root_path, flag='train', size=None,
+                 features='S', data_path='ETTm1.csv',
+                 target='OT', scale=True, timeenc=0, freq='t'):
+        # size [seq_len, label_len, pred_len]
+        # info
+        if size == None:
+            self.seq_len = 24 * 4 * 4
+            self.label_len = 24 * 4
+            self.pred_len = 24 * 4
+        else:
+            self.seq_len = size[0]
+            self.label_len = size[1]
+            self.pred_len = size[2]
+        # init
+        assert flag in ['train', 'test', 'val']
+        type_map = {'train': 0, 'val': 1, 'test': 2}
+        self.set_type = type_map[flag]
+
+        self.features = features
+        self.target = target
+        self.scale = scale
+        self.timeenc = timeenc
+        self.freq = freq
+
+        self.root_path = root_path
+        self.data_path = data_path
+        self.__read_data__()
+
+    def __read_data__(self):
+        self.scaler = StandardScaler()
+        df_raw = pd.read_csv(os.path.join(self.root_path,
+                                          self.data_path))
+
+        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
+        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
+        border1 = border1s[self.set_type]
+        border2 = border2s[self.set_type]
+
+        if self.features == 'M' or self.features == 'MS':
+            cols_data = df_raw.columns[1:]
+            df_data = df_raw[cols_data]
+        elif self.features == 'S':
+            df_data = df_raw[[self.target]]
+
+        if self.scale:
+            train_data = df_data[border1s[0]:border2s[0]]
+            self.scaler.fit(train_data.values)
+            data = self.scaler.transform(df_data.values)
+        else:
+            data = df_data.values
+
+        df_stamp = df_raw[['date']][border1:border2]
+        df_stamp['date'] = pd.to_datetime(df_stamp.date)
+        if self.timeenc == 0:
+            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
+            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
+            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
+            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
+            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
+            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
+            data_stamp = df_stamp.drop(['date'], axis=1).values
+        elif self.timeenc == 1:
+            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
+            data_stamp = data_stamp.transpose(1, 0)
+
+        self.data_x = data[border1:border2]
+        self.data_y = data[border1:border2]
+        self.data_stamp = data_stamp
+
+    def __getitem__(self, index):
+        s_begin = index
+        s_end = s_begin + self.seq_len
+        r_begin = s_end - self.label_len
+        r_end = r_begin + self.label_len + self.pred_len
+
+        seq_x = self.data_x[s_begin:s_end]
+        seq_y = self.data_y[r_begin:r_end]
+        seq_x_mark = self.data_stamp[s_begin:s_end]
+        seq_y_mark = self.data_stamp[r_begin:r_end]
+
+        return seq_x, seq_y, seq_x_mark, seq_y_mark
+
+    def __len__(self):
+        return len(self.data_x) - self.seq_len - self.pred_len + 1
+
+    def inverse_transform(self, data):
+        return self.scaler.inverse_transform(data)
+
+
+class Dataset_Custom(Dataset):
+    def __init__(self, root_path, flag='train', size=None,
+                 features='S', data_path='ETTh1.csv',
+                 target='OT', scale=True, timeenc=0, freq='h'):
+        # size [seq_len, label_len, pred_len]
+        # info
+        if size == None:
+            self.seq_len = 24 * 4 * 4
+            self.label_len = 24 * 4
+            self.pred_len = 24 * 4
+        else:
+            self.seq_len = size[0]
+            self.label_len = size[1]
+            self.pred_len = size[2]
+        # init
+        assert flag in ['train', 'test', 'val']
+        type_map = {'train': 0, 'val': 1, 'test': 2}
+        self.set_type = type_map[flag]
+
+        self.features = features
+        self.target = target
+        self.scale = scale
+        self.timeenc = timeenc
+        self.freq = freq
+
+        self.root_path = root_path
+        self.data_path = data_path
+        self.__read_data__()
+
+    def __read_data__(self):
+        self.scaler = StandardScaler()
+        df_raw = pd.read_csv(os.path.join(self.root_path,
+                                          self.data_path))
+
+        '''
+        df_raw.columns: ['date', ...(other features), target feature]
+        '''
+        cols = list(df_raw.columns)
+        cols.remove(self.target)
+        cols.remove('date')
+        df_raw = df_raw[['date'] + cols + [self.target]]
+        # print(cols)
+        num_train = int(len(df_raw) * 0.7)
+        num_test = int(len(df_raw) * 0.2)
+        num_vali = len(df_raw) - num_train - num_test
+        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
+        border2s = [num_train, num_train + num_vali, len(df_raw)]
+        border1 = border1s[self.set_type]
+        border2 = border2s[self.set_type]
+
+        if self.features == 'M' or self.features == 'MS':
+            cols_data = df_raw.columns[1:]
+            df_data = df_raw[cols_data]
+        elif self.features == 'S':
+            df_data = df_raw[[self.target]]
+
+        if self.scale:
+            train_data = df_data[border1s[0]:border2s[0]]
+            self.scaler.fit(train_data.values)
+            # print(self.scaler.mean_)
+            # exit()
+            data = self.scaler.transform(df_data.values)
+        else:
+            data = df_data.values
+
+        df_stamp = df_raw[['date']][border1:border2]
+        df_stamp['date'] = pd.to_datetime(df_stamp.date)
+        if self.timeenc == 0:
+            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
+            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
+            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
+            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
+            data_stamp = df_stamp.drop(['date'], axis=1).values
+        elif self.timeenc == 1:
+            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
+            data_stamp = data_stamp.transpose(1, 0)
+
+        self.data_x = data[border1:border2]
+        self.data_y = data[border1:border2]
+        self.data_stamp = data_stamp
+
+    def __getitem__(self, index):
+        s_begin = index
+        s_end = s_begin + self.seq_len
+        r_begin = s_end - self.label_len
+        r_end = r_begin + self.label_len + self.pred_len
+
+        seq_x = self.data_x[s_begin:s_end]
+        seq_y = self.data_y[r_begin:r_end]
+        seq_x_mark = self.data_stamp[s_begin:s_end]
+        seq_y_mark = self.data_stamp[r_begin:r_end]
+
+        return seq_x, seq_y, seq_x_mark, seq_y_mark
+
+    def __len__(self):
+        return len(self.data_x) - self.seq_len - self.pred_len + 1
+
+    def inverse_transform(self, data):
+        return self.scaler.inverse_transform(data)
+
+
+class Dataset_Pred(Dataset):
+    def __init__(self, root_path, flag='pred', size=None,
+                 features='S', data_path='ETTh1.csv',
+                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
+        # size [seq_len, label_len, pred_len]
+        # info
+        if size == None:
+            self.seq_len = 24 * 4 * 4
+            self.label_len = 24 * 4
+            self.pred_len = 24 * 4
+        else:
+            self.seq_len = size[0]
+            self.label_len = size[1]
+            self.pred_len = size[2]
+        # init
+        assert flag in ['pred']
+
+        self.features = features
+        self.target = target
+        self.scale = scale
+        self.inverse = inverse
+        self.timeenc = timeenc
+        self.freq = freq
+        self.cols = cols
+        self.root_path = root_path
+        self.data_path = data_path
+        self.__read_data__()
+
+    def __read_data__(self):
+        self.scaler = StandardScaler()
+        df_raw = pd.read_csv(os.path.join(self.root_path,
+                                          self.data_path))
+        '''
+        df_raw.columns: ['date', ...(other features), target feature]
+        '''
+        if self.cols:
+            cols = self.cols.copy()
+            cols.remove(self.target)
+        else:
+            cols = list(df_raw.columns)
+            cols.remove(self.target)
+            cols.remove('date')
+        df_raw = df_raw[['date'] + cols + [self.target]]
+        border1 = len(df_raw) - self.seq_len
+        border2 = len(df_raw)
+
+        if self.features == 'M' or self.features == 'MS':
+            cols_data = df_raw.columns[1:]
+            df_data = df_raw[cols_data]
+        elif self.features == 'S':
+            df_data = df_raw[[self.target]]
+
+        if self.scale:
+            self.scaler.fit(df_data.values)
+            data = self.scaler.transform(df_data.values)
+        else:
+            data = df_data.values
+
+        tmp_stamp = df_raw[['date']][border1:border2]
+        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
+        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)
+
+        df_stamp = pd.DataFrame(columns=['date'])
+        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
+        if self.timeenc == 0:
+            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
+            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
+            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
+            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
+            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
+            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
+            data_stamp = df_stamp.drop(['date'], axis=1).values
+        elif self.timeenc == 1:
+            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
+            data_stamp = data_stamp.transpose(1, 0)
+
+        self.data_x = data[border1:border2]
+        if self.inverse:
+            self.data_y = df_data.values[border1:border2]
+        else:
+            self.data_y = data[border1:border2]
+        self.data_stamp = data_stamp
+
+    def __getitem__(self, index):
+        s_begin = index
+        s_end = s_begin + self.seq_len
+        r_begin = s_end - self.label_len
+        r_end = r_begin + self.label_len + self.pred_len
+
+        seq_x = self.data_x[s_begin:s_end]
+        if self.inverse:
+            seq_y = self.data_x[r_begin:r_begin + self.label_len]
+        else:
+            seq_y = self.data_y[r_begin:r_begin + self.label_len]
+        seq_x_mark = self.data_stamp[s_begin:s_end]
+        seq_y_mark = self.data_stamp[r_begin:r_end]
+
+        return seq_x, seq_y, seq_x_mark, seq_y_mark
+
+    def __len__(self):
+        return len(self.data_x) - self.seq_len + 1
+
+    def inverse_transform(self, data):
+        return self.scaler.inverse_transform(data)
+
+
+class Dataset_Neuro(Dataset):
+    def __init__(self, root_path, flag='train', size=None,
+                 features='S', data_path='ETTh1.csv',
+                 target='OT', scale=True, timeenc=0, freq='h'):
+        # size [seq_len, label_len, pred_len]
+        # info
+        if size == None:
+            self.seq_len = 24 * 4 * 4
+            self.label_len = 24 * 4
+            self.pred_len = 24 * 4
+        else:
+            self.seq_len = size[0]
+            self.label_len = size[1]
+            self.pred_len = size[2]
+        # init
+        assert flag in ['train', 'test', 'val']
+        type_map = {'train': 0, 'val': 1, 'test': 2}
+        self.set_type = type_map[flag]
+
+        self.features = features
+        self.target = target
+        self.scale = scale
+        self.timeenc = timeenc
+        self.freq = freq
+
+        self.root_path = root_path
+        self.data_path = data_path
+        self.__read_data__()
+
+    def __read_data__(self):
+        self.scaler = StandardScaler()
+        train_data = np.load(os.path.join(self.root_path, 'train_data.npy'))
+        val_data = np.load(os.path.join(self.root_path, 'val_data.npy'))
+        test_data = np.load(os.path.join(self.root_path, 'test_data.npy'))
+
+        train_sensors_last = np.swapaxes(train_data, 1, 2)
+        val_sensors_last = np.swapaxes(val_data, 1, 2)
+        test_sensors_last = np.swapaxes(test_data, 1, 2)
+
+        train_data_reshaped = train_sensors_last.reshape(-1, train_sensors_last.shape[-1])
+        val_data_reshaped = val_sensors_last.reshape(-1, val_sensors_last.shape[-1])
+        test_data_reshaped = test_sensors_last.reshape(-1, test_sensors_last.shape[-1])
+
+        if self.scale:
+            self.scaler.fit(train_data_reshaped)
+            train_data_scaled = self.scaler.transform(train_data_reshaped)
+            val_data_scaled = self.scaler.transform(val_data_reshaped)
+            test_data_scaled = self.scaler.transform(test_data_reshaped)
+
+        train_scaled_orig_shape = train_data_scaled.reshape(train_sensors_last.shape)
+        val_scaled_orig_shape = val_data_scaled.reshape(val_sensors_last.shape)
+        test_scaled_orig_shape = test_data_scaled.reshape(test_sensors_last.shape)
+
+        if self.set_type == 0:  # TRAIN
+            train_x, train_y = self.make_full_x_y_data(train_scaled_orig_shape)
+            self.data_x = train_x
+            self.data_y = train_y
+
+        elif self.set_type == 1:  # VAL
+            val_x, val_y = self.make_full_x_y_data(val_scaled_orig_shape)
+            self.data_x = val_x
+            self.data_y = val_y
+
+        elif self.set_type == 2:  # TEST
+            test_x, test_y = self.make_full_x_y_data(test_scaled_orig_shape)
+            self.data_x = test_x
+            self.data_y = test_y
+
+    def make_full_x_y_data(self, array):
+        data_x = []
+        data_y = []
+        for instance in range(0, array.shape[0]):
+            for time in range(0, array.shape[1]):
+                s_begin = time
+                s_end = s_begin + self.seq_len
+                r_begin = s_end - self.label_len
+                r_end = r_begin + self.label_len + self.pred_len
+                if r_end <= array.shape[1]:
+                    data_x.append(array[instance, s_begin:s_end, :])
+                    data_y.append(array[instance, r_begin:r_end, :])
+                else:
+                    break
+        return data_x, data_y
+
+    def __getitem__(self, index):
+        return self.data_x[index], self.data_y[index], 0, 0
+
+    def __len__(self):
+        return len(self.data_x)
+
+    def inverse_transform(self, data):
+        print('DATLOADER INVERSE_TRANSFORM - this might not do what you want it to anymore')
+        return self.scaler.inverse_transform(data)
+
+
+class Dataset_Earthquake(Dataset):
+    def __init__(self, root_path, flag='train', size=None,
+                 features='S', data_path='ETTh1.csv',
+                 target='OT', scale=True, timeenc=0, freq='h'):
+        # size [seq_len, label_len, pred_len]
+        # info
+        if size == None:
+            self.seq_len = 24 * 4 * 4
+            self.label_len = 24 * 4
+            self.pred_len = 24 * 4
+        else:
+            self.seq_len = size[0]
+            self.label_len = size[1]
+            self.pred_len = size[2]
+        # init
+        assert flag in ['train', 'test', 'val']
+        type_map = {'train': 0, 'val': 1, 'test': 2}
+        self.set_type = type_map[flag]
+
+        self.features = features
+        self.target = target
+        self.scale = scale
+        self.timeenc = timeenc
+        self.freq = freq
+
+        self.root_path = root_path
+        self.data_path = data_path
+        self.__read_data__()
+
+    def __read_data__(self):
+        self.scaler = StandardScaler()
+        train_data = np.load(os.path.join(self.root_path, 'train_data.npy'))
+        val_data = np.load(os.path.join(self.root_path, 'val_data.npy'))
+        test_data = np.load(os.path.join(self.root_path, 'test_data.npy'))
+
+        train_sensors_last = np.swapaxes(train_data, 1, 2)
+        val_sensors_last = np.swapaxes(val_data, 1, 2)
+        test_sensors_last = np.swapaxes(test_data, 1, 2)
+
+        train_data_reshaped = train_sensors_last.reshape(-1, train_sensors_last.shape[-1])
+        val_data_reshaped = val_sensors_last.reshape(-1, val_sensors_last.shape[-1])
+        test_data_reshaped = test_sensors_last.reshape(-1, test_sensors_last.shape[-1])
+
+        if self.scale:
+            self.scaler.fit(train_data_reshaped)
+            train_data_scaled = self.scaler.transform(train_data_reshaped)
+            val_data_scaled = self.scaler.transform(val_data_reshaped)
+            test_data_scaled = self.scaler.transform(test_data_reshaped)
+
+        train_scaled_orig_shape = train_data_scaled.reshape(train_sensors_last.shape)
+        val_scaled_orig_shape = val_data_scaled.reshape(val_sensors_last.shape)
+        test_scaled_orig_shape = test_data_scaled.reshape(test_sensors_last.shape)
+
+        if self.set_type == 0:  # TRAIN
+            train_x, train_y = self.make_full_x_y_data(train_scaled_orig_shape)
+            self.data_x = train_x
+            self.data_y = train_y
+
+        elif self.set_type == 1:  # VAL
+            val_x, val_y = self.make_full_x_y_data(val_scaled_orig_shape)
+            self.data_x = val_x
+            self.data_y = val_y
+
+        elif self.set_type == 2:  # TEST
+            test_x, test_y = self.make_full_x_y_data(test_scaled_orig_shape)
+            self.data_x = test_x
+            self.data_y = test_y
+
+        print(self.set_type, len(self.data_x), len(self.data_y), self.data_x[0].shape, self.data_y[0].shape)
+
+    def make_full_x_y_data(self, array):
+        data_x = []
+        data_y = []
+        for instance in range(0, array.shape[0]):
+            for time in range(0, array.shape[1]):
+                s_begin = time
+                s_end = s_begin + self.seq_len
+                r_begin = s_end - self.label_len
+                r_end = r_begin + self.label_len + self.pred_len
+                if r_end <= array.shape[1]:
+                    data_x.append(array[instance, s_begin:s_end, :])
+                    data_y.append(array[instance, r_begin:r_end, :])
+                else:
+                    break
+        return data_x, data_y
+
+    def __getitem__(self, index):
+        return self.data_x[index], self.data_y[index], 0, 0
+
+    def __len__(self):
+        return len(self.data_x)
+
+    def inverse_transform(self, data):
+        print('DATLOADER INVERSE_TRANSFORM - this might not do what you want it to anymore')
+        return self.scaler.inverse_transform(data)
+    
+class Dataset_EEG(Dataset):
+    def __init__(self, root_path, flag='train', size=None,
+                 features=None, data_path='ETTh1.csv',
+                 target='OT', scale=True, timeenc=0, freq='h'):
+        # size [seq_len, label_len, pred_len]
+        # info
+        if size == None:
+            self.seq_len = 24 * 4 * 4
+            self.label_len = 24 * 4
+            self.pred_len = 24 * 4
+        else:
+            self.seq_len = size[0]
+            self.label_len = size[1]
+            self.pred_len = size[2]
+        # init
+        assert flag in ['train', 'test', 'val']
+        type_map = {'train': 0, 'val': 1, 'test': 2}
+        self.set_type = type_map[flag]
+
+        self.features = features
+        self.target = target
+        self.step_size = 1
+        if (self.target == 'OT') or (self.target.lower() == 'half'):
+            self.step_size = self.seq_len // 2
+        elif self.target == "1": 
+            self.step_size = 1
+        else: 
+            self.step_size = int(self.target)
+        self.scale = scale
+        self.timeenc = timeenc
+        self.freq = freq
+        self.event_dict = {'Up': 1, 'Down': 2, 'Left': 3, 'Right': 4, 'Rest': 6}
+
+        self.root_path = root_path
+        self.data_path = data_path
+        self.data_name = Path(data_path).stem 
+
+        self.__read_data__()
+
+    def __read_data__(self):
+        self.scaler = StandardScaler()
+        df_raw = pd.read_csv(os.path.join(self.root_path,
+                                            self.data_path), index_col=0)
+        df_split = pd.read_csv(os.path.join(self.root_path, 
+                                            self.data_name + "-split.csv"), index_col=0)
+        '''
+        df_raw.columns: ['date', ...(other features), target feature]
+        '''
+
+        self.eeg_columns = df_raw.columns[:128]
+
+        if self.features == 'clean_data':
+            # # Remove columns with indices 2 and 4 from self.eeg_columns
+            channels_to_remove = np.genfromtxt(os.path.join(self.root_path, 
+                                            self.data_name + "-bad_channels.csv"), delimiter=',')
+            filtered_columns = [col for idx, col in enumerate(self.eeg_columns) if idx not in channels_to_remove]
+
+            # Update self.eeg_columns with the filtered columns
+            self.eeg_columns = filtered_columns
+
+
+
+        
+        if self.scale:
+            self.scaler.fit(self.make_contiguous_x_data(df_raw, df_split, split='train')) 
+            all_x_scaled = self.scaler.transform(df_raw.loc[:, self.eeg_columns])
+            df_raw.loc[:, self.eeg_columns] = all_x_scaled
+
+        if self.set_type == 0:  # TRAIN
+            train_x, train_y, train_x_label, train_y_label = self.make_full_x_y_data(df_raw, df_split, split='train')
+            self.data_x = train_x
+            self.data_y = train_y
+            self.data_x_label = train_x_label
+            self.data_y_label = train_y_label
+
+        elif self.set_type == 1:  # VAL
+            val_x, val_y, val_x_label, val_y_label = self.make_full_x_y_data(df_raw, df_split, split='val')
+            self.data_x = val_x
+            self.data_y = val_y
+            self.data_x_label = val_x_label
+            self.data_y_label = val_y_label
+
+        elif self.set_type == 2:  # TEST
+            test_x, test_y, test_x_label, test_y_label = self.make_full_x_y_data(df_raw, df_split, split='test')
+            self.data_x = test_x
+            self.data_y = test_y
+            self.data_x_label = test_x_label
+            self.data_y_label = test_y_label
+
+    def make_contiguous_x_data(self, df_raw, df_split, split):         
+        data_x = []
+        for trial_start_ind, r in df_split[df_split['split'] == split].iterrows():
+            trial_end_ind = df_raw.loc[trial_start_ind:][df_raw.loc[trial_start_ind:, 'STI'] == self.event_dict['Rest']].iloc[0].name
+            data_x.append(df_raw.loc[trial_start_ind:trial_end_ind, self.eeg_columns].values)
+        return np.concatenate(data_x)
+            
+    def make_full_x_y_data(self, df_raw, df_split, split):
+        data_x = []
+        data_y = []
+        data_x_label = []
+        data_y_label = []
+        counter = 0
+        for trial_start_ind, r in df_split[df_split['split'] == split].iterrows():
+            trial_end_ind = df_raw.loc[trial_start_ind:][df_raw.loc[trial_start_ind:, 'STI'] == self.event_dict['Rest']].iloc[0].name
+            for time in range(trial_start_ind, trial_end_ind, self.step_size):
+                counter += 1
+                s_begin = time
+                s_end = s_begin + self.seq_len
+                r_begin = s_end - self.label_len
+                r_end = r_begin + self.label_len + self.pred_len
+                if r_end <= trial_end_ind:
+                    data_x.append(df_raw.loc[s_begin:s_end-1, self.eeg_columns].values)
+                    data_y.append(df_raw.loc[r_begin:r_end-1, self.eeg_columns].values)
+                    data_x_label.append(int(r['STI']) - 1)
+                    data_y_label.append(int(r['STI']) - 1) ## These labels are 1 indexed in the original files
+                else:
+                    break
+        return data_x, data_y, data_x_label, data_y_label
+    
+    def __getitem__(self, index):
+        return self.data_x[index], self.data_y[index], self.data_x_label[index], self.data_y_label[index]
+
+    def __len__(self):
+        return len(self.data_x)
+
+    def inverse_transform(self, data):
+        print('DATLOADER INVERSE_TRANSFORM - this might not do what you want it to anymore')
+        return self.scaler.inverse_transform(data)
```



- **Added** `env.yml`

  - **Diff:**

  ```diff

+channels:
+  - defaults
+dependencies:
+  - pandas=1.4.2
+  - matplotlib
+  - seaborn
+  - scipy
+  - numpy
+  - python=3.9
+  - scikit-learn
+  - jupyterlab
+  - pytorch
+  - h5py
+  - imbalanced-learn
+  - pip
+  - pip:
+    - omegaconf
+    - hydra-core
+    - comet-ml
+
```



- **Added** `lib/models/__init__.py`

  - **Affected functions:**

    - `get_model_class`

  - **Diff:**

  ```diff

+from .vqvae import vqvae
+
+model_dict = {
+    'vqvae': vqvae
+}
+
+def get_model_class(model_name):
+    if model_name in model_dict:
+        return model_dict[model_name]
+    else:
+        raise NotImplementedError
```



- **Added** `lib/models/classif.py`

  - **Affected functions:**

    - `Lout`

  - **Diff:**

  ```diff

+import torch
+import torch.nn as nn
+import torch.nn.functional as F
+import math
+import pdb
+# from torch.nn import TransformerEncoder, TransformerEncoderLayer
+from .transformer import TransformerEncoder, TransformerEncoderLayer
+
+class PositionalEncoding(nn.Module):
+    def __init__(self, emb_size: int, dropout: float = 0.0, maxlen: int = 5000):
+        super(PositionalEncoding, self).__init__()
+        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
+        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
+        pos_embedding = torch.zeros((maxlen, emb_size))
+        pos_embedding[:, 0::2] = torch.sin(pos * den)
+        pos_embedding[:, 1::2] = torch.cos(pos * den)
+        pos_embedding = pos_embedding.unsqueeze(-2)
+
+        self.dropout = nn.Dropout(dropout)
+        self.register_buffer("pos_embedding", pos_embedding)
+
+    def forward(self, token_embedding: torch.Tensor):
+        return self.dropout(
+            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
+        )
+
+
+class ScaleEncoding(nn.Module):
+    def __init__(self, emb_size: int, dropout: float = 0.0):
+        super(ScaleEncoding, self).__init__()
+
+        self.proj = nn.Linear(2, emb_size)
+        self.dropout = nn.Dropout(dropout)
+
+    def forward(self, x: torch.Tensor, scale: torch.Tensor):
+        """
+        Args:
+            x: (S, B, d_model)
+            scale: (S, B, 2)
+        """
+        return self.dropout(x + self.proj(scale))
+
+
+class TimeEncoder(nn.Module):
+    def __init__(
+        self,
+        d_in: int,
+        d_model: int,
+        nhead: int,
+        d_hid: int,
+        nlayers: int,
+        seq_len: int = 5000,
+        dropout: float = 0.0,
+        batch_first: bool = False,
+        norm_first: bool = False,
+        return_weights: bool = False, 
+    ):
+        super(TimeEncoder, self).__init__()
+        self.model_type = "Time"
+        self.d_model = d_model
+        self.return_weights = return_weights
+
+        self.has_linear_in = d_in != d_model
+        if self.has_linear_in:
+            self.linear_in = nn.Linear(d_in, d_model)
+
+        self.pos_encoder = PositionalEncoding(d_model, dropout, seq_len + 1)
+
+        encoder_layers = TransformerEncoderLayer(
+            d_model,
+            nhead,
+            dim_feedforward=d_hid,
+            dropout=dropout,
+            batch_first=batch_first,
+            norm_first=norm_first,
+        )
+        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
+
+        # self._reset_parameters()
+        self.apply(self._init_weights)
+
+    def _reset_parameters(self):
+        """Initiate parameters in the transformer model."""
+        for p in self.parameters():
+            if p.dim() > 1:
+                nn.init.xavier_uniform_(p)
+
+    def _init_weights(self, m):
+        if isinstance(m, nn.Linear):
+            nn.init.trunc_normal_(m.weight, std=0.02)
+            if isinstance(m, nn.Linear) and m.bias is not None:
+                nn.init.constant_(m.bias, 0)
+        elif isinstance(m, nn.LayerNorm):
+            nn.init.constant_(m.bias, 0)
+            nn.init.constant_(m.weight, 1.0)
+
+    def forward(self, x: torch.Tensor) -> torch.Tensor:
+        """
+        Arguments:
+            x: tensor of shape (seq_len, batch_size, d_in)
+        Returns:
+            y: tensor of shape (batch_size, d_model)
+        """
+        if self.has_linear_in:
+            x = self.linear_in(x)
+        # pos encoder
+        x = self.pos_encoder(x)
+        x, weights = self.transformer_encoder(x)  # (seq_len, batch, d_model)
+
+        x = self.pos_encoder(x)
+        x = x.mean(dim=0)  # (batch, d_model)
+
+        if self.return_weights: 
+            return x, weights
+        else: 
+            return x
+
+
+class SensorEncoder(nn.Module):
+    def __init__(
+        self,
+        d_in: int,
+        d_model: int,
+        nhead: int,
+        d_hid: int,
+        nlayers: int,
+        seq_len: int = 5000,
+        dropout: float = 0.0,
+        batch_first: bool = False,
+        norm_first: bool = False,
+        return_weights: bool = False, 
+        scale: bool = True
+    ):
+        super(SensorEncoder, self).__init__()
+        self.model_type = "Sensor"
+        self.d_model = d_model
+        self.return_weights = return_weights
+        self.scale = scale
+
+        self.has_linear_in = d_in != d_model
+        if self.has_linear_in:
+            self.linear_in = nn.Linear(d_in, d_model)
+
+        self.pos_encoder = PositionalEncoding(d_model, dropout, seq_len)
+        if self.scale:
+            self.scale_encoder = ScaleEncoding(d_model, dropout)
+
+        encoder_layers = TransformerEncoderLayer(
+            d_model,
+            nhead,
+            dim_feedforward=d_hid,
+            dropout=dropout,
+            batch_first=batch_first,
+            norm_first=norm_first,
+        )
+        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
+
+        # TODO implementing Georgia change
+        self.proj = nn.Linear(d_model, d_model)
+
+        # self._reset_parameters()
+        self.apply(self._init_weights)
+
+    def _reset_parameters(self):
+        """Initiate parameters in the transformer model."""
+        for p in self.parameters():
+            if p.dim() > 1:
+                nn.init.xavier_uniform_(p)
+
+    def _init_weights(self, m):
+        if isinstance(m, nn.Linear):
+            nn.init.trunc_normal_(m.weight, std=0.02)
+            if isinstance(m, nn.Linear) and m.bias is not None:
+                nn.init.constant_(m.bias, 0)
+        elif isinstance(m, nn.LayerNorm):
+            nn.init.constant_(m.bias, 0)
+            nn.init.constant_(m.weight, 1.0)
+
+    def forward(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
+        """
+        Arguments:
+            x: tensor of shape (seq_len, batch_size, d_in)
+            scale: tensor of shape (seq_len, batch_size, 2)
+        Returns:
+            y: tensor of shape (batch_size, seq_out_len)
+        """
+
+        if self.has_linear_in:
+            x = self.linear_in(x)
+        # scale encoding
+        if self.scale: 
+            x = self.scale_encoder(x, scale)
+        # pos encoder
+        x = self.pos_encoder(x)
+        x, weights = self.transformer_encoder(x)  # (seq_len, batch, d_model)
+
+        x = torch.permute(x, (1, 0, 2))  # (batch, seq_len, d_model)
+        # TODO adding change Georgia suggestion
+        x = x.mean(dim=1)
+        # x = x.flatten(start_dim=1)  # (batch, seq_len * d_model)
+        x = self.proj(x)
+
+        if self.return_weights: 
+            return x, weights
+        else: 
+            return x
+
+
+class SensorTimeEncoder(nn.Module):
+    def __init__(
+        self,
+        d_in: int,
+        d_model: int, 
+        nheadt: int,
+        nheads: int,
+        d_hid: int,
+        nlayerst: int,
+        nlayerss: int,
+        seq_lent: int = 5000,
+        seq_lens: int = 5000,
+        dropout: float = 0.0,
+        d_out: int = 1,
+        return_weights: bool = False, 
+        scale: bool = True
+    ):
+        super(SensorTimeEncoder, self).__init__()
+        self.model_type = "SensorTime"
+        self.return_weights = return_weights
+
+        self.timeenc = TimeEncoder(
+            d_in=d_in,
+            d_model=d_model,
+            nhead=nheadt,
+            d_hid=d_hid,
+            nlayers=nlayerst,
+            seq_len=seq_lent,
+            dropout=dropout,
+            return_weights=return_weights, 
+        )
+
+        self.senorenc = SensorEncoder(
+            d_in=d_model,
+            d_model=d_model,
+            nhead=nheads,
+            d_hid=d_hid,
+            nlayers=nlayerss,
+            seq_len=seq_lens,
+            dropout=dropout,
+            return_weights=return_weights, 
+            scale=scale
+        )
+
+        self.classifier = nn.Linear(d_model, d_out)
+
+    def forward(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
+        """
+        Args:
+            x: tensor of shape (B, T, S, d_in)
+            scale: tensor of shape (B, S, 2)
+        Returns:
+            y: tensor of shape (B, 1)
+        """
+
+        B, T, S, dim = x.shape
+        x = torch.permute(x, (1, 0, 2, 3))  # (T, B, S, d_in)
+
+        # prepare input for time encoder
+        x = x.flatten(start_dim=1, end_dim=2)  # (T, B * S, d_in)
+        if self.return_weights:
+            y, time_weights = self.timeenc(x)  # (B * S, d_model)
+        else: 
+            y = self.timeenc(x)  # (B * S, d_model)
+
+        # prepare input to sensor encoder
+        y = y.reshape(B, S, y.shape[-1])  # (B, S, d_model)
+        y = torch.permute(y, (1, 0, 2))  # (S, B, d_model)
+        scale = torch.permute(scale, (1, 0, 2))  # (S, B, 2)
+        if self.return_weights: 
+            z, sensor_weights = self.senorenc(y, scale)  # (B, d_model)
+        else: 
+            z = self.senorenc(y, scale)  # (B, d_model)
+
+        z = self.classifier(z)
+
+        if self.return_weights: 
+            return z, time_weights, sensor_weights
+        else: 
+            return z
+
+
+def Lout(Lin, kernel, stride=1, padding=0, dilation=1):
+    """
+    Returns the length of the tensor after a conv layer
+    From: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
+    """
+    return math.floor((Lin + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)
+
+
+class SimpleMLP(nn.Module):
+    def __init__(self, in_dim, out_dim, hidden_dims, dropout=0.0):
+        super(SimpleMLP, self).__init__()
+
+        self.nlayers = len(hidden_dims)
+
+        layers = []
+        dim = in_dim
+        for i in range(self.nlayers):
+            layer = nn.Linear(dim, hidden_dims[i])
+            layers.append(layer)
+            dim = hidden_dims[i]
+        self.fcs = nn.ModuleList(layers)
+        self.fc_out = nn.Linear(dim, out_dim)
+        self.dropout = nn.Dropout(dropout)
+
+        self._reset_parameters()
+
+    def _reset_parameters(self):
+        """Initiate parameters in the transformer model."""
+        for p in self.parameters():
+            if p.dim() > 1:
+                nn.init.xavier_uniform_(p)
+
+    def forward(self, x):
+        for fc in self.fcs:
+            x = F.relu(fc(x))
+            x = self.dropout(x)
+        x = self.fc_out(x)
+
+        return x
+
+
+# Source: https://github.com/torcheeg/torcheeg/blob/v1.1.0/torcheeg/models/cnn/eegnet.py#L15-L126
+class Conv2dWithConstraint(nn.Conv2d):
+    def __init__(self, *args, max_norm: int = 1, **kwargs):
+        self.max_norm = max_norm
+        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)
+
+    def forward(self, x: torch.Tensor) -> torch.Tensor:
+        self.weight.data = torch.renorm(
+            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
+        )
+        return super(Conv2dWithConstraint, self).forward(x)
+
+
+class EEGNet(nn.Module):
+    """
+    Args:
+        chunk_size (int): T
+        num_electrodes (int): S
+        F1 (int): The filter number of block 1,
+        F2 (int): The filter number of block 2
+        D (int): The depth multiplier (number of spatial filters)
+        num_classes (int): The number of classes to predict
+        kernel_1 (int): The filter size of block 1
+        kernel_2 (int): The filter size of block 2
+        dropout (float): probability of dropout
+    """
+
+    def __init__(
+        self,
+        chunk_size: int = 151,
+        num_electrodes: int = 60,
+        F1: int = 8,
+        F2: int = 16,
+        D: int = 2,
+        kernel_1: int = 64,
+        kernel_2: int = 16,
+        dropout: float = 0.25,
+        num_classes: int = 1, 
+    ):
+        super(EEGNet, self).__init__()
+        self.F1 = F1
+        self.F2 = F2
+        self.D = D
+        self.chunk_size = chunk_size
+        self.num_electrodes = num_electrodes
+        self.kernel_1 = kernel_1
+        self.kernel_2 = kernel_2
+        self.dropout = dropout
+        self.num_classes = num_classes
+
+        self.block1 = nn.Sequential(
+            nn.Conv2d(
+                1,
+                self.F1,
+                (1, self.kernel_1),
+                stride=1,
+                padding=(0, self.kernel_1 // 2),
+                bias=False,
+            ),
+            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
+            Conv2dWithConstraint(
+                self.F1,
+                self.F1 * self.D,
+                (self.num_electrodes, 1),
+                max_norm=1,
+                stride=1,
+                padding=(0, 0),
+                groups=self.F1,
+                bias=False,
+            ),
+            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
+            nn.ELU(),
+            nn.AvgPool2d((1, 4), stride=4),
+            nn.Dropout(p=dropout),
+        )
+
+        self.block2 = nn.Sequential(
+            nn.Conv2d(
+                self.F1 * self.D,
+                self.F1 * self.D,
+                (1, self.kernel_2),
+                stride=1,
+                padding=(0, self.kernel_2 // 2),
+                bias=False,
+                groups=self.F1 * self.D,
+            ),
+            nn.Conv2d(
+                self.F1 * self.D,
+                self.F2,
+                1,
+                padding=(0, 0),
+                groups=1,
+                bias=False,
+                stride=1,
+            ),
+            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
+            nn.ELU(),
+            nn.AvgPool2d((1, 8), stride=8),
+            nn.Dropout(p=dropout),
+        )
+
+        self.lin = nn.Linear(self.feature_dim(), self.num_classes, bias=False)
+
+    def feature_dim(self):
+        with torch.no_grad():
+            mock_eeg = torch.zeros(1, 1, self.num_electrodes, self.chunk_size)
+
+            mock_eeg = self.block1(mock_eeg)
+            mock_eeg = self.block2(mock_eeg)
+
+        return self.F2 * mock_eeg.shape[3]
+
+    def forward(self, x: torch.Tensor) -> torch.Tensor:
+        r"""
+        Args:
+            x (torch.Tensor): (B, 1, S, T)
+        """
+        x = self.block1(x)
+        x = self.block2(x)
+        x = x.flatten(start_dim=1)
+        x = self.lin(x)
+
+        return x
```



- **Added** `lib/models/core.py`

  - **Diff:**

  ```diff

+import torch
+import torch.nn as nn
+
+from abc import ABC, abstractmethod
+
+
+class BaseModel(nn.Module, ABC):
+    def __init__(self):
+        super().__init__()
+
+    @abstractmethod
+    def shared_eval(self, batch, optimizer, mode, comet_logger='None'):
+        pass
+
+    # TODO change this to have an exponentially decaying learning rate
+    def configure_optimizers(self, lr=1e-3):
+        # optimizer = torch.optim.AdamW(self.parameters(), lr=lr)  # adds weight decay
+        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
+        # optimizer = torch.optim.SGD(self.parameters(), lr=lr)
+        return optimizer
```



- **Added** `lib/models/revin.py`

  - **Diff:**

  ```diff

+# code from https://github.com/ts-kim/RevIN, with minor modifications
+
+import torch
+import torch.nn as nn
+
+
+class RevIN(nn.Module):
+    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
+        """
+        :param num_features: the number of features or channels
+        :param eps: a value added for numerical stability
+        :param affine: if True, RevIN has learnable affine parameters
+        """
+        super(RevIN, self).__init__()
+        self.num_features = num_features
+        self.eps = eps
+        self.affine = affine
+        self.subtract_last = subtract_last
+        if self.affine:
+            self._init_params()
+
+    def forward(self, x, mode: str):
+        if mode == "norm":
+            self._get_statistics(x)
+            x = self._normalize(x)
+        elif mode == "denorm":
+            x = self._denormalize(x)
+        else:
+            raise NotImplementedError
+        return x
+
+    def _init_params(self):
+        # initialize RevIN params: (C,)
+        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
+        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))
+
+    def _get_statistics(self, x):
+        dim2reduce = tuple(range(1, x.ndim - 1))
+        if self.subtract_last:
+            self.last = x[:, -1, :].unsqueeze(1)
+        else:
+            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
+        self.stdev = torch.sqrt(
+            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
+        ).detach()
+
+    def _normalize(self, x):
+        if self.subtract_last:
+            x = x - self.last
+        else:
+            x = x - self.mean
+        x = x / self.stdev
+        if self.affine:
+            x = x * self.affine_weight
+            x = x + self.affine_bias
+        return x
+
+    def _denormalize(self, x):
+        if self.affine:
+            x = x - self.affine_bias
+            x = x / (self.affine_weight + self.eps * self.eps)
+        x = x * self.stdev
+        if self.subtract_last:
+            x = x + self.last
+        else:
+            x = x + self.mean
+        return x
```



- **Added** `lib/models/transformer.py`

  - **Affected functions:**

    - `_get_activation_fn`

    - `_get_clones`

  - **Diff:**

  ```diff

+
+
+## A simpler implementation of the nnTransformer that allows for attention weights to be returned
+# From https://buomsoo-kim.github.io/attention/2020/04/27/Attention-mechanism-21.md/  
+import torch.nn.functional as F
+import torch.nn as nn
+
+import copy
+
+
+def _get_activation_fn(activation):
+    if activation == "relu":
+        return F.relu
+    elif activation == "gelu":
+        return F.gelu
+
+    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
+
+
+class TransformerEncoderLayer(nn.Module):
+    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", batch_first=False, norm_first=False):
+        ## norm_first=False works, but norm_first=True does not work
+        super(TransformerEncoderLayer, self).__init__()
+        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
+        # Implementation of Feedforward model
+        self.linear1 = nn.Linear(d_model, dim_feedforward)
+        self.dropout = nn.Dropout(dropout)
+        self.linear2 = nn.Linear(dim_feedforward, d_model)
+
+        self.norm1 = nn.LayerNorm(d_model)
+        self.norm2 = nn.LayerNorm(d_model)
+        self.dropout1 = nn.Dropout(dropout)
+        self.dropout2 = nn.Dropout(dropout)
+
+        self.activation = _get_activation_fn(activation)
+        
+    def __setstate__(self, state):
+        if 'activation' not in state:
+            state['activation'] = F.relu
+        super(TransformerEncoderLayer, self).__setstate__(state)
+
+    def forward(self, src, src_mask=None, src_key_padding_mask=None):
+        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
+        src2, weights = self.self_attn(src, src, src, attn_mask=src_mask,
+                              key_padding_mask=src_key_padding_mask, average_attn_weights=False)
+        src = src + self.dropout1(src2)
+        src = self.norm1(src)
+        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
+        src = src + self.dropout2(src2)
+        src = self.norm2(src)
+        return src, weights
+    
+
+def _get_clones(module, N):
+    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
+
+class TransformerEncoder(nn.Module):
+    __constants__ = ['norm']
+    def __init__(self, encoder_layer, num_layers, norm=None):
+        super(TransformerEncoder, self).__init__()
+        self.layers = _get_clones(encoder_layer, num_layers)
+        self.num_layers = num_layers
+        self.norm = norm
+
+    def forward(self, src, mask=None, src_key_padding_mask=None):
+        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
+        output = src
+        weights = []
+        for mod in self.layers:
+            output, weight = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
+            weights.append(weight)
+
+        if self.norm is not None:
+            output = self.norm(output)
+        return output, weights
```



- **Added** `lib/models/vqvae.py`

  - **Diff:**

  ```diff

+import pdb
+import torch
+import torch.nn as nn
+import torch.nn.functional as F
+
+from .core import BaseModel
+
+
+class Residual(nn.Module):
+    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
+        super(Residual, self).__init__()
+        # TODO original with padding
+        self._block = nn.Sequential(
+            nn.ReLU(True),
+            nn.Conv1d(in_channels=in_channels,
+                      out_channels=num_residual_hiddens,
+                      kernel_size=3, stride=1, padding=1, bias=False),
+            nn.ReLU(True),
+            nn.Conv1d(in_channels=num_residual_hiddens,
+                      out_channels=num_hiddens,
+                      kernel_size=1, stride=1, bias=False)
+        )
+
+    def forward(self, x):
+        return x + self._block(x)
+
+
+class ResidualStack(nn.Module):
+    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
+        super(ResidualStack, self).__init__()
+        self._num_residual_layers = num_residual_layers
+        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
+                                      for _ in range(self._num_residual_layers)])
+
+    def forward(self, x):
+        for i in range(self._num_residual_layers):
+            x = self._layers[i](x)
+        return F.relu(x)
+
+
+class Encoder(nn.Module):
+    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim, compression_factor):
+        super(Encoder, self).__init__()
+        if compression_factor == 4:
+            # TODO original
+            self._conv_1 = nn.Conv1d(in_channels=in_channels,
+                                     out_channels=num_hiddens // 2,
+                                     kernel_size=4,
+                                     stride=2, padding=1)
+            self._conv_2 = nn.Conv1d(in_channels=num_hiddens // 2,
+                                     out_channels=num_hiddens,
+                                     kernel_size=4,
+                                     stride=2, padding=1)
+            self._conv_3 = nn.Conv1d(in_channels=num_hiddens,
+                                     out_channels=num_hiddens,
+                                     kernel_size=3,
+                                     stride=1, padding=1)
+            self._residual_stack = ResidualStack(in_channels=num_hiddens,
+                                                 num_hiddens=num_hiddens,
+                                                 num_residual_layers=num_residual_layers,
+                                                 num_residual_hiddens=num_residual_hiddens)
+
+            self._pre_vq_conv = nn.Conv1d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)
+
+        elif compression_factor == 8:
+            self._conv_1 = nn.Conv1d(in_channels=in_channels,
+                                     out_channels=num_hiddens // 2,
+                                     kernel_size=4,
+                                     stride=2, padding=1)
+            self._conv_2 = nn.Conv1d(in_channels=num_hiddens // 2,
+                                     out_channels=num_hiddens,
+                                     kernel_size=4,
+                                     stride=2, padding=1)
+            self._conv_A = nn.Conv1d(in_channels=num_hiddens,
+                                     out_channels=num_hiddens,
+                                     kernel_size=4,
+                                     stride=2, padding=1)
+            self._conv_3 = nn.Conv1d(in_channels=num_hiddens,
+                                     out_channels=num_hiddens,
+                                     kernel_size=3,
+                                     stride=1, padding=1)
+            self._residual_stack = ResidualStack(in_channels=num_hiddens,
+                                                 num_hiddens=num_hiddens,
+                                                 num_residual_layers=num_residual_layers,
+                                                 num_residual_hiddens=num_residual_hiddens)
+
+            self._pre_vq_conv = nn.Conv1d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)
+
+        elif compression_factor == 12:
+            self._conv_1 = nn.Conv1d(in_channels=in_channels,
+                                     out_channels=num_hiddens // 2,
+                                     kernel_size=4,
+                                     stride=2, padding=1)
+            self._conv_2 = nn.Conv1d(in_channels=num_hiddens // 2,
+                                     out_channels=num_hiddens,
+                                     kernel_size=4,
+                                     stride=2, padding=1)
+            self._conv_3 = nn.Conv1d(in_channels=num_hiddens,
+                                     out_channels=num_hiddens,
+                                     kernel_size=4,
+                                     stride=3, padding=1)
+            self._conv_4 = nn.Conv1d(in_channels=num_hiddens,
+                                     out_channels=num_hiddens,
+                                     kernel_size=3,
+                                     stride=1, padding=1)
+            self._residual_stack = ResidualStack(in_channels=num_hiddens,
+                                                 num_hiddens=num_hiddens,
+                                                 num_residual_layers=num_residual_layers,
+                                                 num_residual_hiddens=num_residual_hiddens)
+
+            self._pre_vq_conv = nn.Conv1d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)
+
+        elif compression_factor == 16:
+            self._conv_1 = nn.Conv1d(in_channels=in_channels,
+                                     out_channels=num_hiddens // 2,
+                                     kernel_size=4,
+                                     stride=2, padding=1)
+            self._conv_2 = nn.Conv1d(in_channels=num_hiddens // 2,
+                                     out_channels=num_hiddens,
+                                     kernel_size=4,
+                                     stride=2, padding=1)
+            self._conv_A = nn.Conv1d(in_channels=num_hiddens,
+                                     out_channels=num_hiddens,
+                                     kernel_size=4,
+                                     stride=2, padding=1)
+            self._conv_B = nn.Conv1d(in_channels=num_hiddens,
+                                     out_channels=num_hiddens,
+                                     kernel_size=4,
+                                     stride=2, padding=1)
+            self._conv_3 = nn.Conv1d(in_channels=num_hiddens,
+                                     out_channels=num_hiddens,
+                                     kernel_size=3,
+                                     stride=1, padding=1)
+            self._residual_stack = ResidualStack(in_channels=num_hiddens,
+                                                 num_hiddens=num_hiddens,
+                                                 num_residual_layers=num_residual_layers,
+                                                 num_residual_hiddens=num_residual_hiddens)
+
+            self._pre_vq_conv = nn.Conv1d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)
+
+    def forward(self, inputs, compression_factor):
+        if compression_factor == 4:
+            # TODO original
+            x = inputs.view([inputs.shape[0], 1, inputs.shape[-1]])
+
+            x = self._conv_1(x)
+            x = F.relu(x)
+
+            x = self._conv_2(x)
+            x = F.relu(x)
+
+            x = self._conv_3(x)
+            x = self._residual_stack(x)
+            x = self._pre_vq_conv(x)
+            return x
+
+        elif compression_factor == 8:
+            x = inputs.view([inputs.shape[0], 1, inputs.shape[-1]])
+
+            x = self._conv_1(x)
+            x = F.relu(x)
+
+            x = self._conv_2(x)
+            x = F.relu(x)
+
+            x = self._conv_A(x)
+            x = F.relu(x)
+
+            x = self._conv_3(x)
+            x = self._residual_stack(x)
+            x = self._pre_vq_conv(x)
+            return x
+
+        elif compression_factor == 12:
+            x = inputs.view([inputs.shape[0], 1, inputs.shape[-1]])
+
+            x = self._conv_1(x)
+            x = F.relu(x)
+
+            x = self._conv_2(x)
+            x = F.relu(x)
+
+            x = self._conv_3(x)
+            x = F.relu(x)
+
+            x = self._conv_4(x)
+            x = self._residual_stack(x)
+            x = self._pre_vq_conv(x)
+            return x
+
+        elif compression_factor == 16:
+            x = inputs.view([inputs.shape[0], 1, inputs.shape[-1]])
+
+            x = self._conv_1(x)
+            x = F.relu(x)
+
+            x = self._conv_2(x)
+            x = F.relu(x)
+
+            x = self._conv_A(x)
+            x = F.relu(x)
+
+            x = self._conv_B(x)
+            x = F.relu(x)
+
+            x = self._conv_3(x)
+            x = self._residual_stack(x)
+            x = self._pre_vq_conv(x)
+            return x
+
+
+class Decoder(nn.Module):
+    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, compression_factor):
+        super(Decoder, self).__init__()
+        if compression_factor == 4:
+            # TODO original don't change
+            self._conv_1 = nn.Conv1d(in_channels=in_channels,
+                                     out_channels=num_hiddens,
+                                     kernel_size=3,
+                                     stride=1, padding=1)
+
+            self._residual_stack = ResidualStack(in_channels=num_hiddens,
+                                                 num_hiddens=num_hiddens,
+                                                 num_residual_layers=num_residual_layers,
+                                                 num_residual_hiddens=num_residual_hiddens)
+
+            self._conv_trans_1 = nn.ConvTranspose1d(in_channels=num_hiddens,
+                                                    out_channels=num_hiddens // 2,
+                                                    kernel_size=4,
+                                                    stride=2, padding=1)
+
+            self._conv_trans_2 = nn.ConvTranspose1d(in_channels=num_hiddens // 2,
+                                                    out_channels=1,
+                                                    kernel_size=4,
+                                                    stride=2, padding=1)
+
+        elif compression_factor == 8:
+            self._conv_1 = nn.Conv1d(in_channels=in_channels,
+                                     out_channels=num_hiddens,
+                                     kernel_size=3,
+                                     stride=1, padding=1)
+
+            self._residual_stack = ResidualStack(in_channels=num_hiddens,
+                                                 num_hiddens=num_hiddens,
+                                                 num_residual_layers=num_residual_layers,
+                                                 num_residual_hiddens=num_residual_hiddens)
+
+            self._conv_trans_A = nn.ConvTranspose1d(in_channels=num_hiddens,
+                                                    out_channels=num_hiddens,
+                                                    kernel_size=4,
+                                                    stride=2, padding=1)
+
+            self._conv_trans_1 = nn.ConvTranspose1d(in_channels=num_hiddens,
+                                                    out_channels=num_hiddens // 2,
+                                                    kernel_size=4,
+                                                    stride=2, padding=1)
+
+            self._conv_trans_2 = nn.ConvTranspose1d(in_channels=num_hiddens // 2,
+                                                    out_channels=1,
+                                                    kernel_size=4,
+                                                    stride=2, padding=1)
+
+        elif compression_factor == 12:
+            self._conv_1 = nn.Conv1d(in_channels=in_channels,
+                                     out_channels=num_hiddens,
+                                     kernel_size=3,
+                                     stride=1, padding=1)
+
+            self._residual_stack = ResidualStack(in_channels=num_hiddens,
+                                                 num_hiddens=num_hiddens,
+                                                 num_residual_layers=num_residual_layers,
+                                                 num_residual_hiddens=num_residual_hiddens)
+
+            # TODO to get the correct shape back the kernel size has to be 5 not 4
+            self._conv_trans_2 = nn.ConvTranspose1d(in_channels=num_hiddens,
+                                                    out_channels=num_hiddens,
+                                                    kernel_size=5,
+                                                    stride=3, padding=1)
+
+            self._conv_trans_3 = nn.ConvTranspose1d(in_channels=num_hiddens,
+                                                    out_channels=num_hiddens // 2,
+                                                    kernel_size=4,
+                                                    stride=2, padding=1)
+
+            self._conv_trans_4 = nn.ConvTranspose1d(in_channels=num_hiddens // 2,
+                                                    out_channels=1,
+                                                    kernel_size=4,
+                                                    stride=2, padding=1)
+
+        elif compression_factor == 16:
+            self._conv_1 = nn.Conv1d(in_channels=in_channels,
+                                     out_channels=num_hiddens,
+                                     kernel_size=3,
+                                     stride=1, padding=1)
+
+            self._residual_stack = ResidualStack(in_channels=num_hiddens,
+                                                 num_hiddens=num_hiddens,
+                                                 num_residual_layers=num_residual_layers,
+                                                 num_residual_hiddens=num_residual_hiddens)
+
+            self._conv_trans_A = nn.ConvTranspose1d(in_channels=num_hiddens,
+                                                    out_channels=num_hiddens,
+                                                    kernel_size=4,
+                                                    stride=2, padding=1)
+
+            self._conv_trans_B = nn.ConvTranspose1d(in_channels=num_hiddens,
+                                                    out_channels=num_hiddens,
+                                                    kernel_size=4,
+                                                    stride=2, padding=1)
+
+            self._conv_trans_1 = nn.ConvTranspose1d(in_channels=num_hiddens,
+                                                    out_channels=num_hiddens // 2,
+                                                    kernel_size=4,
+                                                    stride=2, padding=1)
+
+            self._conv_trans_2 = nn.ConvTranspose1d(in_channels=num_hiddens // 2,
+                                                    out_channels=1,
+                                                    kernel_size=4,
+                                                    stride=2, padding=1)
+
+    def forward(self, inputs, compression_factor):
+        if compression_factor == 4:
+            # TODO original
+            x = self._conv_1(inputs)
+
+            x = self._residual_stack(x)
+
+            x = self._conv_trans_1(x)
+            x = F.relu(x)
+
+            x = self._conv_trans_2(x)
+
+            return torch.squeeze(x)
+
+        elif compression_factor == 8:
+            x = self._conv_1(inputs)
+
+            x = self._residual_stack(x)
+
+            x = self._conv_trans_A(x)
+            x = F.relu(x)
+
+            x = self._conv_trans_1(x)
+            x = F.relu(x)
+
+            x = self._conv_trans_2(x)
+
+            return torch.squeeze(x)
+
+        elif compression_factor == 12:
+            x = self._conv_1(inputs)
+            x = self._residual_stack(x)
+
+            x = self._conv_trans_2(x)
+            x = F.relu(x)
+
+            x = self._conv_trans_3(x)
+            x = F.relu(x)
+
+            x = self._conv_trans_4(x)
+
+            return torch.squeeze(x)
+
+        elif compression_factor == 16:
+            x = self._conv_1(inputs)
+
+            x = self._residual_stack(x)
+
+            x = self._conv_trans_A(x)
+            x = F.relu(x)
+
+            x = self._conv_trans_B(x)
+            x = F.relu(x)
+
+            x = self._conv_trans_1(x)
+            x = F.relu(x)
+
+            x = self._conv_trans_2(x)
+
+            return torch.squeeze(x)
+
+
+class VectorQuantizer(nn.Module):
+    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
+        super(VectorQuantizer, self).__init__()
+
+        self._embedding_dim = embedding_dim
+        self._num_embeddings = num_embeddings
+
+        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
+        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
+        self._commitment_cost = commitment_cost
+
+    def forward(self, inputs):
+        # convert inputs from BCHW -> BHWC
+        # import pdb; pdb.set_trace()
+        inputs = inputs.permute(0, 2, 1).contiguous()
+        input_shape = inputs.shape
+
+        # Flatten input
+        flat_input = inputs.view(-1, self._embedding_dim)
+
+        # Calculate distances
+        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True) + torch.sum(self._embedding.weight ** 2, dim=1) - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
+
+        # Encoding
+        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
+        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
+        encodings.scatter_(1, encoding_indices, 1)
+
+        # Quantize and unflatten
+        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
+
+        # Loss
+        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
+        q_latent_loss = F.mse_loss(quantized, inputs.detach())
+        loss = q_latent_loss + self._commitment_cost * e_latent_loss
+
+        quantized = inputs + (quantized - inputs).detach()
+
+        avg_probs = torch.mean(encodings, dim=0)
+        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
+        return loss, quantized.permute(0, 2, 1).contiguous(), perplexity, self._embedding.weight, encoding_indices, encodings
+
+
+class vqvae(BaseModel):
+    def __init__(self, vqvae_config):
+        super().__init__()
+        num_hiddens = vqvae_config['block_hidden_size']
+        num_residual_layers = vqvae_config['num_residual_layers']
+        num_residual_hiddens = vqvae_config['res_hidden_size']
+        embedding_dim = vqvae_config['embedding_dim']
+        num_embeddings = vqvae_config['num_embeddings']
+        commitment_cost = vqvae_config['commitment_cost']
+        self.compression_factor = vqvae_config['compression_factor']
+
+        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
+        self.encoder = Encoder(1, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim, self.compression_factor)
+        self.decoder = Decoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens, self.compression_factor)
+
+    def shared_eval(self, batch, optimizer, mode, comet_logger=None):
+        # TODO note this function is overwritten by the quantized output module
+        if mode == 'train':
+            optimizer.zero_grad()
+            z = self.encoder(batch, self.compression_factor)
+            vq_loss, quantized, perplexity, embedding_weight, encoding_indices, encodings = self.vq(z)
+            data_recon = self.decoder(quantized, self.compression_factor)
+            recon_error = F.mse_loss(data_recon, batch)
+            loss = recon_error + vq_loss
+            loss.backward()
+            optimizer.step()
+
+        if mode == 'val' or mode == 'test':
+            with torch.no_grad():
+                z = self.encoder(batch, self.compression_factor)
+                vq_loss, quantized, perplexity, embedding_weight, encoding_indices, encodings = self.vq(z)
+                data_recon = self.decoder(quantized, self.compression_factor)
+                recon_error = F.mse_loss(data_recon, batch)
+                loss = recon_error + vq_loss
+
+        comet_logger.log_metric(f'{mode}_vqvae_loss_each_batch', loss.item())
+        comet_logger.log_metric(f'{mode}_vqvae_vq_loss_each_batch', vq_loss.item())
+        comet_logger.log_metric(f'{mode}_vqvae_recon_loss_each_batch', recon_error.item())
+        comet_logger.log_metric(f'{mode}_vqvae_perplexity_each_batch', perplexity.item())
+
+        return loss, vq_loss, recon_error, data_recon, perplexity, embedding_weight, encoding_indices, encodings
+
+
```



- **Added** `lib/utils/checkpoint.py`

  - **Diff:**

  ```diff

+import torch
+import numpy as np
+import os
+
+
+class EarlyStopping:
+    def __init__(self, patience=7, path="/data/georgia", delta=0):
+        self.patience = patience
+        self.counter = 0
+        self.best_mae = None
+        self.best_mse = None
+        self.early_stop = False
+        self.delta = delta
+        self.path = path
+
+    def __call__(self, mse, mae, models):
+        if self.best_mae is None:  # first time
+            self.best_mae = mae
+            self.best_mse = mse
+            self.save_checkpoint(models)
+        elif (mae > self.best_mae + self.delta) and (mse > self.best_mse + self.delta):
+            # both metrics got worse
+            self.counter += 1
+            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
+            if self.counter >= self.patience:
+                self.early_stop = True
+        else:
+            self.best_mae = mae
+            self.best_mse = mse
+            self.save_checkpoint(models)
+            self.counter = 0
+
+    def save_checkpoint(self, models):
+        if self.path is not None:
+            print("Saving model...")
+            for name, model in models.items():
+                torch.save(
+                    model.state_dict(),
+                    os.path.join(self.path, "%d_checkpoint.pth" % (name)),
+                )
```



- **Added** `lib/utils/env.py`

  - **Affected functions:**

    - `seed_all_rng`

  - **Diff:**

  ```diff

+import numpy as np
+import torch
+import random
+import os
+from datetime import datetime
+
+
+def seed_all_rng(seed=None):
+    """
+    Set the random seed for the RNG in torch, numpy and python.
+
+    Args:
+        seed (int): if None, will use a strong random seed.
+
+    Use:
+        seed_all_rng(None if seed < 0 else seed)
+    """
+    if seed is None:
+        seed = (
+            os.getpid()
+            + int(datetime.now().strftime("%S%f"))
+            + int.from_bytes(os.urandom(2), "big")
+        )
+        print("Using a generated random seed {}".format(seed))
+    np.random.seed(seed)
+    torch.manual_seed(seed)
+    random.seed(seed)
+    os.environ["PYTHONHASHSEED"] = str(seed)
```



- **Added** `lib/utils/timefeatures.py`

  - **Affected functions:**

    - `time_features`

    - `time_features_from_frequency_str`

  - **Diff:**

  ```diff

+from typing import List
+
+import numpy as np
+import pandas as pd
+from pandas.tseries import offsets
+from pandas.tseries.frequencies import to_offset
+
+
+class TimeFeature:
+    def __init__(self):
+        pass
+
+    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
+        pass
+
+    def __repr__(self):
+        return self.__class__.__name__ + "()"
+
+
+class SecondOfMinute(TimeFeature):
+    """Minute of hour encoded as value between [-0.5, 0.5]"""
+
+    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
+        return index.second / 59.0 - 0.5
+
+
+class MinuteOfHour(TimeFeature):
+    """Minute of hour encoded as value between [-0.5, 0.5]"""
+
+    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
+        return index.minute / 59.0 - 0.5
+
+
+class HourOfDay(TimeFeature):
+    """Hour of day encoded as value between [-0.5, 0.5]"""
+
+    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
+        return index.hour / 23.0 - 0.5
+
+
+class DayOfWeek(TimeFeature):
+    """Hour of day encoded as value between [-0.5, 0.5]"""
+
+    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
+        return index.dayofweek / 6.0 - 0.5
+
+
+class DayOfMonth(TimeFeature):
+    """Day of month encoded as value between [-0.5, 0.5]"""
+
+    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
+        return (index.day - 1) / 30.0 - 0.5
+
+
+class DayOfYear(TimeFeature):
+    """Day of year encoded as value between [-0.5, 0.5]"""
+
+    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
+        return (index.dayofyear - 1) / 365.0 - 0.5
+
+
+class MonthOfYear(TimeFeature):
+    """Month of year encoded as value between [-0.5, 0.5]"""
+
+    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
+        return (index.month - 1) / 11.0 - 0.5
+
+
+class WeekOfYear(TimeFeature):
+    """Week of year encoded as value between [-0.5, 0.5]"""
+
+    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
+        return (index.isocalendar().week - 1) / 52.0 - 0.5
+
+
+def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
+    """
+    Returns a list of time features that will be appropriate for the given frequency string.
+    Parameters
+    ----------
+    freq_str
+        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
+    """
+
+    features_by_offsets = {
+        offsets.YearEnd: [],
+        offsets.QuarterEnd: [MonthOfYear],
+        offsets.MonthEnd: [MonthOfYear],
+        offsets.Week: [DayOfMonth, WeekOfYear],
+        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
+        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
+        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
+        offsets.Minute: [
+            MinuteOfHour,
+            HourOfDay,
+            DayOfWeek,
+            DayOfMonth,
+            DayOfYear,
+        ],
+        offsets.Second: [
+            SecondOfMinute,
+            MinuteOfHour,
+            HourOfDay,
+            DayOfWeek,
+            DayOfMonth,
+            DayOfYear,
+        ],
+    }
+
+    offset = to_offset(freq_str)
+
+    for offset_type, feature_classes in features_by_offsets.items():
+        if isinstance(offset, offset_type):
+            return [cls() for cls in feature_classes]
+
+    supported_freq_msg = f"""
+    Unsupported frequency {freq_str}
+    The following frequencies are supported:
+        Y   - yearly
+            alias: A
+        M   - monthly
+        W   - weekly
+        D   - daily
+        B   - business days
+        H   - hourly
+        T   - minutely
+            alias: min
+        S   - secondly
+    """
+    raise RuntimeError(supported_freq_msg)
+
+
+def time_features(dates, freq='h'):
+    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])
```



- **Added** `scripts/step1.sh`

  - **Diff:**

  ```diff

+repo_dir="/path/to/TOTEM_for_EEG_code"
+
+dataset_name="example"
+
+python -m steps.STEP1_save_revin_xdata_for_vqvae \
+    +preprocessing=step1_eeg \
+    "++preprocessing.root_paths=['${repo_dir}/data']" \
+    "++preprocessing.data_paths=['${dataset_name}.csv']" \
+    ++preprocessing.save_path="${repo_dir}/pipeline/step1_revin_x_data/${dataset_name}"
```



- **Added** `scripts/step2.sh`

  - **Diff:**

  ```diff

+repo_dir="/path/to/TOTEM_for_EEG_code"
+
+dataset_name="example"
+python -m steps.STEP2_train_vqvae \
+    +exp=train_vqvae \
+    ++exp.save_dir="${repo_dir}/pipeline/step2_train_vqvae/${dataset_name}" \
+    ++exp.vqvae_config.dataset=${dataset_name} \
+    ++exp.vqvae_config.dataset_base_path="${repo_dir}/pipeline/step1_revin_x_data" \
+    +logging=comet
```



- **Added** `scripts/step3.sh`

  - **Diff:**

  ```diff

+repo_dir="/path/to/TOTEM_for_EEG_code"
+
+dataset_name="example"
+python -m pdb -m steps.STEP3_save_classification_data \
+    +preprocessing=step3_eeg \
+    "++preprocessing.train_root_paths=['${repo_dir}/data']" \
+    "++preprocessing.train_data_paths=['${dataset_name}.csv']" \
+    "++preprocessing.test_root_paths=['${repo_dir}/data']" \
+    "++preprocessing.test_data_paths=['${dataset_name}.csv']" \
+    ++preprocessing.save_path="${repo_dir}/pipeline/step3_classification_data/${dataset_name}" \
+    ++preprocessing.trained_vqvae_model_path="${repo_dir}/pipeline/step2_train_vqvae/${dataset_name}/checkpoints/final_model.pth" \
```



- **Added** `scripts/step4.sh`

  - **Diff:**

  ```diff

+repo_dir="/path/to/TOTEM_for_EEG_code"
+
+dataset_name="example"
+
+python -m steps.STEP4_train_xformer \
+    +exp=train_xformer \
+    ++exp.classifier_config.data_path="${repo_dir}/pipeline/step3_classification_data/${dataset_name}" \
+    ++exp.classifier_config.checkpoint_path="${repo_dir}/pipeline/step4_train_xformer" \
+    ++exp.classifier_config.exp_name=${dataset_name} \
+    +logging=comet
```



- **Added** `steps/STEP1_save_revin_xdata_for_vqvae.py`

  - **Affected functions:**

    - `main`

  - **Diff:**

  ```diff

+import argparse
+import numpy as np
+import os
+import pdb
+import random
+import torch
+
+from data_provider.data_factory_vqvae_no_shuffle import data_provider_flexPath
+from lib.models.revin import RevIN
+
+from omegaconf import DictConfig, OmegaConf
+import hydra
+
+import logging
+
+log = logging.getLogger(__name__)
+
+@hydra.main(config_path="../conf", version_base="1.1")
+def main(cfg: DictConfig) -> None:
+    log.info("STEP1: Save RevIN x data for training VQ-VAE")
+    log.info(OmegaConf.to_yaml(cfg, resolve=True))
+    log.info(f'Working directory {os.getcwd()}')
+
+    args = cfg.preprocessing
+
+    # random seed
+    fix_seed = args.random_seed
+    random.seed(fix_seed)
+    torch.manual_seed(fix_seed)
+    np.random.seed(fix_seed)
+
+    ## Only allow gpu usage if cuda is available
+    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
+
+    ## print 
+    print('Args in experiment:')
+    print(args)
+
+    Exp = ExtractData
+    exp = Exp(args)  # set experiments
+    exp.extract_data()
+
+
+
+class ExtractData:
+    def __init__(self, args):
+        self.args = args
+        self.device = 'cuda:' + str(self.args.gpu)
+        self.revin_layer_x = RevIN(num_features=self.args.enc_in, affine=False, subtract_last=False)
+        self.revin_layer_y = RevIN(num_features=self.args.enc_in, affine=False, subtract_last=False)
+
+    def _get_data(self, root_path, data_path, flag):
+        data_set, data_loader = data_provider_flexPath(args = self.args, root_path = root_path, data_path = data_path, flag=flag)
+        return data_set, data_loader
+
+    def one_loop(self, loader):
+        x_in_revin_space = []
+        y_in_revin_space = []
+
+        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
+            batch_x = batch_x.float().to(self.device)
+            batch_y = batch_y.float().to(self.device)
+
+            # data going into revin should have dim:[bs x seq_len x nvars]
+            x_in_revin_space.append(np.array(self.revin_layer_x(batch_x, "norm").detach().cpu()))
+            y_in_revin_space.append(np.array(self.revin_layer_y(batch_y, "norm").detach().cpu()))
+
+        x_in_revin_space_arr = np.concatenate(x_in_revin_space, axis=0)
+        y_in_revin_space_arr = np.concatenate(y_in_revin_space, axis=0)
+
+        print(x_in_revin_space_arr.shape, y_in_revin_space_arr.shape)
+        return x_in_revin_space_arr, y_in_revin_space_arr
+
+    def extract_data(self):
+
+        x_train_arr_list = []  # Initialize an empty list to hold x_train_arr from each iteration
+        x_val_arr_list = []
+        x_test_arr_list = []
+
+        for root_path_name, data_path_name in zip(self.args.root_paths, self.args.data_paths):          
+            _, train_loader = self._get_data(root_path_name, data_path_name, flag='train')
+            _, vali_loader = self._get_data(root_path_name, data_path_name, flag='val')
+            _, test_loader = self._get_data(root_path_name, data_path_name, flag='test')
+
+            print('got loaders starting revin')
+
+            # These have dimension [bs, ntime, nvars]
+            x_train_in_revin_space_arr, y_train_in_revin_space_arr = self.one_loop(train_loader)
+            print('starting val')
+            x_val_in_revin_space_arr, y_val_in_revin_space_arr = self.one_loop(vali_loader)
+            print('starting test')
+            x_test_in_revin_space_arr, y_test_in_revin_space_arr = self.one_loop(test_loader)
+
+            print('Flattening Sensors Out')
+            if self.args.seq_len != self.args.pred_len:
+                print('HoUstoN wE haVE A prOblEm')
+                pdb.set_trace()
+            else:
+                # These have dimension [bs x nvars, ntime]
+                x_train_arr = np.swapaxes(x_train_in_revin_space_arr, 1,2).reshape((-1, self.args.pred_len))
+                x_val_arr = np.swapaxes(x_val_in_revin_space_arr, 1, 2).reshape((-1, self.args.pred_len))
+                x_test_arr = np.swapaxes(x_test_in_revin_space_arr, 1, 2).reshape((-1, self.args.pred_len))
+                print("Final output")
+                print(x_train_arr.shape, x_val_arr.shape, x_test_arr.shape)
+
+                x_train_arr_list.append(x_train_arr)
+                x_val_arr_list.append(x_val_arr)
+                x_test_arr_list.append(x_test_arr)
+
+                print(len(x_train_arr_list), len(x_val_arr_list), len(x_test_arr_list))
+
+
+        # Concatenate all arrays into a single array
+        x_train_arr = np.concatenate(x_train_arr_list, axis=0)
+        x_val_arr = np.concatenate(x_val_arr_list, axis=0)
+        x_test_arr = np.concatenate(x_test_arr_list, axis=0)
+
+        # Now, x_train_arr contains the concatenated array from all iterations
+        print("Final x_train_arr shape:", x_train_arr.shape)
+        print("Final x_val_arr shape:", x_val_arr.shape)
+        print("Final x_test_arr shape:", x_test_arr.shape)
+
+        if not os.path.exists(self.args.save_path):
+            os.makedirs(self.args.save_path)
+        np.save(self.args.save_path + '/train_data_x.npy', x_train_arr)
+        np.save(self.args.save_path + '/val_data_x.npy', x_val_arr)
+        np.save(self.args.save_path + '/test_data_x.npy', x_test_arr)
+
+
+if __name__ == '__main__':
+    main()
```



- **Added** `steps/STEP2_train_vqvae.py`

  - **Affected functions:**

    - `create_datloaders`

    - `main`

    - `runner`

    - `start_training`

    - `train_model`

  - **Diff:**

  ```diff

+import argparse
+import copy
+
+import comet_ml
+import json
+import numpy as np
+import os
+import pdb
+import random
+import time
+import torch
+
+from lib.models import get_model_class
+from time import gmtime, strftime
+
+from omegaconf import DictConfig, OmegaConf
+import hydra
+
+import logging
+
+log = logging.getLogger(__name__)
+
+@hydra.main(config_path="../conf", version_base="1.1")
+def main(cfg: DictConfig) -> None:
+    log.info("STEP2: Train VQ-VAE")
+    log.info(OmegaConf.to_yaml(cfg, resolve=True))
+    log.info(f'Working directory {os.getcwd()}')
+
+    exp_cfg = cfg.exp
+    logging_cfg = cfg.logging
+    save_dir = exp_cfg.save_dir 
+
+    # (6) Setting up the comet logger
+    if logging_cfg.comet:
+        comet_config = logging_cfg.comet
+        # Create an experiment with your api key
+        comet_logger = comet_ml.Experiment(
+            api_key=comet_config['api_key'],
+            project_name=comet_config['project_name'],
+            workspace=comet_config['workspace'],
+        )
+        comet_logger.add_tag(comet_config.comet_tag)
+        comet_logger.set_name(comet_config.comet_name)
+    else:
+        print('PROBLEM: not saving to comet')
+        comet_logger = None
+        pdb.set_trace()
+
+    # (7) Set up GPU / CPU
+    if torch.cuda.is_available() and exp_cfg.gpu_id >= 0:
+        assert exp_cfg.gpu_id < torch.cuda.device_count()  # sanity check
+        device = 'cuda:{:d}'.format(exp_cfg.gpu_id)
+    else:
+        device = 'cpu'
+
+    # (8) Where to init data for training (cpu or gpu) -->  will be trained wherever args.model_init_num_gpus says
+    if exp_cfg.data_init_cpu_or_gpu == 'gpu':
+        data_init_loc = device  # we do this so that data_init_loc will have the correct cuda:X if gpu
+    else:
+        data_init_loc = 'cpu'
+
+    # (9) call runner
+    runner(device, exp_cfg, save_dir, comet_logger, data_init_loc)
+
+
+def runner(device, config, save_dir, logger, data_init_loc):
+    # (1) Create/overwrite checkpoints folder and results folder
+    # Create model checkpoints folder
+    if os.path.exists(os.path.join(save_dir, 'checkpoints')):
+        print("Checkpoint Directory:", os.path.join(save_dir, 'checkpoints'))
+        print('Checkpoint Directory Already Exists - if continue will overwrite files inside. Press c to continue.')
+        # pdb.set_trace()
+    else:
+        os.makedirs(os.path.join(save_dir, 'checkpoints'))
+
+
+    # (3) log the config parameters to comet_ml
+    logger.log_parameters(config)
+
+    # (4) Run start training
+    vqvae_config, summary = start_training(device=device, vqvae_config=config['vqvae_config'], save_dir=save_dir,
+                                           logger=logger, data_init_loc=data_init_loc)
+
+
+def start_training(device, vqvae_config, save_dir, logger, data_init_loc):
+    # (1) Create summary dictionary
+    summary = {}
+
+    # (2) Sample and fix a random seed if not set
+    if 'general_seed' not in vqvae_config:
+        vqvae_config['seed'] = random.randint(0, 9999)
+
+    general_seed = vqvae_config['general_seed']
+    summary['general_seed'] = general_seed
+    torch.manual_seed(general_seed)
+    random.seed(general_seed)
+    np.random.seed(general_seed)
+    # if use another random library need to set that seed here too
+
+    torch.backends.cudnn.deterministic = True  # makes cuDNN to only have to only use determinisitic convolution algs.
+
+    # summary['dataset'] = datamodule.summary  # add dataset name to the summary
+    summary['data initialization location'] = data_init_loc
+    summary['device'] = device  # add the cpu/gpu to the summary
+
+    # (4) Setup model
+    model_class = get_model_class(vqvae_config['model_name'].lower())
+    model = model_class(vqvae_config)  # Initialize model
+
+    # Uncomment if want to know the total number of trainable parameters
+    print('Total # trainable parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
+
+    if vqvae_config['pretrained']:
+        # pretrained needs to be the path to the trained model if you want it to load
+        model = torch.load(vqvae_config['pretrained'])  # Get saved pytorch model.
+    summary['vqvae_config'] = vqvae_config  # add the model information to the summary
+
+    # (5) Start training the model
+    start_time = time.time()
+    model = train_model(model, device, vqvae_config, save_dir, logger)
+
+    # (6) Once the model has trained - Save full pytorch model
+    torch.save(model, os.path.join(save_dir, 'checkpoints/final_model.pth'))
+    # logger.log_model("Prog_Model", os.path.join(save_dir, 'checkpoints'))  # save model information to comet
+
+    # (7) Save and return
+    summary['total_time'] = round(time.time() - start_time, 3)
+    return vqvae_config, summary
+
+
+def train_model(model, device, vqvae_config, save_dir, logger):
+    # Set the optimizer
+    optimizer = model.configure_optimizers(lr=vqvae_config['learning_rate'])
+    # If want a learning rate scheduler instead uncomment this
+    # lr_lambda = lambda epoch: max(1e-6, 0.9**(int(epoch/250))*train_config['learning_rate'])
+    # lr_scheduler = LambdaLR(optimizer, lr_lambda)
+
+    # Setup model (send to device, set to train)
+    model.to(device)
+    start_time = time.time()
+
+    print('BATCHSIZE:', vqvae_config["batch_size"])
+    train_loader, vali_loader, test_loader = create_datloaders(batchsize=vqvae_config["batch_size"], dataset=vqvae_config["dataset"], base_path=vqvae_config.get("dataset_base_path", "/data/sabera/mts_v2_datasets/pipeline/revin_data_to_train_vqvae"))
+
+    # do + 0.5 to ciel it
+    for epoch in range(int(vqvae_config['num_epochs'])):
+        model.train()
+        for i, (batch_x) in enumerate(train_loader):
+            tensor_all_data_in_batch = torch.tensor(batch_x, dtype=torch.float, device=device)
+
+            loss, vq_loss, recon_error, x_recon, perplexity, embedding_weight, encoding_indices, encodings = \
+                model.shared_eval(tensor_all_data_in_batch, optimizer, 'train', comet_logger=logger)
+
+        if epoch % 10 == 0:
+            with (torch.no_grad()):
+                model.eval()
+                for i, (batch_x) in enumerate(vali_loader):
+                    tensor_all_data_in_batch = torch.tensor(batch_x, dtype=torch.float, device=device)
+
+                    val_loss, val_vq_loss, val_recon_error, val_x_recon, val_perplexity, val_embedding_weight, \
+                        val_encoding_indices, val_encodings = \
+                        model.shared_eval(tensor_all_data_in_batch, optimizer, 'val', comet_logger=logger)
+
+        if epoch % 10 == 0:
+            # save the model checkpoints locally and to comet
+            torch.save(model, os.path.join(save_dir, f'checkpoints/model_epoch_{epoch}.pth'))
+            print('Saved model from epoch ', epoch)
+
+    print('total time: ', round(time.time() - start_time, 3))
+    return model
+
+
+def create_datloaders(batchsize=100, dataset="dummy", base_path="/data/sabera/mts_v2_datasets/pipeline/revin_data_to_train_vqvae"):
+
+    if dataset == 'weather':
+        print('weather')
+        full_path = base_path + '/weather'
+        
+    elif dataset == 'electricity':
+        print('electricity')
+        full_path = base_path + '/electricity'
+
+    elif dataset == 'ETTh1':
+        print('ETTh1')
+        full_path = base_path + '/ETTh1'
+
+    elif dataset == 'ETTm1':
+        print('ETTm1')
+        full_path = base_path + '/ETTm1'
+
+    elif dataset == 'ETTh2':
+        print('ETTh2')
+        full_path = base_path + '/ETTh2'
+
+    elif dataset == 'ETTm2':
+        print('ETTm2')
+        full_path = base_path + '/ETTm2'
+
+    elif dataset == 'pt12':
+        print('PT 12')
+        full_path = base_path + '/pt12'
+
+    elif dataset == 'pt2':
+        print('PT 2')
+        full_path = base_path + '/pt2'
+
+    elif dataset == 'pt5':
+        print('PT 5')
+        full_path = base_path + '/pt5'
+
+    elif dataset == 'earthquake_0split':
+        print('earthquake_0shotsplot')
+        # it says compression4 but that is just where the original data is saved --> FIX ME LATER
+        full_path = '/data/sabera/mts_v2_datasets/earthquake_clean_0shotsplit/vqvae_results/compression4'
+
+    elif dataset == 'earthquake_randomsplit':
+        print('earthquake random split')
+        # it says compression4 but that is just where the original data is saved --> FIX ME LATER
+        full_path = '/data/sabera/mts_v2_datasets/earthquake_clean_randomsplit/vqvae_results/compression4'
+
+    elif dataset == 'neuro_train2and5_test12':
+        print('neuro_train2and5_test12')
+        full_path = base_path + '/trainpt2and5_testpt12'
+    else:
+        full_path = base_path + f'/{dataset}'
+        print(f'using {full_path} for dataset')
+
+    train_data = np.load(os.path.join(full_path, "train_data_x.npy"), allow_pickle=True)
+    val_data = np.load(os.path.join(full_path, "val_data_x.npy"), allow_pickle=True)
+    test_data = np.load(os.path.join(full_path, "test_data_x.npy"), allow_pickle=True)
+
+    train_dataloader = torch.utils.data.DataLoader(train_data,
+                                                   batch_size=batchsize,
+                                                   shuffle=True,
+                                                   num_workers=1,
+                                                   drop_last=True)
+
+    val_dataloader = torch.utils.data.DataLoader(val_data,
+                                                batch_size=batchsize,
+                                                shuffle=False,
+                                                num_workers=1,
+                                                drop_last=False)
+
+    test_dataloader = torch.utils.data.DataLoader(test_data,
+                                                batch_size=batchsize,
+                                                shuffle=False,
+                                                num_workers=1,
+                                                drop_last=False)
+
+    return train_dataloader, val_dataloader, test_dataloader
+
+
+if __name__ == '__main__':
+    main()
```



- **Added** `steps/STEP3_save_classification_data.py`

  - **Affected functions:**

    - `codes2time`

    - `main`

    - `save_files_classification`

    - `time2codes`

  - **Diff:**

  ```diff

+import argparse
+import numpy as np
+import os
+import pdb
+import pickle
+import torch
+import torch.nn as nn
+
+from data_provider.data_factory_vqvae_no_shuffle import data_provider_flexPath
+from lib.models.revin import RevIN
+
+from omegaconf import DictConfig, OmegaConf
+import hydra
+
+import logging
+
+log = logging.getLogger(__name__)
+
+@hydra.main(config_path="../conf", version_base="1.1")
+def main(cfg: DictConfig) -> None:
+    log.info("STEP3: Save classification data")
+    log.info(OmegaConf.to_yaml(cfg, resolve=True))
+    log.info(f'Working directory {os.getcwd()}')
+
+    args = cfg.preprocessing
+
+    # random seed
+    fix_seed = args.random_seed
+    # random.seed(fix_seed)  # This isn't used
+    torch.manual_seed(fix_seed)
+    np.random.seed(fix_seed)
+
+    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
+
+    print('Args in experiment:')
+    print(args)
+
+    if len(args.train_root_paths) == 0 and len(args.train_data_paths) == 0 and len(args.test_root_paths) == 0 and len(args.test_data_paths) == 0:
+        ## Single session data extraction (train, val, test all from one file)
+        print("Saving single file")
+        Exp = ExtractData
+        exp = Exp(args)  # set experiments
+        exp.extract_data(save_data=True)
+    else:
+        ## Multi-session data extraction (train, val from train_data_paths) (test from test_data_paths)
+        assert len(args.train_root_paths) == len(args.train_data_paths), "Train Root paths and data_paths lengths don't match! Cannot get multiple datas"
+        assert len(args.test_root_paths) == len(args.test_data_paths), "Test Root paths and data_paths lengths don't match! Cannot get multiple datas"
+
+        train_val_exps = []
+        for i in range(len(args.train_root_paths)): 
+            root_path_name = args.train_root_paths[i]
+            data_path_name = args.train_data_paths[i]
+            
+            Exp = ExtractData
+            exp = Exp(args)
+            exp.extract_data(root_path_name=root_path_name, data_path_name=data_path_name, save_data=False)
+            train_val_exps.append(exp)
+        
+        test_exps = []
+        for i in range(len(args.test_root_paths)): 
+            root_path_name = args.test_root_paths[i]
+            data_path_name = args.test_data_paths[i]
+            
+            Exp = ExtractData
+            exp = Exp(args)
+            exp.extract_data(root_path_name=root_path_name, data_path_name=data_path_name, save_data=False)
+            test_exps.append(exp)
+
+        ## Create train val dictionaries concatenated across files 
+        train_dict = {}
+        val_dict = {} 
+        for train_val_exp in train_val_exps: 
+            ## Train data from train_val data paths will be used in the train
+            for k, v in train_val_exp.train_data_dict.items(): 
+                if k not in train_dict: 
+                    train_dict[k] = v
+                elif k != "codebook": # Only concatenate if not codebook
+                    train_dict[k] = np.concatenate([train_dict[k], v], axis=0)
+            ## Val data from train_val data paths will be used in the val
+            for k, v in train_val_exp.val_data_dict.items(): 
+                if k not in val_dict: 
+                    val_dict[k] = v
+                elif k != "codebook": # Only concatenate if not codebook
+                    val_dict[k] = np.concatenate([val_dict[k], v], axis=0)
+            
+        ## Create test dictionaries concatenated across files 
+        test_dict = {} 
+        for test_exp in test_exps: 
+            for k, v in test_exp.test_data_dict.items():
+                if k not in test_dict: 
+                    test_dict[k] = v
+                elif k != "codebook": # Only concatenate if not codebook
+                    test_dict[k] = np.concatenate([test_dict[k], v], axis=0)
+
+        ## Save files 
+        if len(train_val_exps) > 0: 
+            save_files_classification(args.save_path, train_dict, mode="train", save_codebook=True)
+            save_files_classification(args.save_path, val_dict, mode="val", save_codebook=False)
+        if len(test_exps) > 0:
+            save_files_classification(args.save_path, test_dict, mode="test", save_codebook=False)
+
+class ExtractData:
+    def __init__(self, args):
+        self.args = args
+        self.device = 'cuda:' + str(self.args.gpu)
+        self.revin_layer_x = RevIN(num_features=self.args.enc_in, affine=False, subtract_last=False)
+        self.revin_layer_y = RevIN(num_features=self.args.enc_in, affine=False, subtract_last=False)
+
+    # def _get_data(self, flag):
+    #     data_set, data_loader = data_provider(self.args, flag)
+    #     return data_set, data_loader
+    
+    def _get_data(self, root_path, data_path, flag):
+        data_set, data_loader = data_provider_flexPath(args = self.args, root_path = root_path, data_path = data_path, flag=flag)
+        return data_set, data_loader
+
+    def one_loop_classification(self, loader, vqvae_model):
+        x_original_all = []
+        x_code_ids_all = []
+        x_reverted_all = []
+        x_labels_all = []
+        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
+            batch_x = batch_x[:, 1:, :] # making time go from 1001 to 1000 by dropping the first time step
+
+            x_original_all.append(batch_x)
+            batch_x = batch_x.float().to(self.device)
+
+            # data going into revin should have dim:[bs x time x nvars]
+            x_in_revin_space = self.revin_layer_x(batch_x, "norm")
+
+            # expects time to be dim [bs x nvars x time]
+            x_codes, x_code_ids, codebook = time2codes(x_in_revin_space.permute(0, 2, 1), self.args.compression_factor, vqvae_model.encoder, vqvae_model.vq)
+
+            x_code_ids_all.append(np.array(x_code_ids.detach().cpu()))
+
+            # expects code to be dim [bs x nvars x compressed_time]
+            x_predictions_revin_space, x_predictions_original_space = codes2time(x_code_ids, codebook, self.args.compression_factor, vqvae_model.decoder, self.revin_layer_x)
+
+            x_reverted_all.append(np.array(x_predictions_original_space.detach().cpu()))
+
+            x_labels_all.append(batch_x_mark)
+
+
+        x_original_arr = np.concatenate(x_original_all, axis=0)
+        x_code_ids_all_arr = np.concatenate(x_code_ids_all, axis=0)
+        x_reverted_all_arr = np.concatenate(x_reverted_all, axis=0)
+        x_labels_all_arr = np.concatenate(x_labels_all, axis=0)
+
+        data_dict = {}
+        data_dict['x_original_arr'] = x_original_arr
+        data_dict['x_code_ids_all_arr'] = np.swapaxes(x_code_ids_all_arr, 1, 2) # order will be [bs x compressed_time x sensors)
+        data_dict['x_reverted_all_arr'] = x_reverted_all_arr
+        data_dict['x_labels_all_arr'] = x_labels_all_arr
+        data_dict['codebook'] = np.array(codebook.detach().cpu())
+
+        print("x_original_arr shape", data_dict['x_original_arr'].shape)
+        print("x_code_ids_all_arr shape", data_dict['x_code_ids_all_arr'].shape)
+        print("x_reverted_all_arr shape", data_dict['x_reverted_all_arr'].shape)
+        print("x_labels_all_arr shape", data_dict['x_labels_all_arr'].shape)
+        print("codebook shape", data_dict['codebook'].shape)
+
+        return data_dict
+
+    def extract_data(self, root_path_name, data_path_name, save_data=True):
+        vqvae_model = torch.load(self.args.trained_vqvae_model_path)
+        vqvae_model.to(self.device)
+        vqvae_model.eval()
+
+        _, train_loader = self._get_data(root_path_name, data_path_name, flag='train')
+        _, vali_loader = self._get_data(root_path_name, data_path_name, flag='val')
+        _, test_loader = self._get_data(root_path_name, data_path_name, flag='test')
+
+
+        print('CLASSIFYING')
+        # These have dimension [bs, ntime, nvars]
+        print('-------------TRAIN-------------')
+        self.train_data_dict = self.one_loop_classification(train_loader, vqvae_model)
+        if save_data: 
+            save_files_classification(self.args.save_path, self.train_data_dict, 'train', save_codebook=True)
+
+        print('-------------VAL-------------')
+        self.val_data_dict = self.one_loop_classification(vali_loader, vqvae_model)
+        if save_data: 
+            save_files_classification(self.args.save_path, self.val_data_dict, 'val', save_codebook=False)
+
+        print('-------------Test-------------')
+        self.test_data_dict = self.one_loop_classification(test_loader, vqvae_model)
+        if save_data: 
+            save_files_classification(self.args.save_path, self.test_data_dict, 'test', save_codebook=False)
+
+
+def save_files_classification(path, data_dict, mode, save_codebook, description=''):
+    if not os.path.exists(path):
+        os.makedirs(path)
+    np.save(os.path.join(path, mode + f'{"_"+description if description != "" else ""}' + '_x_original.npy'), data_dict['x_original_arr'])
+    np.save(os.path.join(path, mode + f'{"_"+description if description != "" else ""}' +'_x_codes.npy'), data_dict['x_code_ids_all_arr'])
+    np.save(os.path.join(path, mode + f'{"_"+description if description != "" else ""}' + '_x_reverted.npy'), data_dict['x_reverted_all_arr'])
+    np.save(os.path.join(path, mode + f'{"_"+description if description != "" else ""}' + '_x_labels.npy'), data_dict['x_labels_all_arr'])
+
+    if save_codebook:
+        np.save(os.path.join(path, description + 'codebook.npy'), data_dict['codebook'])
+
+
+
+
+def time2codes(revin_data, compression_factor, vqvae_encoder, vqvae_quantizer):
+    '''
+    Args:
+        revin_data: [bs x nvars x pred_len or seq_len]
+        compression_factor: int
+        vqvae_model: trained vqvae model
+        use_grad: bool, if True use gradient, if False don't use gradients
+
+    Returns:
+        codes: [bs, nvars, code_dim, compressed_time]
+        code_ids: [bs, nvars, compressed_time]
+        embedding_weight: [num_code_words, code_dim]
+
+    Helpful VQVAE Comments:
+        # Into the vqvae encoder: batch.shape: [bs x seq_len] i.e. torch.Size([256, 12])
+        # into the quantizer: z.shape: [bs x code_dim x (seq_len/compresion_factor)] i.e. torch.Size([256, 64, 3])
+        # into the vqvae decoder: quantized.shape: [bs x code_dim x (seq_len/compresion_factor)] i.e. torch.Size([256, 64, 3])
+        # out of the vqvae decoder: data_recon.shape: [bs x seq_len] i.e. torch.Size([256, 12])
+    '''
+
+    bs = revin_data.shape[0]
+    nvar = revin_data.shape[1]
+    T = revin_data.shape[2]  # this can be either the prediction length or the sequence length
+    compressed_time = int(T / compression_factor)  # this can be the compressed time of either the prediction length or the sequence length
+
+    with torch.no_grad():
+        flat_revin = revin_data.reshape(-1, T)  # flat_y: [bs * nvars, T]
+        latent = vqvae_encoder(flat_revin.to(torch.float), compression_factor)  # latent_y: [bs * nvars, code_dim, compressed_time]
+        vq_loss, quantized, perplexity, embedding_weight, encoding_indices, encodings = vqvae_quantizer(latent)  # quantized: [bs * nvars, code_dim, compressed_time]
+        code_dim = quantized.shape[-2]
+        codes = quantized.reshape(bs, nvar, code_dim,
+                                  compressed_time)  # codes: [bs, nvars, code_dim, compressed_time]
+        code_ids = encoding_indices.view(bs, nvar, compressed_time)  # code_ids: [bs, nvars, compressed_time]
+
+    return codes, code_ids, embedding_weight
+
+
+def codes2time(code_ids, codebook, compression_factor, vqvae_decoder, revin_layer):
+    '''
+    Args:
+        code_ids: [bs x nvars x compressed_pred_len]
+        codebook: [num_code_words, code_dim]
+        compression_factor: int
+        vqvae_model: trained vqvae model
+        use_grad: bool, if True use gradient, if False don't use gradients
+        x_or_y: if 'x' use revin_denorm_x if 'y' use revin_denorm_y
+    Returns:
+        predictions_revin_space: [bs x original_time_len x nvars]
+        predictions_original_space: [bs x original_time_len x nvars]
+    '''
+    # print('CHECK in codes2time - should be TRUE:', vqvae_decoder.training)
+    bs = code_ids.shape[0]
+    nvars = code_ids.shape[1]
+    compressed_len = code_ids.shape[2]
+    num_code_words = codebook.shape[0]
+    code_dim = codebook.shape[1]
+    device = code_ids.device
+    input_shape = (bs * nvars, compressed_len, code_dim)
+
+    with torch.no_grad():
+        # scatter the label with the codebook
+        one_hot_encodings = torch.zeros(int(bs * nvars * compressed_len), num_code_words, device=device)  # one_hot_encodings: [bs x nvars x compressed_pred_len, num_codes]
+        one_hot_encodings.scatter_(1, code_ids.reshape(-1, 1).to(device),1)  # one_hot_encodings: [bs x nvars x compressed_pred_len, num_codes]
+        quantized = torch.matmul(one_hot_encodings, torch.tensor(codebook)).view(input_shape)  # quantized: [bs * nvars, compressed_pred_len, code_dim]
+        quantized_swaped = torch.swapaxes(quantized, 1,2)  # quantized_swaped: [bs * nvars, code_dim, compressed_pred_len]
+        prediction_recon = vqvae_decoder(quantized_swaped.to(device), compression_factor)  # prediction_recon: [bs * nvars, pred_len]
+        prediction_recon_reshaped = prediction_recon.reshape(bs, nvars, prediction_recon.shape[-1])  # prediction_recon_reshaped: [bs x nvars x pred_len]
+        predictions_revin_space = torch.swapaxes(prediction_recon_reshaped, 1,2)  # prediction_recon_nvars_last: [bs x pred_len x nvars]
+        predictions_original_space = revin_layer(predictions_revin_space, 'denorm')  # predictions:[bs x pred_len x nvars]
+
+    return predictions_revin_space, predictions_original_space
+
+
+if __name__ == '__main__':
+    main()
```



- **Added** `steps/STEP4_train_xformer.py`

  - **Affected functions:**

    - `create_dataloader`

    - `inference`

    - `main`

    - `to_categorical`

    - `train`

    - `train_one_epoch`

  - **Diff:**

  ```diff

+import comet_ml
+import torch
+from torch.utils.data import Dataset, DataLoader
+import torch.nn as nn
+import torch.nn.functional as F
+import numpy as np
+import os
+import math
+import pdb
+import argparse
+from lib.models.revin import RevIN
+from lib.models.classif import SimpleMLP, EEGNet, SensorTimeEncoder
+from lib.utils.checkpoint import EarlyStopping
+from sklearn.metrics import confusion_matrix
+from lib.utils.env import seed_all_rng
+from datetime import datetime 
+import json 
+
+
+from omegaconf import DictConfig, OmegaConf
+import hydra
+
+import logging
+
+log = logging.getLogger(__name__)
+
+@hydra.main(config_path="../conf", version_base="1.1")
+def main(cfg: DictConfig) -> None:
+    log.info("STEP4: Train xFormer classifier")
+    log.info(OmegaConf.to_yaml(cfg, resolve=True))
+    log.info(f'Working directory {os.getcwd()}')
+
+    exp_cfg = cfg.exp
+    logging_cfg = cfg.logging
+
+
+    # (6) Setting up the comet logger
+    if logging_cfg.comet:
+        comet_config = logging_cfg.comet
+        # Create an experiment with your api key
+        comet_logger = comet_ml.Experiment(
+            api_key=comet_config['api_key'],
+            project_name=comet_config['project_name'],
+            workspace=comet_config['workspace'],
+        )
+        comet_logger.add_tag(comet_config.comet_tag)
+        comet_logger.set_name(comet_config.comet_name)
+    else:
+        print('PROBLEM: not saving to comet')
+        comet_logger = None
+        pdb.set_trace()
+
+    ## Set device 
+    device = torch.device("cuda:%d" % (exp_cfg.cuda_id))
+    torch.cuda.set_device(device)
+
+    # Log the config parameters to comet_ml
+    comet_logger.log_parameters(exp_cfg["classifier_config"])
+    train(classifier_config=exp_cfg["classifier_config"], comet_logger=comet_logger, device=device)
+
+
+def to_categorical(y, num_classes):
+    """ 1-hot encodes a tensor """
+    return np.eye(num_classes, dtype='uint8')[y]
+
+def create_dataloader(datapath, num_classes=1, batchsize=8):
+    dataloaders = {}
+    for split in ["train", "val", "test"]:
+        x_file = os.path.join(datapath, "%s_x_original.npy" % (split))
+        x = np.load(x_file)
+        x = torch.from_numpy(x).to(dtype=torch.float32)
+
+        y_file = os.path.join(datapath, "%s_x_labels.npy" % (split)) 
+        y = np.load(y_file)
+        if num_classes > 1: 
+            y = to_categorical(y, num_classes=num_classes) 
+        y = torch.from_numpy(y).to(dtype=torch.float32)
+
+        codes_file = os.path.join(datapath, "%s_x_codes.npy" % (split))
+        codes = np.load(codes_file)
+        codes = torch.from_numpy(codes).to(dtype=torch.int64)
+
+        print("[Dataset][%s] %d examples" % (split, x.shape[0]))
+
+        dataset = torch.utils.data.TensorDataset(x, codes, y)
+        dataloaders[split] = torch.utils.data.DataLoader(
+            dataset,
+            batch_size=batchsize,
+            shuffle=True if split == "train" else False,
+            num_workers=1,
+            drop_last=True if split == "train" else False,
+        )
+
+    return dataloaders
+
+
+def train_one_epoch(
+    dataloader,
+    model,
+    codebook,
+    optimizer,
+    scheduler,
+    epoch,
+    device,
+    comet_logger, 
+    normalize_trial=False,
+):
+    running_loss, last_loss = 0.0, 0.0
+    running_acc, last_acc = 0.0, 0.0
+    log_every = max(len(dataloader) // 3, 3)
+    total_loss = 0.0
+    total_acc = 0.0
+    num_batches = 0
+
+    y_pred = []
+    y_true = []
+    for i, data in enumerate(dataloader):
+        # ----- LOAD DATA ------ #
+        x, code_ids, y = data
+        # x: (B, T, S)
+        # codes: (B, TC, S)  TC = T // compression
+        # y: (B,)
+        x = x.to(device)
+        code_ids = code_ids.to(device)
+        y = y.to(device)
+
+        # reshape data
+        B, T, S = x.shape
+        B, TC, S = code_ids.shape
+
+        # get codewords for input x
+        code_ids = code_ids.flatten()
+        xcodes = codebook[code_ids]  # (B*TC*S, D)
+        xcodes = xcodes.reshape((B, TC, S, xcodes.shape[-1]))  # (B, TC, S, D)
+
+        # revin time series
+        norm_x = model.revin(x, "norm")
+
+        if isinstance(model, SimpleMLP):
+            x = x.flatten(start_dim=1)
+            predy = model(x)
+        elif isinstance(model, EEGNet):
+            if normalize_trial: 
+                ## Use the normalized trial as input to the model
+                norm_x = torch.permute(norm_x, (0, 2, 1))  # (B, S, T)
+                norm_x = norm_x.unsqueeze(1)  # (B, 1, S, T)
+                predy = model(norm_x)
+            else: 
+                ## Use the original time series as input to the model
+                x = torch.permute(x, (0, 2, 1))  # (B, S, T)
+                x = x.unsqueeze(1)  # (B, 1, S, T)
+                predy = model(x)
+        elif isinstance(model, SensorTimeEncoder):
+            scale = torch.cat((model.revin.mean, model.revin.stdev), dim=1)
+            scale = torch.permute(scale, (0, 2, 1))
+            predy = model(xcodes, scale)
+        else:
+            raise ValueError("womp womp")
+        loss = F.cross_entropy(predy, y) #F.binary_cross_entropy_with_logits(predy, y)
+
+        # optimization
+        optimizer.zero_grad()
+        loss.backward()
+        optimizer.step()
+
+        # log
+        running_loss += loss.item()
+        with torch.no_grad():
+            prediction_logits = predy.argmax(dim=1)
+            y_logits = y.argmax(dim=1)
+            y_pred.extend(list(prediction_logits.cpu().numpy()))
+            y_true.extend(list(y_logits.cpu().numpy()))
+            # pdb.set_trace()
+            correct = (prediction_logits == y_logits)
+            running_acc += correct.sum().float() / float(y.size(0)) #((predy.sigmoid() > 0.5) == y).float().mean()
+            total_acc += correct.sum().float() / float(y.size(0))
+            total_loss += loss.item()
+            num_batches += 1
+        if i % log_every == log_every - 1:
+            last_loss = running_loss / log_every  # loss per batch
+            last_acc = running_acc / log_every
+            lr = optimizer.param_groups[0]["lr"]
+            # lr = scheduler.get_last_lr()[0]
+            print(
+                f"| epoch {epoch:3d} | {i+1:5d}/{len(dataloader):5d} batches | "
+                f"lr {lr:02.5f} | loss {last_loss:5.3f} | acc {last_acc:5.3f}"
+            )
+            running_loss = 0.0
+            running_acc = 0.0
+
+        if scheduler is not None:
+            scheduler.step()
+    
+    total_acc = total_acc / num_batches # acc per element 
+    total_loss = total_loss / num_batches # loss per element
+
+    comet_logger.log_metric(f'train_acc', total_acc, epoch=epoch)
+    comet_logger.log_metric(f'train_loss', total_loss, epoch=epoch)
+    matrix = confusion_matrix(y_true, y_pred)
+    comet_logger.log_confusion_matrix(title=f'Train Confusion Matrix', matrix=matrix, epoch=epoch, file_name=f'train_confusion_matrix.json')
+
+
+def inference(
+    data,
+    model,
+    codebook,
+    device,
+    normalize_trial=False
+):
+    x, code_ids, _ = data
+    x = x.to(device)
+    code_ids = code_ids.to(device)
+
+    # reshape data
+    B, T, S = x.shape
+    B, TC, S = code_ids.shape
+
+    # get codewords for input x
+    code_ids = code_ids.flatten()
+    xcodes = codebook[code_ids]  # (B*TC*S, D)
+    xcodes = xcodes.reshape((B, TC, S, xcodes.shape[-1]))  # (B, TC, S, D)
+
+    # revin time series
+    norm_x = model.revin(x, "norm")
+
+    if isinstance(model, SimpleMLP):
+        x = x.flatten(start_dim=1)
+        predy = model(x)
+    elif isinstance(model, EEGNet):
+        if normalize_trial: 
+            ## Use the normalized trial as input to the model
+            norm_x = torch.permute(norm_x, (0, 2, 1))  # (B, S, T)
+            norm_x = norm_x.unsqueeze(1)  # (B, 1, S, T)
+            predy = model(norm_x)
+        else: 
+            ## Use the original time series as input to the model
+            x = torch.permute(x, (0, 2, 1))  # (B, S, T)
+            x = x.unsqueeze(1)  # (B, 1, S, T)
+            predy = model(x)
+    elif isinstance(model, SensorTimeEncoder):
+        scale = torch.cat((model.revin.mean, model.revin.stdev), dim=1)
+        scale = torch.permute(scale, (0, 2, 1))
+        predy = model(xcodes, scale)
+    else:
+        raise ValueError("wamp wamp")
+
+    return predy
+
+
+def train(classifier_config, comet_logger, device):
+
+    # -------- SET SEED ------- #
+    print('Setting seed to {}'.format(classifier_config['seed']))
+    seed_all_rng(None if classifier_config['seed'] < 0 else classifier_config['seed'])
+
+    # -------- PARAMS ------- #
+    batchsize = classifier_config['batchsize'] 
+    datapath = classifier_config['data_path'] 
+    expname = classifier_config['exp_name'] 
+    nsensors = classifier_config['nsensors'] 
+    d_out = classifier_config['nclasses'] 
+
+    # -------- CHECKPOINT ------- #
+    checkpath = None
+    if classifier_config['checkpoint']:
+        checkpath = os.path.join(classifier_config['checkpoint_path'], expname)
+        os.makedirs(checkpath, exist_ok=True)
+        os.makedirs(os.path.join(checkpath, 'configs'), exist_ok=True)
+        os.makedirs(os.path.join(checkpath, 'checkpoints'), exist_ok=True)
+    early_stopping = EarlyStopping(patience=classifier_config['patience'], path=checkpath)
+    
+    # # Save the json copy
+    # with open(os.path.join(checkpath, 'configs', 'config_file.json'), 'w+') as f:
+    #     json.dump(classifier_config, f, indent=4)
+
+    # ------ DATA LOADERS ------- #
+    dataloaders = create_dataloader(
+        datapath=datapath, num_classes=d_out, batchsize=batchsize
+    )
+    train_dataloader = dataloaders["train"]
+    val_dataloader = dataloaders["val"]
+    test_dataloader = dataloaders["test"]
+
+    # -------- CODEBOOK ------- #
+    codebook = np.load(os.path.join(datapath, "codebook.npy"), allow_pickle=True)
+    codebook = torch.from_numpy(codebook).to(device=device, dtype=torch.float32)
+    vocab_size, vocab_dim = codebook.shape
+    assert vocab_size == classifier_config['codebook_size']
+    dim = vocab_size if classifier_config['onehot'] else vocab_dim
+
+    # ------- MODEL -------- #
+    if classifier_config['model_type'] == "mlp":
+        # time --> class (baseline)
+        model = SimpleMLP(
+            in_dim=nsensors * classifier_config['Tin'], out_dim=1, hidden_dims=[1024, 512, 256], dropout=0.0
+        )
+    elif classifier_config['model_type'] == "eeg":
+        # time --> class (baseline)
+        model = EEGNet(
+            chunk_size=classifier_config['Tin'],
+            num_electrodes=nsensors,
+            F1=8,
+            F2=16,
+            D=2,
+            kernel_1=64,
+            kernel_2=16,
+            dropout=0.25,
+            num_classes=d_out, 
+        )
+    elif classifier_config['model_type'] == "xformer":
+        # code --> class (ours)
+        model = SensorTimeEncoder(
+            d_in=dim,
+            d_model=classifier_config['d_model'],
+            nheadt=classifier_config['nhead'],
+            nheads=classifier_config['nhead'],
+            d_hid=classifier_config['d_hid'],
+            nlayerst=classifier_config['nlayers'],
+            nlayerss=classifier_config['nlayers'],
+            seq_lent=classifier_config['Tin'] // classifier_config['compression'],
+            seq_lens=nsensors,
+            dropout=0.25,
+            d_out=d_out, 
+            scale=classifier_config['scale'],
+        )
+    else:
+        raise ValueError("Unknown model type %s" % (classifier_config['model_type']))
+    model.revin = RevIN(num_features=nsensors, affine=False)  # expects as input (B, T, S)
+    model.to(device)
+
+    # ------- OPTIMIZER -------- #
+    num_iters = classifier_config['epochs'] * len(train_dataloader)
+    step_lr_in_iters = classifier_config['steps'] * len(train_dataloader)
+    model_params = list(model.parameters())
+    if classifier_config['optimizer'] == "sgd":
+        optimizer = torch.optim.SGD(model_params, lr=classifier_config['baselr'], momentum=0.9)
+    elif classifier_config['optimizer'] == "adam":
+        optimizer = torch.optim.Adam(model_params, lr=classifier_config['baselr'])
+    elif classifier_config['optimizer'] == "adamw":
+        optimizer = torch.optim.AdamW(model_params, lr=classifier_config['baselr'])
+    else:
+        raise ValueError("Uknown optimizer type %s" % (classifier_config['optimizer']))
+    if classifier_config['scheduler'] == "step":
+        scheduler = torch.optim.lr_scheduler.StepLR(
+            optimizer, step_lr_in_iters, gamma=0.1
+        )
+    elif classifier_config['scheduler'] == "onecycle":
+        # The learning rate will increate from max_lr / div_factor to max_lr in the first pct_start * total_steps steps,
+        # and decrease smoothly to max_lr / final_div_factor then.
+        scheduler = torch.optim.lr_scheduler.OneCycleLR(
+            optimizer=optimizer,
+            max_lr=classifier_config['baselr'],
+            steps_per_epoch=len(train_dataloader),
+            epochs=classifier_config['epochs'],
+            pct_start=0.2,
+        )
+    else:
+        raise ValueError("Uknown scheduler type %s" % (classifier_config['scheduler']))
+
+    # ------- TRAIN & EVAL -------- #
+    best_val_loss = float("inf")
+    for epoch in range(classifier_config['epochs']):
+        model.train()
+        train_one_epoch(
+            train_dataloader,
+            model,
+            codebook,
+            optimizer,
+            scheduler,
+            epoch,
+            device,
+            comet_logger=comet_logger,
+            normalize_trial=classifier_config.get('normalize_trial', False)
+        )
+
+        if epoch % 10 == 0:
+            # save the model checkpoints locally and to comet
+            torch.save(model.state_dict(), os.path.join(checkpath, f'checkpoints/model_epoch_{epoch}.pth'))
+            print('Saved model from epoch ', epoch)
+
+        if val_dataloader is not None:
+            model.eval()
+            running_acc = 0.0
+            running_loss = 0.0
+            total_num = 0.0
+            # Disable gradient computation and reduce memory consumption.
+            with torch.no_grad():
+                y_pred = [] 
+                y_true = []
+                for i, vdata in enumerate(val_dataloader):
+                    pred = inference(
+                        vdata,
+                        model,
+                        codebook,
+                        device,
+                        normalize_trial=classifier_config.get('normalize_trial', False)
+                    )
+                    y = vdata[-1]
+                    y = y.to(device)
+
+                    prediction_logits = pred.argmax(dim=1)
+                    y_logits = y.argmax(dim=1)
+                    y_pred.extend(list(prediction_logits.cpu().numpy()))
+                    y_true.extend(list(y_logits.cpu().numpy()))
+                    correct = (prediction_logits == y_logits)
+                    running_acc += correct.sum().float()
+
+                    running_loss += F.cross_entropy(
+                        pred, y, reduction="sum"
+                    )
+                    total_num += y.size(0)
+            curr_acc = running_acc / total_num
+            curr_loss = running_loss / total_num
+            print(f"| [Val] loss {curr_loss:5.3f} | acc {curr_acc:5.3f}")
+            comet_logger.log_metric(f'val_acc', curr_acc, epoch=epoch)
+            comet_logger.log_metric(f'val_loss', curr_loss, epoch=epoch)
+            matrix = confusion_matrix(y_true, y_pred)
+            comet_logger.log_confusion_matrix(title=f'Val Confusion Matrix', matrix=matrix, epoch=epoch, file_name=f'val_confusion_matrix.json')
+
+            if curr_loss < best_val_loss: 
+                best_val_loss = curr_loss
+                torch.save(model.state_dict(), os.path.join(checkpath, f'checkpoints/model_best.pth'))
+                print('Saved best model from epoch ', epoch)
+            
+        if test_dataloader is not None:
+            model.eval()
+            running_acc = 0.0
+            total_num = 0.0
+            # Disable gradient computation and reduce memory consumption.
+            with torch.no_grad():
+                for i, tdata in enumerate(test_dataloader):
+                    pred = inference(
+                        tdata,
+                        model,
+                        codebook,
+                        device,
+                    )
+                    y = tdata[-1]
+                    y = y.to(device)
+
+                    prediction_logits = pred.argmax(dim=1)
+                    y_logits = y.argmax(dim=1)
+                    correct = (prediction_logits == y_logits)
+                    running_acc += correct.sum().float()
+
+                    total_num += y.size(0)
+            running_acc = running_acc / total_num
+            print(f"| [Test] acc {running_acc:5.3f}")
+            comet_logger.log_metric(f'test_acc', running_acc, epoch=epoch)
+
+            if early_stopping.early_stop:
+                print("Early stopping....")
+                return
+    
+    ## Final save
+    torch.save(model.state_dict(), os.path.join(checkpath, 'checkpoints/final_model.pth'))
+
+    ## Log the final model test acc to comet 
+    best_model_path = os.path.join(checkpath, 'checkpoints', 'model_best.pth')
+    state_dict = torch.load(best_model_path)
+    model.load_state_dict(state_dict)
+    model.to(device)
+    model.eval()
+
+    if test_dataloader is not None:
+        model.eval()
+        running_acc = 0.0
+        total_num = 0.0
+        # Disable gradient computation and reduce memory consumption.
+        with torch.no_grad():
+            for i, tdata in enumerate(test_dataloader):
+                pred = inference(
+                    tdata,
+                    model,
+                    codebook,
+                    device,
+                )
+                y = tdata[-1]
+                y = y.to(device)
+
+                prediction_logits = pred.argmax(dim=1)
+                y_logits = y.argmax(dim=1)
+                correct = (prediction_logits == y_logits)
+                running_acc += correct.sum().float()
+
+                total_num += y.size(0)
+        running_acc = running_acc / total_num
+        print(f"| [Test best model] acc {running_acc:5.3f}")
+        comet_logger.log_metric(f'best_model_test_acc', running_acc)
+
+
+if __name__ == "__main__":
+    main()
```



- **Added** `steps/__init__.py`

  - **Diff:**

  ```diff

```



---


