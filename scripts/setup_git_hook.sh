#!/bin/bash

REPO_DIR="/home/triniborrell/home/projects/TOTEM_for_EEG_code"
HOOK_DIR="$REPO_DIR/.git/hooks"
HOOK_PATH="$HOOK_DIR/post-commit"

# Create the post-commit hook
cat > "$HOOK_PATH" << 'EOF'
#!/bin/bash
# Post-commit hook to update documentation

export GIT_HOOK_RUNNING=true
python /home/triniborrell/home/projects/TOTEM_for_EEG_code/scripts/document_changes.py

# Don't commit the generated files in this hook to avoid infinite loop
# They'll be committed in next commit
EOF

# Make the hook executable
chmod +x "$HOOK_PATH"

echo "Git post-commit hook installed successfully!"
echo "Documentation will be automatically generated after each commit."