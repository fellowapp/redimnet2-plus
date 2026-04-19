---
name: gacp
description: Performs a fast one-shot git add, one-line commit, and push flow for small experiment changes. Use when user wants quick commits for tiny edits and does not want a commit message body.
---

# Fast git add, commit, and push

## When this applies
Use when the user wants to quickly ship a small local change for experiments.

## Workflow
1. Check pending work:
   - `git status --short`
   - `git diff --name-only` (optional: confirm what changed)
2. If there are no changes, stop and report `No changes to commit.`
3. Stage everything:
   - `git add -A`
4. Create a short, one-line commit title from the staged diff and commit:
   - `CHANGED="$(git diff --cached --name-only | head -n 3 | paste -sd ',' -)"`
   - `TITLE="exp: ${CHANGED:-quick experiment patch}"`
   - `git commit -m "${TITLE}"`
5. Push current branch:
   - `if git rev-parse --abbrev-ref --symbolic-full-name @{u} >/dev/null 2>&1; then git push; else git push -u origin HEAD; fi`

## Rules
- Keep commit title on one line only.
- Do not add a commit message body.
- Do not use `--force` unless explicitly requested.
- Do not skip `.gitignore` behavior: `git add -A` is required unless the user requested scoped files.
- Do not run tests or additional validation for this skill unless explicitly requested.
