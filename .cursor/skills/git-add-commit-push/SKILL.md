---
name: git-add-commit-push
description: Runs git add -A (everything except .gitignore), writes a short subject plus detailed body commit message from git status/diff, commits, and pushes the current branch to the remote. Use when the user asks to push to GitHub, commit and push, sync changes, or similar one-shot git workflows.
---

# Git add, commit, and push

## When this applies

The user wants to ship local work in one flow: stage → commit → push. Typical prompts: "push on github", "commit and push", "sync my changes", "add everything and push".

## Workflow

1. **Inspect** — Run `git status` and `git diff` (unstaged). If there are staged changes, also run `git diff --cached`. Use this to understand *what* changed, not only filenames.
2. **Stage** — Default: from the repo root, run **`git add -A`** so **all** changes are staged: new files, modifications, and removals. Git skips anything listed in **`.gitignore`** (and global excludes); do **not** hand-pick paths to leave out (e.g. do not skip `.cursor/` or other folders) unless the user explicitly asked for a **partial** commit or named specific paths. If they only want certain files, stage only those paths instead.
3. **Commit message** — Must have **two parts**:
   - **Short subject** (first line): imperative mood, concise, one line. Aim for ≤72 characters. States *what* the change does in plain language (optionally `type(scope):` prefix if the repo already uses Conventional Commits — infer from recent `git log`).
   - **Blank line**, then **long body**: paragraphs or bullet list describing *what* was updated and *why*, derived from the diff/status. Mention notable files or areas when helpful.
4. **Commit** — Use a heredoc or `git commit -m` for the subject and `-m` again for the body, or `git commit -F` with a temp file so the body can be multi-line without shell escaping issues.
5. **Push** — `git push -u origin HEAD` if upstream is missing, otherwise `git push` on the **current branch**. Do not change branch unless the user asked.

## Commit message template

```
Short subject in imperative mood

Longer explanation of the update:
- Key change or file group
- Another bullet if useful

Optional: breaking changes, follow-ups, or context that is not obvious from the diff.
```

## Rules

- **Stage everything by default** — Use `git add -A`; rely on `.gitignore` for exclusions. Do not exclude untracked folders from the commit unless the user asked for a scoped add.
- **Do not** `--force` push to `main`/`master` or rewrite shared history unless the user explicitly requests it.
- If there is **nothing to commit** after staging, report that and skip commit/push (or only push if there are unpushed commits — check with `git status` / `git log @{u}..HEAD` when relevant).
- If **multiple unrelated** change sets are mixed and the user did not ask for one squashed commit, suggest splitting or ask which scope to commit; if they want one shot, one commit with a body that lists logical groups is acceptable.

## Quick reference

| Step   | Command idea |
|--------|----------------|
| Inspect | `git status`, `git diff`, `git diff --cached` |
| Stage   | `git add -A` (default; honors `.gitignore`) or explicit paths if user scoped the request |
| Push    | `git push` or `git push -u origin HEAD` |
