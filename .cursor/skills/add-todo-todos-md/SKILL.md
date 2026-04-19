---
name: add-todo-todos-md
description: Appends new checklist items to TODOs.md at the repo root with correct Markdown sections and formatting. Use when the user asks to add a TODO, backlog item, or task to TODOs.md; references @TODOs.md or TODOs.md; uses a slash command like /add-todo; or says to track work in the project TODO file.
---

# Add TODO to `TODOs.md`

## When this applies

The user wants a new open item recorded in **`TODOs.md`** (repository root: `TODOs.md`), including via `@TODOs.md` mentions or prompts like “add a todo”, “put this on the backlog”, or “track in TODOs”.

## Workflow

1. **Read** `TODOs.md` — Confirm current structure, avoid duplicate lines, and see which section fits.
2. **Normalize the text** — One short, actionable phrase (imperative or noun phrase, consistent with existing bullets). Fix typos; do not change the user’s meaning.
3. **Choose the section**
   - **`## Open items (tracked todos)`** — Default for active or near-term work.
   - **`## Backlog`** — Use when the user says *backlog*, *later*, *someday*, or *low priority*, or when the item is clearly exploratory/nice-to-have vs current sprint-style tasks.
   - If unclear, use **Open items** and optionally ask once whether it belongs in Backlog instead.
4. **Insert** — Add a new line immediately **after the last** `- [ ]` line in that section (keep items grouped; do not insert into the Mermaid block or above `## Overview`).
5. **Format** — Exactly:

   ```markdown
   - [ ] Your todo text here
   ```

   Single space after `]`. No nested checkboxes unless the file already uses them in that section.

6. **Preserve** — Do not edit unrelated sections (intro, architecture tables, locked plan notes) or remove existing todos unless the user explicitly asked.

## Rules

- **No duplicates** — If the same idea already exists (exact or near-duplicate), point it out and do not add again unless the user wants a reworded duplicate.
- **One item per request** unless the user listed several; then add each as its own `- [ ]` line in order.
- **Do not** alter `AGENTS.md`, `configs/**`, or filter probabilities unless the user’s todo text explicitly requires it (this skill is only for editing `TODOs.md`).

## Quick reference

| Goal        | Section            |
|-------------|--------------------|
| Default     | Open items (tracked todos) |
| Later / nice-to-have | Backlog      |
