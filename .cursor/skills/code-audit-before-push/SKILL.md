---
name: code-audit-before-push
description: Audits local git diffs for real bug risk before pushing to remote. Use when the user asks for a pre-push check, code audit, diff review, release sanity check, or mentions pushing code.
---

# Code Audit Before Push

## Purpose

Run a focused pre-push quality audit that checks what changed, finds likely mistakes, and flags questionable logic without overengineering.

## When To Use

- Use when the user explicitly requests this skill (for example: `/code-audit-before-push`).

## Audit Principles

- Prioritize real failures: correctness, data loss, security, crashes, broken contracts, and obvious performance regressions.
- Review the full diff that will be pushed, not only the latest file or commit.
- Keep feedback concise and actionable.
- Do not block on style-only issues unless they can hide bugs.
- If an issue is uncertain or low impact, ask whether the user wants deeper investigation or a fix proposal.

## Workflow

1. Gather git context in parallel:
   - `git status --short`
   - `git diff --staged && git diff`
   - `git log --oneline --decorate -n 12`
2. Identify changed areas with highest risk first:
   - Data/model/schema changes
   - Concurrency/async/error handling
   - Auth/permissions/secrets/input validation
   - IO boundaries (filesystem, network, database)
3. Inspect changed code for:
   - Behavior regressions and edge-case handling
   - Incorrect assumptions, null/empty paths, off-by-one/indexing issues
   - Missing cleanup/rollback/error propagation
   - Dangerous defaults or silent fallback behavior
   - Tests missing for changed critical paths
4. Produce findings in severity order.
5. If there are only minor/uncertain findings, ask if the user wants fixes now.

## Output Format

Use this structure:

```markdown
## Findings
- Critical: <issue and impact> (`path`)
- Warning: <issue and impact> (`path`)
- Note: <questionable area or test gap> (`path`)

## Questions
- <only if uncertainty exists or trade-off needs confirmation>

## Push Readiness
- Ready to push: Yes/No
- Why: <one sentence>
```

Rules for output:
- If no issues are found, explicitly state: `No critical or warning findings.` and list residual risk/test gaps as notes.
- Keep each finding to one short bullet with impact.
- Prefer concrete evidence from the diff.

## Boundaries

- Do not invent architecture changes during audit.
- Do not auto-commit or push unless user asks.
- Do not claim certainty when evidence is weak; ask a clarifying question instead.
