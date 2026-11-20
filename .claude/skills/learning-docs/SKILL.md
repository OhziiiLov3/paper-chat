---
name: learning-docs
description: Create or update teaching notes in documentation/ for a code folder in this AI fundamentals project, following the project's standard study-guide template.
argument-hint: "<folder-name>"
---

# Learning Docs

Create teaching notes for the code folder `$ARGUMENTS` and save them as `documentation/$ARGUMENTS.md`.

If `$ARGUMENTS` is empty, list the code folders that don't have a matching file in `documentation/` yet, and ask which one to document.

## Audience and voice

- Write as an experienced teacher who trains working professionals in AI fundamentals.
- The notes are for reviewing a lesson later, studying, and teaching or training others.
- Keep them concise and clear. Use plain language and avoid heavy jargon.
- When a technical term is needed, explain it in a short phrase the first time it appears.

## Steps

1. Read **every** file in `$ARGUMENTS/` in full.
2. Read one or two existing files in `documentation/` and match their structure and tone exactly.
3. If `documentation/$ARGUMENTS.md` already exists, read it and update it rather than replacing it. Keep sections that are still accurate.
4. Write the notes using the template below.
5. Check what you can. If a claim can be checked locally (for example, the output of a tokenizer or a validation error message), run a short script with `.venv/bin/python` and use the real output. Don't make paid API calls to check things.
6. **Never edit the code files.** Only write to `documentation/`.

## Template

```markdown
# <Topic>: <Short Plain-English Subtitle>

**Folder:** `<folder>/` (`file1.py`, `file2.py`)

## What you'll learn
- 3-5 bullets, each a concrete skill or idea

<1-2 sentence plain-language intro: what this is and the problem it solves>

---

## 1. <First concept>
Short explanation, then a minimal code example taken from the user's files.
Use a table when comparing options or explaining parameters.
Use a `> **Key point:**` callout for the single most important idea.

---

## 2. <Next concept> ... (one numbered section per concept, in the order the code introduces them)

---

## Key ideas
- 4-6 one-line takeaways

## Common mistakes
- Real errors from the code or from building it, plus well-known pitfalls

## Practice
1. 3-5 hands-on exercises that build on the user's code

## Teaching tip
One short paragraph: a demo or contrast that makes the main idea stick for learners.
```

## Content rules

- Base code examples on the user's actual files. Fix typos and spelling in the notes, but keep the logic the same.
- Explain *why* as well as *how*. Connect each concept to real AI application design.
- Link to related notes with relative links, for example `[prompt-engineering.md](prompt-engineering.md)`, when a concept builds on an earlier lesson.
- Don't state prices, model availability, or API limits as current facts unless you checked them. Note that they may change.
- Leave out anything you're unsure of, or mark it clearly as unverified.

## After writing

Reply to the user with:
- One line naming the audience (for example, "Written for: you and the professionals you'll train.")
- A short summary of what the notes cover
- Any bugs, typos, or differences you noticed between the code and the notes, with clickable file and line links. Don't fix the code.
- Anything you couldn't check
