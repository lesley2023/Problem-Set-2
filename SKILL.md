---
name: tp
description: "Traceable Prompt (TP) skill for BUSFIN 8200 problem sets. Invoke @TP at the start of every substantive AI request (checking/critiquing math or economic reasoning, empirical coding, debugging, formatting/translation) to comply with the course AI policy. Creates git commits before/after the interaction and appends an auditable entry to AI_INTERACTIONS.md."
---

# TP (Traceable Prompt)

Create a reusable AI skill called @TP, short for traceable prompt, for this problem set project. The purpose of this skill is to help me comply with the BUSFIN 8200 problem set AI policy. When I invoke @TP at the beginning of a substantive AI request, do the following.

1. First, identify the problem set item to which my request relates. If the item is not clear, ask me to specify it before doing the substantive work.
2. Before doing the substantive work, create a Git commit that records the current state of the repository. If there are no file changes to commit, create an empty commit so that the state before the interaction is still recorded. Record the commit hash for the state before the interaction.
3. Complete my substantive request, subject to the course AI policy. If the request asks you to make a substantive mathematical, economic, or empirical design decision that I have not specified, identify the ambiguity and ask me to decide before implementing it.
4. After completing the substantive work, append a new entry to AI_INTERACTIONS.md. Do not modify, delete, combine, or rewrite previous entries.
5. The new entry in AI_INTERACTIONS.md must include the problem set item; my substantive prompt; the purpose of the request; the Git commit before the interaction; a concise but complete description of the assistance you provided; the files you inspected; the files you directly modified, if any; any errors, omissions, or ambiguities you identified; any substantive mathematical, economic, or empirical suggestions you made; whether the interaction involved checking mathematics, checking economic reasoning, empirical implementation, code debugging, formatting/translation, or another form of assistance; and whether any minor subsequent debugging or formatting requests were grouped into this same interaction.
6. After updating AI_INTERACTIONS.md, create another Git commit recording the state of the repository after the interaction. If there are no file changes to commit, create an empty commit so that the state after the interaction is still recorded.
7. If I explicitly group closely related minor subsequent debugging or formatting requests into the same interaction, document them in the same AI_INTERACTIONS.md entry. Only group subsequent requests when they concern the same problem set item, occur in the same work session, and are documented together.

The skill should help create an auditable record. It should not weaken or replace any requirement in the course AI policy.