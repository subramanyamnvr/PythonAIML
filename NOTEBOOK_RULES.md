# Notebook Rules

Use these rules for all notebook creation and notebook edits in this project.

## Core Goal

Write notebooks so they teach for real understanding, not just surface familiarity.

A good notebook should help the reader answer:

- what the code means
- what objects exist at runtime
- what changes in memory
- what the interpreter is doing
- why surprising behavior happens

## Content Rules

For each important topic:

- explain the topic in plain English first
- then deepen it with runtime-level explanation
- then deepen it with memory/reference-level explanation
- then deepen it with bytecode or interpreter-level explanation when useful
- finish with a revision-style takeaway

Do not stop at syntax.

Prefer mechanism over slogans.

## Structure Rules

Notebook format should stay simple:

- title
- descriptive teaching text
- runnable code

Avoid too many subheadings.

Do not use repetitive template-like section structures unless they truly help.

The notebook should read more like thoughtful chapter notes than generated outline content.

## Writing Style

Write in a human teaching voice.

Prefer:

- clear
- direct
- reflective
- patient
- practical

Avoid self-referential writing such as:

- "this notebook has been rewritten"
- "in this section we will"
- "let us now"
- "this section explains"

Do not narrate the document structure more than necessary.

Focus on teaching the topic itself.

## Depth Rules

For major ideas, include as many of these as relevant:

- object identity
- binding and rebinding
- mutability
- frame behavior
- code object inspection
- bytecode inspection
- protocol/data-model translation
- CPython-specific notes where clearly relevant
- common bugs and failure cases

Useful tools include:

- `id()`
- `type()`
- `__dict__`
- `__slots__`
- `__code__`
- `co_varnames`
- `co_consts`
- `co_names`
- `co_freevars`
- `co_cellvars`
- `__closure__`
- `inspect.currentframe()`
- `dis.dis()`
- `sys.getrefcount()`

## Code Rules

All notebook code should be runnable top to bottom.

Each code block should prove or reveal an idea.

Avoid filler code that only decorates the notebook without teaching something.

Prefer:

- small examples
- revealing examples
- introspection-friendly examples
- failure-case examples

When a helper is needed, make the notebook self-contained enough that execution does not depend on hidden state from another notebook.

## Revision Rule

Every notebook should end with:

- `Final Takeaway`

That section should act like chapter revision notes, not just a casual summary.

It should help the reader quickly recall:

- the runtime model
- the main mental picture
- the common trap
- the most important idea to remember

## Anti-Patterns To Avoid

Do not:

- add repeated environment/version cells in every notebook
- overuse generic subheadings
- make notebooks feel obviously machine-generated
- keep notebooks shallow just to keep them short
- rely on bullet lists when prose explanation is needed
- push code that has not been executed

## Quality Rule

After notebook edits:

- run all affected notebooks
- fix runtime issues
- remove temporary artifacts created only for editing or execution

## Project Rule

Apply these rules across the project for future notebook work unless the user gives more specific instructions that override them.
