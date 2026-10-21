# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

`docs/design-spec.md` is the source of truth for what we build and the hard-to-reverse decisions behind it. Record a new decision or term there, in the section it shapes; this repo keeps no `docs/adr/` and no `GLOSSARY.md`. `docs/ROADMAP.md` is the source of truth for scope: each feature's tier and what is out of scope.

## Before exploring, read

- **`docs/ROADMAP.md`**: the feature's row, its issue, and the Limits that apply to it.
- **`docs/design-spec.md`**: the sections that touch the area you're about to work in.

## Flag spec conflicts

If your output contradicts `docs/design-spec.md` or `docs/ROADMAP.md`, surface it explicitly rather than silently overriding:

> _Contradicts the design spec (one model, ops dispatched by device), but worth reopening because…_
