# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

This repo keeps **no `GLOSSARY.md`**. Decisions are recorded as ADRs only. Don't create a glossary.

## Before exploring, read these

- **`docs/adr/`**: read ADRs that touch the area you're about to work in.

If `docs/adr/` doesn't exist, **proceed silently**. Don't flag its absence; don't suggest creating it upfront. ADRs are created lazily when a hard-to-reverse decision actually gets made.

## File structure

```
/
├── docs/adr/
│   ├── 0001-libtorch-tensors.md
│   └── 0002-qwen3-primary-model.md
└── src/
```

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0001 (LibTorch tensors), but worth reopening because…_
