<!-- Title: Conventional Commits without a scope, e.g. `feat: preemption by recompute`.
     The squash merge makes it the commit subject. -->

## What and why

<!-- 2-3 sentences: what changed, why, and its effect.
     Link the issue with "Closes #N" if there is one. -->

## Interface changes

<!-- Changes to anything another module calls: an ops:: signature, the Model or Engine API.
     If the change contradicts docs/design-spec.md, say where and why. -->
None

## Testing

- [ ] I added or updated tests for the changed logic
- [ ] Existing tests already cover the changed logic
- [ ] No logic changed (docs, comments, config)

<!-- CI has no GPU. For kernel, model or engine changes, paste the GPU test output and name
     the GPU, driver, CUDA and torch versions and the build preset. -->

## Checklist

- [ ] I can explain every line of this change
- [ ] `uv run pre-commit run --all-files` passes
- [ ] A hard-to-reverse decision is recorded in `docs/design-spec.md`, or there is none
- [ ] No model weights, no files over 1 MB, no AI attribution
