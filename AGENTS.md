# small-vLLM

C++/CUDA LLM inference engine on LibTorch. What we build and why: `docs/design-spec.md`. Scope, by tier: `docs/ROADMAP.md`. This file is the workflow and coding standards for people and agents alike; follow it.

## Commands

```bash
uv sync --extra cu126                  # GPU machine; `--extra cpu` without a GPU
uv run pre-commit install              # once: format and lint on every commit
uv run cmake --preset release          # presets: release, debug, relwithdebinfo, cpu (no CUDA)
uv run cmake --build --preset release
uv run ctest --preset release
SVLLM_BUILD_DIR=build/release uv run pytest
uv run pre-commit run --all-files
```

Without an NVIDIA GPU (e.g. macOS), only the `cpu` preset builds; GPU tests skip themselves.

## Layout

```
src/ops/         op declarations + device dispatch; kernels in cuda/ and cpu/
src/model/       config, safetensors loader, model forward (Qwen3; Llama 3.2 via config flags)
src/engine/      Device, block manager, prefix cache, scheduler, KV offload, engine loop
src/tokenizer/   tokenizers-cpp wrapper
src/apps/        svllm-bench and svllm-serve entry points
bindings/        pybind11 module svllm_ops, for tests and benchmarks only
tests/cpp/       googletest: engine and model logic; CUDA cases skip without a GPU
tests/python/    pytest: ops vs PyTorch, end to end vs Hugging Face
bench/           kernel and throughput benchmarks, plotting
scripts/         model download, environment checks
cmake/           torch discovery, OpenMP, pinned third-party dependencies
```

## Workflow

- Before starting an issue, check that no open PR already covers it: `gh pr list --search "<number> in:body"`.
- Branch off `main` as `<github-handle>/<topic>`. Every change lands through a PR, squash-merged, so the PR title becomes the commit: Conventional Commits without a scope, e.g. `feat: preemption by recompute`.
- CI has no GPU. For PRs touching kernels, the model or the engine, run the GPU tests on a GPU machine and paste the output into the PR with that of `scripts/check_env.sh` and the build preset.
- Record a decision that is hard to reverse in `docs/design-spec.md`, with the reason.

## Coding standards

- C++20, everything in namespace `svllm`, following the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) except that functions and variables are `snake_case`, as in LibTorch. So: files `snake_case`, types `PascalCase`, private members end in `_`, constants and enumerators `kPascalCase`. `.clang-format` enforces the layout. Inside a function, separate the steps with a blank line: input checks, setup, the work, the return.
- Python (tests, benchmarks, scripts) follows ruff through pre-commit.
- `.cu` for kernels, `.cuh` only for device code shared between `.cu` files, `.h`/`.cpp` for everything else.
- Check inputs with `TORCH_CHECK`; host code may throw. No owning raw pointers: hold a resource in `std::unique_ptr` (with a deleter for a C handle) or a container, so a class needs no destructor or copy/move code (Rule of Zero). Raw pointers that only borrow, like kernel arguments and `data_ptr()`, are fine.
- An op is declared in `src/ops/<op>.h`, with kernels in `src/ops/cuda/<op>.cu` and `src/ops/cpu/<op>.cpp`, and dispatches on its input's device. An op ships both kernels in one PR and errors on an unsupported device. Attention is the exception: its CPU kernels come with the CPU backend, and until then it errors on CPU.
- Read model sizes from `config.json`. Shapes, paths and GPU architectures come from config or arguments.
- C++ deps: FetchContent in `cmake/dependencies.cmake`, pinned to a release tarball, or a commit tarball when no release builds, and its SHA256. Python deps: `uv add`, and commit `uv.lock`.

### Names

| Name | Meaning |
| --- | --- |
| `Request` | One prompt and its generated tokens, from `add_request` until it finishes |
| `num_seqs` | Requests in a step's batch; a kernel's batch dimension |
| `num_tokens` | Tokens in a step's batch, across all requests |
| `num_heads`, `num_kv_heads`, `head_dim` | Query heads, KV heads, size of one head |
| `block_size` | Tokens per KV block |
| `block_table` | A request's KV block ids, in order |
| `slot_mapping` | Per token, the KV slot it writes: `block_id * block_size + offset` |
| `seq_lens` | Per request, tokens in the KV cache including this step's |
| `query_start_loc` | Prefix sums of the per-request query lengths (flash-attn's `cu_seqlens_q`) |

Counts start with `num_`, lengths end in `_len`/`_lens`, limits start with `max_`, ids end in `_id`/`_ids`.

### Comments

Comment what the code can't say: the reason, an invariant, a constraint from elsewhere.

- Each public declaration in a header states its contract: what it does, tensor shapes, dtype and device, and what it throws (see `src/model/config.h`).
- Tensor shapes use the names above in brackets: `// [num_tokens, num_heads, head_dim]`.
- A TODO names its issue: `// TODO(#25): batch prefill chunks with decodes`.
- Delete dead code; git keeps it.

## Tests

Test where it pays off:

- **Each op:** one parametrized pytest against PyTorch at bf16 tolerance over a few shapes, including GQA ratios and block-size edges.
- **The engine:** one end-to-end test that greedy output matches Hugging Face on a few prompts.
- **Block manager and scheduler:** googletest cases for the tricky paths (prefix hits, eviction, admission when blocks run short).
- Getters, glue code and CLI flags need no tests.
- **A bug fix** comes with a test that fails on `main` and passes with the fix; paste both runs in the PR.
- **Fix the code, not the test.** A tolerance, skip or assertion changes only with the reason in the PR.

## Rules

- Build only what `docs/ROADMAP.md` puts in scope. For a feature it lists as out of scope, or doesn't list, stop and say so; moving it into scope is a PR that edits the roadmap. Bug fixes, refactors, and doc or tooling changes with their own issue need no row.
- You must be able to explain every line you submit. Don't add code the task doesn't need.
- Change only what the task needs. If you change an interface another module calls (an `ops::` signature, the `Model` or `Engine` API), list it under "Interface changes" in the PR description.
- If your change contradicts `docs/design-spec.md`, say so in the PR instead of silently overriding it.
- Never commit model weights or files over 1 MB.
- Never add AI attribution to commits or PRs: no `Co-Authored-By` trailer for an AI agent, no "Generated with" line.
- When editing `AGENTS.md` or `docs/agents/`: keep `AGENTS.md` under 150 lines, link to a source instead of copying it, and remove or merge a rule for each one you add.

## Agent skills

### Issue tracker

Issues live in GitHub Issues for Yu-TingLee/small-vLLM, via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Domain docs

Decisions live in `docs/design-spec.md`, scope in `docs/ROADMAP.md`; no ADRs, no glossary. See `docs/agents/domain.md`.
