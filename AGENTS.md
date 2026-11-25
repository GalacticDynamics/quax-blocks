# quax-blocks — Agent Instructions

`quax-blocks` provides reusable **mixin building blocks** for authoring
[`quax`](https://github.com/patrick-kidger/quax) `ArrayValue` classes: operator
support (arithmetic, bitwise, comparison, rounding, container, copy) implemented
on top of [`quaxed`](https://github.com/GalacticDynamics/quaxed) so the
operators dispatch through quax.

For using quax-blocks mixins from _outside_ this repo (composing mixins onto
your own `ArrayValue`, the Lax-vs-NumPy semantics contract, the `__eq__` gotcha,
troubleshooting), read
[skills/quax-blocks/SKILL.md](skills/quax-blocks/SKILL.md). This file is for
working inside the repo.

## Essential Commands

```bash
uv run pytest                    # run the full suite (incl. Sybil doctests in src/ and tests/)
uv run prek run --all-files      # lint + format (ruff, mypy, taplo, codespell, prettier, ...)
uv run pylint quax_blocks        # pylint (also available as `uv run nox -s pylint`)
uv run nox -s lint               # precommit + pylint in an isolated env
uv run nox -s test               # tests via nox
uv run nox -s docs -- --serve    # build & serve the docs (zensical)
```

> Always use `uv run` — never bare `python`/`pytest`. Sync first with
> `uv sync --group dev --locked`.

## Code graph

This repository has a committed graphify knowledge graph at
`graphify-out/graph.json`: every symbol, the links between them, the
architectural hubs, and which subsystem each symbol belongs to. It is built from
the source by tree-sitter, so it is as current as the last commit.

Querying it is cheaper and more accurate than grepping. The rules:

- For any question about this codebase, query the graph **before** your first
  grep, glob or source-file read:
  - `graphify query "<question>"` — a scoped subgraph for a plain-language
    question
  - `graphify explain "<symbol>"` — one symbol and its neighbours
  - `graphify path "<A>" "<B>"` — how two things connect
  - `graphify-out/GRAPH_REPORT.md` — hubs and subsystems, for broad architecture
    review only

  Read source files for the lines a query has already located, not to find them.

- Put this rule in every subagent prompt you write that explores the code. A
  subagent that greps from scratch undoes the point of the graph.
- After changing code, refresh it: `PYTHONHASHSEED=0 graphify update . --force`
  — tree-sitter only, no model call, no network. `--force` keeps your rebuild
  equal to the one the commit hook and CI produce; without it the rebuild is
  incremental and drifts.
- Ignore `built_at_commit` in the graph, and `GRAPH_REPORT.md`'s "built from
  commit … compare to check if the graph is stale". `update` does not rewrite
  that field, so it names an old commit for ever and is not a staleness signal.
  Do not rebuild on the strength of it.
- Use the pinned release, the one in `.pre-commit-config.yaml`, so your rebuild
  clusters the same way the hook does:
  `uvx --from graphifyy==0.9.119 graphify ...`.
- `graphify-out/graph.json` is generated. Do not read it directly, do not review
  it in a diff, and do not hand-resolve a conflict in it: rebuild it instead.
- A rebuild that rewrites the graph in this checkout is **expected** if the
  package has been installed or the tests have been run here: graphify's output
  depends on which files are on disk, and a generated `_version.py` is enough to
  move community numbering. Stage it and commit; it is not a bug and not a
  reason to skip the hook.
- Skip the graph only when the task is about the graph tooling itself being
  stale or wrong, or when asked to. An uncommitted or just-rebuilt graph file is
  normal and is not a reason to skip it.

Setup (the one-time merge driver) and the day-to-day escapes are in
[.github/CONTRIBUTING.md](.github/CONTRIBUTING.md).

## Architecture

Public API is re-exported from `src/quax_blocks/__init__.py`; implementation
lives in `src/quax_blocks/_src/`. Every operator comes in two flavors —
**`Lax*`** (backed by `quaxed.lax`) and **`Numpy*`** (backed by `quaxed.numpy`)
— and each mixin is `Generic[T, R]` (`T` = other-operand type, `R` = return
type).

| Module                                                  | Provides                                                                                                                                                                                       |
| ------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [\_src/rich.py](src/quax_blocks/_src/rich.py)           | Rich comparison mixins: `__eq__`, `__ne__`, `__lt__`, `__le__`, `__gt__`, `__ge__` (+ `Lax/NumpyComparisonMixin` aggregates)                                                                   |
| [\_src/binary.py](src/quax_blocks/_src/binary.py)       | Binary arithmetic & bitwise mixins (`add`/`sub`/`mul`/`matmul`/`truediv`/`floordiv`/`mod`/`divmod`/`pow`, shifts, `and`/`or`/`xor`), each with forward, reflected (`R*`), and `Both*` variants |
| [\_src/unary.py](src/quax_blocks/_src/unary.py)         | Unary mixins: `__pos__`, `__neg__`, `__invert__`, `__abs__`                                                                                                                                    |
| [\_src/round.py](src/quax_blocks/_src/round.py)         | `__round__`, `trunc`, `floor`, `ceil` mixins                                                                                                                                                   |
| [\_src/container.py](src/quax_blocks/_src/container.py) | `HasShape` protocol and container mixins (`__len__`, `__length_hint__`)                                                                                                                        |
| [\_src/copy.py](src/quax_blocks/_src/copy.py)           | Copy mixins (`__copy__`, `__deepcopy__`)                                                                                                                                                       |
| [\_src/example.py](src/quax_blocks/_src/example.py)     | `AbstractVal` — a minimal example `quax.ArrayValue` used throughout the docstrings/tests                                                                                                       |

## Composing a type

Mix the blocks you need onto a `quax.ArrayValue` subclass (see the docstring
examples in each module). The user-facing material — a worked example, the
Lax-vs-NumPy semantics contract, the `__eq__` reassignment gotcha, class-var
knobs like `_ROUNDING_METHOD`, and a troubleshooting table — lives in
[skills/quax-blocks/SKILL.md](skills/quax-blocks/SKILL.md); it is the single
source of truth for that, do not duplicate it here.

## Conventions & pitfalls

- **uv-native**: dependencies use PEP-735 dependency-groups; CI runs
  `uv sync --locked` + `prek`. Regenerate the lock with `uv lock` after editing
  `pyproject.toml`.
- **Doctests are tests**: `testpaths` includes `src`, and Sybil executes the
  `>>> ` examples in docstrings — keep them correct and deterministic. This does
  not extend to Markdown (`skills/`, `docs/`, `README.md`); Sybil only collects
  `*.py`/`*.rst`.
- **`# fmt: off` blocks**: the big `__all__` lists (and their duplicates across
  modules) are intentionally hand-formatted; don't reflow them.
- **Two flavors stay in sync**: when adding/altering an operator, update both
  the `Lax*` and `Numpy*` mixin.
- **Every dispatch call needs the `NotImplemented` guard**: wrap `qlax.*`/
  `qnp.*` calls in `try/except DISPATCH_ERRORS: return NotImplemented` (see
  `_src/_compat.py`); a missing guard breaks reflected-operand dispatch for the
  _other_ type. Unary mixins (no other operand) call unguarded.
- **Type checking**: `mypy --strict` and `pyright` both run via pre-commit,
  scoped to `src/` (mypy) and the paths in `[tool.pyright]` (pyright); type
  errors elsewhere (e.g. `tests/`) may not be caught by both.
- **`qnp`/`qlax` are typed `Any` at check time**: quaxed's plain-JAX annotations
  can't model quax's dispatch, so a typo like `qnp.subtractt` passes type
  checking silently — only
  [tests/test_quaxed_names.py](tests/test_quaxed_names.py)'s runtime scan
  catches it.

## Further Reading

- [skills/quax-blocks/SKILL.md](skills/quax-blocks/SKILL.md) — using quax-blocks
  mixins from outside this repo
- [.github/skills/code-review/SKILL.md](.github/skills/code-review/SKILL.md) —
  what to look for when reviewing a quax-blocks change (also picked up by GitHub
  Copilot code review)
- [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md) — setup and workflow
- [docs/guides/mixins.md](docs/guides/mixins.md) — full mixin reference table
- [docs/](docs/) — full documentation
