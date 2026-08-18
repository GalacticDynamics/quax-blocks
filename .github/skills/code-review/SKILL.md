---
name: code-review
description:
  Use when reviewing a pull request or diff in the quax-blocks repository.
  Covers the quax-blocks-specific defects that generic review misses — a mixin
  missing its NotImplemented dispatch-miss guard, Lax/Numpy flavours drifting
  out of sync, an intentional Lax-semantics divergence getting "fixed" into a
  regression, `__eq__` shadowing gaps, and `__all__`/export lists left stale.
---

# Reviewing quax-blocks changes

`quax-blocks` is a library of small mixin classes, each implementing one Python
dunder method by calling into `quaxed.lax` or `quaxed.numpy` and catching a
dispatch miss. The entire defect surface is narrow and repetitive by design —
nearly every real bug in this repo has been one of a handful of shapes, listed
below.

## Scope of this review

Leave these alone — they are already gated or already covered:

- Formatting, import order, naming, line length. `prek` runs ruff, mypy
  (`--strict` on `src/`), pyright, taplo, and codespell on every commit.
- Generic security checklists. There is no user input, no network, no
  serialisation of untrusted data — this is a pure operator-dispatch library.
- Whether a _covered_ mixin's happy path returns the right value — its docstring
  doctest already asserts that and runs in CI (`testpaths` includes `src`, and
  `conftest.py`'s sybil config collects `>>>` examples from every `.py` file).
  See [Doctests are real tests](#doctests-are-real-tests) for what that does
  _not_ cover.

Spend the review on the sections below instead.

## What changed → what to check

| Change                                                        | Check                                                                                     |
| ------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| A new or edited mixin class in `src/quax_blocks/_src/*.py`    | [The dispatch-miss guard](#the-dispatch-miss-guard), [Lax/Numpy parity](#laxnumpy-parity) |
| A `Lax*` mixin specifically                                   | [Intentional semantic divergence](#intentional-semantic-divergence)                       |
| `__eq__`/`__ne__` or a `Comparison` mixin                     | [The `__eq__` shadowing gap](#the-__eq__-shadowing-gap)                                   |
| `__all__` in any `_src/*.py` or `src/quax_blocks/__init__.py` | [Export lists](#export-lists)                                                             |
| A docstring `>>>` example                                     | [Doctests are real tests](#doctests-are-real-tests)                                       |
| `qnp.*` / `qlax.*` calls                                      | [Permissive quaxed typing](#permissive-quaxed-typing)                                     |
| `_compat.py`, or anything gated by a `jax`/`quax` version     | [Version gates](#version-gates)                                                           |
| Anything under `tests/`                                       | [Tests](#tests)                                                                           |

## The dispatch-miss guard

Every mixin method that calls `qlax.*`/`qnp.*` must wrap the call and turn a
dispatch miss into `NotImplemented`:

```python
def __add__(self, other):
    try:
        return qlax.add(self, other)
    except DISPATCH_ERRORS:  # from ._compat — (TypeError, NotFoundLookupError, ...)
        return NotImplemented
```

A method missing this guard raises instead of deferring, which breaks Python's
reflected-operand protocol for the _other_ type too (its `__radd__` never gets a
chance). This is not hypothetical: a shipped comparison mixin was missing
exactly this guard and was the only one of twelve comparison methods that
behaved differently. Check every new `__dunder__` method against its siblings in
the same file — they should all follow the identical
`try/except DISPATCH_ERRORS: return NotImplemented` shape, imported from
`._compat`. A bespoke `except TypeError:` (missing `NotFoundLookupError`, or
inventing a new tuple instead of importing `DISPATCH_ERRORS`) is a defect, not a
style nit — see [Version gates](#version-gates) for why the tuple itself is not
`(TypeError,)`.

Unary mixins with no "other operand" (`__pos__`, `__neg__`, `__abs__`,
`__invert__`) have nothing to decline, so they call `qlax`/`qnp` unguarded —
don't ask for a guard there.

## Lax/Numpy parity

Every operator ships as a `Lax*` mixin (one `jax.lax` primitive) and a `Numpy*`
mixin (`quaxed.numpy`). A PR adding one flavour without the other is almost
always incomplete, unless the primitive genuinely doesn't exist on one side —
the two real precedents are `__getitem__` (no general `jax.lax` indexing
primitive) and `__copy__`/`__deepcopy__` (no `jax.lax.copy`). Anything else
missing a counterpart should be questioned.

Check that a new/edited pair actually agrees on **interface** (same operand
handling, same reflected-method behaviour) even where they intentionally
disagree on **semantics** — see the next section for how those two differ.

## Intentional semantic divergence

Several `Lax*` mixins deliberately return a different value than the Python
operator they implement, because `jax.lax` primitives have no NumPy-level
semantics layered on top: `/` truncates on integers (`lax.div`), `%` takes the
sign of the dividend (`lax.rem`), `**` and `//` reject integer dtypes entirely,
`divmod` truncates instead of floors. These are documented in
`docs/guides/mixins.md` and in each mixin's own docstring/warning admonition.

**Do not "fix" a `Lax*` mixin to match Python/NumPy semantics** — that is
exactly what the `Numpy*` counterpart is for, and changing the `Lax*` one breaks
the stated Lax-vs-NumPy contract for anyone relying on the tight XLA primitive
mapping. The one exception that _is_ a real bug, not intentional semantics:
`LaxMatMulMixin` originally called only `jax.lax.dot`, which rejects batched
(≥3-D) operands outright — that was a missing-rank-dispatch bug (now fixed by
branching on operand rank), not an intentional Lax/NumPy divergence, because
`jax.lax.dot` simply doesn't support the batched case at all rather than
supporting it differently. When a new numeric mixin lands, check which category
a semantic difference falls into before accepting or requesting a change: "no
primitive covers this shape" (bug, needs a fix or an explicit unsupported-case
error) vs. "the primitive's own semantics differ from Python's" (intentional,
needs a docstring warning, not a fix).

## The `__eq__` shadowing gap

`quax.ArrayValue` subclasses `equinox.Module`, which defines its own structural
`__eq__`. A mixin's `__eq__`/`__ne__` does not override it through ordinary
inheritance — every usage site (docstring example, test fixture, `docs/`
example) that wants element-wise comparison must explicitly do
`__eq__ = LaxEqMixin.__eq__` (or the `Numpy`/aggregate equivalent) in the
subclass body. If a PR adds a new usage example, a new test fixture, or a new
`Comparison`-mixing class without that explicit reassignment, flag it — the
example will run but silently exercise structural equality rather than the mixin
it's meant to demonstrate.

## Export lists

Every new public mixin name must appear in **three** places, or it is
unimportable from the top-level package despite existing:

1. The `# fmt: off` / `# fmt: on` `__all__` list at the top of its `_src/*.py`
   module.
2. The `import` block in `src/quax_blocks/__init__.py`.
3. The (larger) `# fmt: off` `__all__` list in `src/quax_blocks/__init__.py`.

The `# fmt: off` blocks are intentionally hand-formatted (grouped by operator,
with alignment comments like `# __add__`) — don't let a formatter or a "cleanup"
reflow them; check a diff to one of these blocks reads as an addition/removal in
place, not a full rewrite.

## Doctests are real tests

Unlike some sibling repos in this ecosystem, **doctests here are collected and
run in CI** — `testpaths = ["tests", "src"]` and `conftest.py` installs sybil to
parse `>>>` blocks from every `.py` file, including `src/`. A mixin docstring's
example is therefore a genuine assertion, not illustrative prose: if a PR
hand-edits an example's shown output without running it, treat that as
unverified and ask for the actual output. This does **not** extend to Markdown —
sybil's patterns are `*.rst`/`*.py` only, so examples in
`skills/quax-blocks/SKILL.md`, `docs/`, or `README.md` are not auto-run; the
skill's `python` blocks are covered separately by
[tests/test_skill_examples.py](../../../tests/test_skill_examples.py).

## Permissive quaxed typing

`qlax`/`qnp` are imported normally at runtime but typed as `Any` under
`TYPE_CHECKING` in every `_src/*.py` module, because `quaxed`'s plain-JAX
annotations can't express quax's dispatch. That means `mypy`/`pyright` will
**not** catch a typo like `qnp.subtractt` — only
[tests/test_quaxed_names.py](../../../tests/test_quaxed_names.py)'s runtime
attribute scan does. A PR adding a new `qnp.*`/`qlax.*` reference needs that
test to have actually run (check CI), not just the type checker passing.

## Version gates

`_compat.py`'s `DISPATCH_ERRORS` tuple conditionally includes `AssertionError`,
gated on `jax < 0.9.2` — below that version, quax's own dispatch-miss path ends
in a bare `assert False` rather than raising `TypeError`. This is easy to get
backwards (gating on the `quax` version instead of the `jax` version, or picking
the wrong boundary release) and hard to catch without testing at the dependency
floor, since a recent local jax install never exercises the guarded branch. If a
PR touches version gating, confirm which library's version actually changed the
behaviour, and confirm the CI matrix includes a floor job that would exercise
it.

## Tests

- [tests/test_operators.py](../../../tests/test_operators.py) is behavioural:
  reflected-operand order, the `NotImplemented` fallback path, and — most
  importantly — pins the documented Lax-vs-Python divergences so they cannot
  drift silently. A new divergence (or a fix to a real bug that happens to
  change output) needs an entry here, not just a docstring update.
- [tests/test_quaxed_names.py](../../../tests/test_quaxed_names.py) and
  [tests/test_skill_examples.py](../../../tests/test_skill_examples.py) are both
  scanners over source/skill text, not behavioural tests — a PR that adds a new
  `_src/*.py` module or new fenced `python` block in the skill should still pass
  them, but "tests pass" here means "nothing referenced a name that doesn't
  exist," not "the new code is correct."

## Repo conventions

- **`uv run` for everything** — `uv run pytest`, `uv run prek run --all-files`.
  Never bare `python` or `pytest`.
- Commits use gitmoji plus conventional commits: `🐛 fix: ...`, `✅ test: ...`,
  `📝 docs: ...`.
- Mypy `--strict` runs via pre-commit scoped to `src/` only (see
  `.pre-commit-config.yaml`); type errors under `tests/` reach CI unreviewed by
  mypy (pyright covers a broader path — check which tool a given issue needs).

## Further reading

- [skills/quax-blocks/SKILL.md](../../../skills/quax-blocks/SKILL.md) — the
  Lax/Numpy contract, the `__eq__` trap, and a troubleshooting table, written
  for consumers of the library rather than reviewers.
- [AGENTS.md](../../../AGENTS.md) — module-by-module architecture and commands.
- [docs/guides/mixins.md](../../../docs/guides/mixins.md) — the full mixin
  reference table.
