# quax-blocks — Agent Instructions

`quax-blocks` provides reusable **mixin building blocks** for authoring
[`quax`](https://github.com/patrick-kidger/quax) `ArrayValue` classes: operator
support (arithmetic, bitwise, comparison, rounding, container, copy) implemented
on top of [`quaxed`](https://github.com/GalacticDynamics/quaxed) so the
operators dispatch through quax.

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
examples in each module):

```python
from jaxtyping import Array
from quax_blocks import AbstractVal, LaxAddMixin


class Val(AbstractVal, LaxAddMixin[object, Array]):
    v: Array
```

Note the recurring pattern for `__eq__` (Equinox's PyTree `__eq__` cannot be
overridden by subclassing, so it must be re-assigned explicitly):
`__eq__ = LaxEqMixin.__eq__`.

## Conventions & pitfalls

- **uv-native**: dependencies use PEP-735 dependency-groups; CI runs
  `uv sync --locked` + `prek`. Regenerate the lock with `uv lock` after editing
  `pyproject.toml`.
- **Doctests are tests**: `testpaths` includes `src`, and Sybil executes the
  `>>> ` examples in docstrings — keep them correct and deterministic.
- **`# fmt: off` blocks**: the big `__all__` lists (and their duplicates across
  modules) are intentionally hand-formatted; don't reflow them.
- **Two flavors stay in sync**: when adding/altering an operator, update both
  the `Lax*` and `Numpy*` mixin.
- **Type checking**: `mypy --strict` runs via pre-commit (isolated,
  `files: src`). Pyright is not yet wired up (tracked separately).
