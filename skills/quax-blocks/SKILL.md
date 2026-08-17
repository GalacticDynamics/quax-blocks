---
name: quax-blocks
description:
  Use when writing, reviewing, or debugging code that composes quax-blocks
  mixins (Lax*/Numpy* Add/Sub/Mul/.../ComparisonMixin, UnaryMixin, RoundMixin,
  ...) onto a quax.ArrayValue subclass. Also use when a mixed-in operator
  returns NotImplemented / raises TypeError unexpectedly, when `==` on a
  quax-blocks type does structural PyTree equality instead of element-wise
  comparison, when Lax-flavoured `/`, `%`, `**`, or `//` gives a surprising
  result, or when deciding which mixin (Lax vs Numpy, single vs Both) to
  inherit.
---

# Using quax-blocks Effectively

`quax-blocks` is a library of mixin classes that implement the ~40 dunder
methods (`__add__`, `__radd__`, comparisons, bitwise, rounding, container, copy)
a `quax.ArrayValue` needs to behave like a Python number/array. Each mixin is a
couple of lines: call the quaxed equivalent, catch a dispatch miss, return
`NotImplemented`. The library exists so you never hand-write that couple of
lines forty times.

**Read the quax and quaxed skills first if this is your first quax type.** This
skill assumes you already have a working `quax.ArrayValue` (with
`aval()`/`materialise()`) and covers only the mixin layer on top of it:

- [`quax` agent skill](https://github.com/nstarman/quax/blob/main/skills/quax/SKILL.md)
  (upstream, not in this repo) — `Value`/`ArrayValue`, `aval()`/`materialise()`,
  `quax.register`, the quaxify+`jax.jit` performance rule.
- [`quaxed` agent skill](https://github.com/GalacticDynamics/quaxed/blob/main/skills/quaxed/SKILL.md)
  (upstream, not in this repo) — the `quaxed.numpy`/`quaxed.lax` fallback
  behaviour that every mixin call goes through.

Checked against quax-blocks main (`jax>=0.7.2`, `quax>=0.3.7`, `quaxed>=0.10.5`,
`plum-dispatch>=2.8.0`, Python >=3.12). Docs:
[Mixin reference](https://github.com/GalacticDynamics/quax-blocks/blob/main/docs/guides/mixins.md).

## Quick start

Mix the aggregate classes onto an `ArrayValue` and the operators just work:

```python
import jax
import jax.numpy as jnp
import quax
from jaxtyping import Array

import quax_blocks as qb


class MyArray(
    qb.NumpyBinaryOpsMixin,  # +, -, *, /, //, %, **, @  (and reflected)
    qb.NumpyComparisonMixin,  # ==, !=, <, <=, >, >=
    qb.NumpyUnaryMixin,  # +x, -x, abs(x)
    quax.ArrayValue,
):
    data: Array
    __eq__ = qb.NumpyComparisonMixin.__eq__  # see "The __eq__ trap" below

    def aval(self) -> jax.core.ShapedArray:
        return jax.core.ShapedArray(self.data.shape, self.data.dtype)

    def materialise(self) -> Array:
        return self.data


x = MyArray(jnp.array([1.0, 2.0, 3.0]))
y = MyArray(jnp.array([4.0, 5.0, 6.0]))
x + y
# Array([5., 7., 9.], dtype=float32)
```

Or pick individual operators — `qb.NumpyAddMixin` alone gives you `__add__`
only. Every mixin's own docstring carries a runnable example; read the one for
the mixin you're using rather than guessing at behaviour.

## Two flavours, one interface, different semantics

Every operator ships as **`Lax*`** (backed by `quaxed.lax`, one-to-one with an
XLA primitive) and **`Numpy*`** (backed by `quaxed.numpy`, NumPy semantics).
They are not interchangeable defaults — several `Lax*` mixins **silently return
a different value** than the operator they implement, because they map straight
onto a `jax.lax` primitive with no Python-level semantics on top:

| Operator | `Lax*` behaviour                                                                                               | Python / `Numpy*` behaviour     |
| -------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| `/`      | `jax.lax.div` — **integer division** on int operands: `[4,5,6] / 2` → `[2, 2, 3]`                              | true division → `[2., 2.5, 3.]` |
| `%`      | `jax.lax.rem` — C-style remainder, sign of the **dividend**: `-7 % 3` → `-1`                                   | sign of the **divisor** → `2`   |
| `**`     | `jax.lax.pow` — **floating-point operands only**; int raises `"pow does not accept dtype int32"`               | integers supported              |
| `//`     | `lax.floor(lax.div(...))` — **floating-point operands only**; int raises `"floor does not accept dtype int32"` | integers supported              |
| `divmod` | truncates toward zero (C-style): `divmod(-7, 3)` → `(-2, -1)`                                                  | floors: `(-3, 2)`               |

None of these raise to announce themselves — `x / 2` on a `LaxTrueDivMixin` type
just quietly gives you the wrong number if you expected Python division. Default
to the `Numpy*` mixin unless you specifically want the lax primitive (tightest
XLA coupling, no Python overhead). This divergence is intentional and
documented, not a bug — do not "fix" a `Lax*` mixin to match Python semantics;
add a test locking the current behaviour instead if you touch one.

## The `__eq__` trap

Equinox's `Module` (which `quax.ArrayValue` subclasses) defines its own `__eq__`
for structural PyTree comparison. Python does not let a mixin's `__eq__` win
over a base class's by ordinary MRO in this case — you must **explicitly
reassign it** in the subclass body:

```python
from jaxtyping import Array, Bool
from quax_blocks import AbstractVal, LaxEqMixin


class Val(AbstractVal, LaxEqMixin[object, Bool[Array, "..."]]):
    v: Array
    __eq__ = LaxEqMixin.__eq__  # required, or you get Equinox's structural __eq__
```

Forgetting this line is not a type error and not caught by inheriting
`LaxComparisonMixin`/`NumpyComparisonMixin` either — the aggregate mixin has
exactly the same problem, so reassign from whichever one you used:
`__eq__ = NumpyComparisonMixin.__eq__`. If `x == x` returns a bare `bool`
instead of an elementwise array, this is almost always why.

## The `Both*` pattern and type parameters

For every binary operator `__xxx__` there is a reflected `__rxxx__`. Three
granularities exist: `LaxAddMixin` (`__add__` only), `LaxRAddMixin` (`__radd__`
only), `LaxBothAddMixin` (both). Aggregates compose upward:
`LaxMathMixin`/`NumpyMathMixin` (all arithmetic), `LaxBitwiseMixin`/
`NumpyBitwiseMixin` (shifts + and/xor/or), `LaxBinaryOpsMixin`/
`NumpyBinaryOpsMixin` (both of those together).

Mixins with a second operand are `Generic[T, R]`: `T` is the type of the _other_
operand (default `object`), `R` is the return type (default `bool` for
comparisons, otherwise unconstrained). These are static-checking only —
`NumpyAddMixin` and `NumpyAddMixin[object, Array]` dispatch identically at
runtime:

```python
from jaxtyping import Array
from quax_blocks import AbstractVal, NumpyAddMixin


class Val(AbstractVal, NumpyAddMixin[object, Array]):
    v: Array


Val(jnp.array([1, 2, 3])) + Val(jnp.array([1, 2, 3]))
# Array([2, 4, 6], dtype=int32)
```

## Dispatch misses return `NotImplemented`, not an exception

Every mixin method wraps its quaxed call and turns a dispatch miss into
`NotImplemented`, so Python falls through to the other operand's reflected
method instead of raising:

```python
def __add__(self, other):
    try:
        return qlax.add(self, other)
    except (TypeError, NotFoundLookupError):  # + AssertionError on jax < 0.9.2
        return NotImplemented
```

This is why `val + "not an array"` cleanly raises Python's own
`TypeError: unsupported operand type(s)` instead of leaking a `quax`/`plum`
traceback: the mixin declines, Python tries `"not an array".__radd__(val)`, that
also declines, and _then_ Python raises its own error. If you write a new mixin
method by hand instead of composing existing ones, **it must have this guard**,
or a genuinely unsupported operand raises the wrong exception type and breaks
reflected dispatch for the _other_ operand's type too — this was a real, shipped
bug (a single comparison mixin was missing it).

The exception tuple is version-gated, not a fixed constant — on `jax` 0.7.2
through 0.9.1, quax's own dispatch-miss path ends in a bare `assert False`
rather than raising `TypeError`, so the guard must also catch `AssertionError`
there. This is invisible unless you test at the dependency floor; it does not
reproduce on a recent jax install.

## Class-variable knobs

Two mixins expose a class variable instead of a constructor argument, because
the choice is a property of the _type_, not a call:

- `LaxRoundMixin._ROUNDING_METHOD` (default `RoundingMethod.AWAY_FROM_ZERO`) —
  `NumpyRoundMixin` always uses banker's rounding (ties to even, matching
  `numpy.round`); override this on the Lax mixin to change its tie-breaking.
- `LaxRShiftMixin`/`LaxRRShiftMixin._RIGHT_SHIFT_LOGICAL` (default `True`) —
  selects `jax.lax.shift_right_logical` vs `..._arithmetic`, since `jax.lax` has
  no single "right shift" primitive.

## `matmul` dispatches on operand rank

`LaxMatMulMixin` cannot just call one `jax.lax` primitive: `jax.lax.dot` handles
1-D/2-D operands but rejects batched (≥3-D) ones, and `jax.lax.batch_matmul` is
the reverse (rejects rank < 2). The mixin picks between them based on
`max(ndim(lhs), ndim(rhs))` at trace time (shapes are static, so this costs
nothing at runtime). What it still cannot do — because no `jax.lax` primitive
does — is NumPy's **mixed-rank broadcasting** (`(2,2,2) @ (2,2)`); that raises a
`ValueError` from lax. Use `NumpyMatMulMixin` if you need full `numpy.matmul`
semantics.

## Copy mixins return a bare array, not a new instance

`NumpyCopyMixin.__copy__` and `NumpyDeepCopyMixin.__deepcopy__` both call
`quaxed.numpy.copy`, which returns whatever `quax` dispatch produces for the
type — typically a plain array, not a reconstructed instance of your class. This
is documented, current behaviour, not a bug to fix reflexively. `__deepcopy__`
does consult and populate the `memo` dict it's handed, so repeated references to
the same object inside one `copy.deepcopy()` call share a single copy — a mixin
that ignores `memo` silently breaks that sharing invariant and is a real
regression if you touch this file.

`LaxLenMixin`/`NumpyLenMixin.__len__` and the `__length_hint__` mixins return
`0` for a 0-d (scalar) value rather than raising `TypeError`, which is where
NumPy itself diverges from Python's `len()` protocol expectations — this is
intentional, matching `self.shape[0] if self.shape else 0`, not an oversight.

## Troubleshooting

| Symptom                                                                              | Cause / fix                                                                                                                                                                                           |
| ------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `x == x` gives a plain `bool`, not an elementwise array                              | `__eq__` not reassigned in the subclass body — Equinox's `Module.__eq__` wins otherwise. See [The `__eq__` trap](#the-__eq__-trap).                                                                   |
| `x / 2` truncates when you expected a float                                          | You're on a `Lax*` mixin (`jax.lax.div`); switch to the `Numpy*` mixin, or the divergence is intentional and this is expected.                                                                        |
| `x ** 2` / `x // 2` raises `"... does not accept dtype int32"`                       | `LaxPowMixin`/`LaxFloorDivMixin` require floating-point operands; cast, or use `NumpyPowMixin`/`NumpyFloorDivMixin`.                                                                                  |
| A binary op raises instead of returning `NotImplemented`                             | A hand-written operator method is missing the `except (TypeError, NotFoundLookupError, ...)` guard — see [Dispatch misses](#dispatch-misses-return-notimplemented-not-an-exception).                  |
| `b @ b` raises `"dimension_numbers must be specified..."` on a batched (3-D) value   | Using a version of `LaxMatMulMixin` before rank dispatch, or a mixed-rank `(3-D) @ (2-D)` call that no `jax.lax` primitive supports — use `NumpyMatMulMixin`.                                         |
| `copy.deepcopy` of a cyclic/shared structure duplicates a node that should be shared | A custom `__deepcopy__` is not consulting/populating `memo`; see [Copy mixins](#copy-mixins-return-a-bare-array-not-a-new-instance).                                                                  |
| `mypy`/`pyright` doesn't catch a typo like `qnp.subtractt`                           | The mixin modules type `qnp`/`qlax` as `Any` at check time (quaxed's plain-JAX annotations can't model quax dispatch) — only `tests/test_quaxed_names.py`'s runtime scan catches this, run the tests. |
| Everything works but you're re-deriving mixins by hand                               | Check the [Mixin reference](https://github.com/GalacticDynamics/quax-blocks/blob/main/docs/guides/mixins.md) — the operator you want probably already exists.                                         |

## Version notes

`quax-blocks` pins to `quax>=0.3.7` specifically because earlier quax patch
releases don't import cleanly across the full supported `jax` range
(`jax>=0.7.2`); if you see an import-time failure on an old lockfile, bump
`quax` first. The `AssertionError` addition to the dispatch-miss guard is scoped
to `jax < 0.9.2` and is a compatibility shim expected to be deleted once the
`jax` floor rises past that version — don't rely on it being permanent if you're
reading the source rather than this skill.
