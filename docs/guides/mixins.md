# Mixin reference

Every mixin in `quax-blocks` is documented here, grouped by operator category.

## Lax vs NumPy

Each operator is available in two flavours:

| Flavour   | Prefix   | Underlying function                                        |
| --------- | -------- | ---------------------------------------------------------- |
| **Lax**   | `Lax…`   | `jax.lax.*` primitive — lower-level, always XLA-compilable |
| **NumPy** | `Numpy…` | `jax.numpy.*` — higher-level, matches NumPy semantics      |

**When to use Lax:** When you are implementing a type that wraps a raw
`jax.Array` and you want the tightest possible coupling to the XLA compiler. Lax
operations map 1-to-1 to XLA HLO instructions and have no Python overhead.

**When to use NumPy:** When you want NumPy broadcasting rules, NumPy-compatible
type promotion, or when a Lax equivalent does not exist (e.g. `divmod`,
`__invert__`).

Both flavours work transparently with `jit`, `vmap`, and `grad`.

### Lax operators do not follow Python's operator semantics

Because the `Lax…` mixins map each operator straight onto a `jax.lax` primitive,
some of them **behave differently from the Python operator they implement**. The
`Numpy…` mixins follow Python/NumPy semantics in every case. Prefer them unless
you specifically want the lax primitive.

| Operator | Lax mixin behaviour                                                                        | Python / NumPy behaviour        |
| -------- | ------------------------------------------------------------------------------------------ | ------------------------------- |
| `/`      | `jax.lax.div` — **integer division** on integer operands: `Val([4,5,6]) / 2` → `[2, 2, 3]` | true division → `[2., 2.5, 3.]` |
| `%`      | `jax.lax.rem` — C-style remainder, sign of the **dividend**: `-7 % 3` → `-1`               | sign of the **divisor** → `2`   |
| `**`     | `jax.lax.pow` — **floating-point operands only**; integer operands raise                   | integers supported              |
| `//`     | `lax.floor(lax.div(…))` — **floating-point operands only**; integer operands raise         | integers supported              |

!!! warning

    The `/` and `%` cases return a *different value* rather than raising, so
    they will not announce themselves. If you need Python's semantics, use
    `NumpyTrueDivMixin` / `NumpyModMixin`.

## The `Both*` pattern

For every binary operator `__xxx__` there is a reflected variant `__rxxx__`
(e.g. `__add__` / `__radd__`). `quax-blocks` provides three levels of
granularity:

| Class             | What it includes             |
| ----------------- | ---------------------------- |
| `LaxAddMixin`     | `__add__` only               |
| `LaxRAddMixin`    | `__radd__` only              |
| `LaxBothAddMixin` | `__add__` **and** `__radd__` |

The aggregate `LaxBinaryOpsMixin` / `NumpyBinaryOpsMixin` includes all binary
operators (arithmetic and bitwise).

## Type parameters

Mixins that involve a second operand are generic over:

- `T` — the type of the _other_ operand (default: `object`)
- `R` — the return type (default: `bool` for comparisons, `object` for
  arithmetic)

```python
from jaxtyping import Array
from quax_blocks import AbstractVal, NumpyAddMixin


class Val(AbstractVal, NumpyAddMixin[object, Array]):
    v: Array
```

The parameters are for static type-checking only — they do not affect runtime
dispatch.

---

## Comparison operators

| Operator        | Lax mixin            | NumPy mixin            |
| --------------- | -------------------- | ---------------------- |
| All comparisons | `LaxComparisonMixin` | `NumpyComparisonMixin` |
| `__eq__`        | `LaxEqMixin`         | `NumpyEqMixin`         |
| `__ne__`        | `LaxNeMixin`         | `NumpyNeMixin`         |
| `__lt__`        | `LaxLtMixin`         | `NumpyLtMixin`         |
| `__le__`        | `LaxLeMixin`         | `NumpyLeMixin`         |
| `__gt__`        | `LaxGtMixin`         | `NumpyGtMixin`         |
| `__ge__`        | `LaxGeMixin`         | `NumpyGeMixin`         |

!!! warning "Overriding `__eq__` with Equinox" Equinox's `Module` base class
defines `__eq__` for structural comparison. To replace it with the element-wise
JAX version you must explicitly assign the method in your class body:

    ```python
    class Val(AbstractVal, LaxEqMixin):
        v: Array
        __eq__ = LaxEqMixin.__eq__
    ```

---

## Binary operators

### Aggregate mixins

| Mixin                                       | Includes                                                                               |
| ------------------------------------------- | -------------------------------------------------------------------------------------- |
| `LaxBinaryOpsMixin` / `NumpyBinaryOpsMixin` | All arithmetic **and** bitwise operations below                                        |
| `LaxMathMixin` / `NumpyMathMixin`           | All arithmetic operations (add, sub, mul, matmul, truediv, floordiv, mod, divmod, pow) |
| `LaxBitwiseMixin` / `NumpyBitwiseMixin`     | All bitwise operations (lshift, rshift, and, xor, or)                                  |

### Arithmetic operators

| Operator                         | Lax mixin              | NumPy mixin              |
| -------------------------------- | ---------------------- | ------------------------ |
| `__add__`                        | `LaxAddMixin`          | `NumpyAddMixin`          |
| `__radd__`                       | `LaxRAddMixin`         | `NumpyRAddMixin`         |
| `__add__` + `__radd__`           | `LaxBothAddMixin`      | `NumpyBothAddMixin`      |
| `__sub__`                        | `LaxSubMixin`          | `NumpySubMixin`          |
| `__rsub__`                       | `LaxRSubMixin`         | `NumpyRSubMixin`         |
| `__sub__` + `__rsub__`           | `LaxBothSubMixin`      | `NumpyBothSubMixin`      |
| `__mul__`                        | `LaxMulMixin`          | `NumpyMulMixin`          |
| `__rmul__`                       | `LaxRMulMixin`         | `NumpyRMulMixin`         |
| `__mul__` + `__rmul__`           | `LaxBothMulMixin`      | `NumpyBothMulMixin`      |
| `__matmul__`                     | `LaxMatMulMixin`       | `NumpyMatMulMixin`       |
| `__rmatmul__`                    | `LaxRMatMulMixin`      | `NumpyRMatMulMixin`      |
| `__matmul__` + `__rmatmul__`     | `LaxBothMatMulMixin`   | `NumpyBothMatMulMixin`   |
| `__truediv__`                    | `LaxTrueDivMixin`      | `NumpyTrueDivMixin`      |
| `__rtruediv__`                   | `LaxRTrueDivMixin`     | `NumpyRTrueDivMixin`     |
| `__truediv__` + `__rtruediv__`   | `LaxBothTrueDivMixin`  | `NumpyBothTrueDivMixin`  |
| `__floordiv__`                   | `LaxFloorDivMixin`     | `NumpyFloorDivMixin`     |
| `__rfloordiv__`                  | `LaxRFloorDivMixin`    | `NumpyRFloorDivMixin`    |
| `__floordiv__` + `__rfloordiv__` | `LaxBothFloorDivMixin` | `NumpyBothFloorDivMixin` |
| `__mod__`                        | `LaxModMixin`          | `NumpyModMixin`          |
| `__rmod__`                       | `LaxRModMixin`         | `NumpyRModMixin`         |
| `__mod__` + `__rmod__`           | `LaxBothModMixin`      | `NumpyBothModMixin`      |
| `__divmod__`                     | _(not implemented)_    | `NumpyDivModMixin`       |
| `__rdivmod__`                    | _(not implemented)_    | `NumpyRDivModMixin`      |
| `__divmod__` + `__rdivmod__`     | _(not implemented)_    | `NumpyBothDivModMixin`   |
| `__pow__`                        | `LaxPowMixin`          | `NumpyPowMixin`          |
| `__rpow__`                       | `LaxRPowMixin`         | `NumpyRPowMixin`         |
| `__pow__` + `__rpow__`           | `LaxBothPowMixin`      | `NumpyBothPowMixin`      |

### Bitwise operators

| Operator                     | Lax mixin            | NumPy mixin            |
| ---------------------------- | -------------------- | ---------------------- |
| `__lshift__`                 | `LaxLShiftMixin`     | `NumpyLShiftMixin`     |
| `__rlshift__`                | `LaxRLShiftMixin`    | `NumpyRLShiftMixin`    |
| `__lshift__` + `__rlshift__` | `LaxBothLShiftMixin` | `NumpyBothLShiftMixin` |
| `__rshift__`                 | `LaxRShiftMixin`     | `NumpyRShiftMixin`     |
| `__rrshift__`                | `LaxRRShiftMixin`    | `NumpyRRShiftMixin`    |
| `__rshift__` + `__rrshift__` | `LaxBothRShiftMixin` | `NumpyBothRShiftMixin` |
| `__and__`                    | `LaxAndMixin`        | `NumpyAndMixin`        |
| `__rand__`                   | `LaxRAndMixin`       | `NumpyRAndMixin`       |
| `__and__` + `__rand__`       | `LaxBothAndMixin`    | `NumpyBothAndMixin`    |
| `__xor__`                    | `LaxXorMixin`        | `NumpyXorMixin`        |
| `__rxor__`                   | `LaxRXorMixin`       | `NumpyRXorMixin`       |
| `__xor__` + `__rxor__`       | `LaxBothXorMixin`    | `NumpyBothXorMixin`    |
| `__or__`                     | `LaxOrMixin`         | `NumpyOrMixin`         |
| `__ror__`                    | `LaxROrMixin`        | `NumpyROrMixin`        |
| `__or__` + `__ror__`         | `LaxBothOrMixin`     | `NumpyBothOrMixin`     |

---

## Unary operators

| Operator     | Lax mixin           | NumPy mixin        |
| ------------ | ------------------- | ------------------ |
| All unary    | `LaxUnaryMixin`     | `NumpyUnaryMixin`  |
| `__pos__`    | `LaxPosMixin`       | `NumpyPosMixin`    |
| `__neg__`    | `LaxNegMixin`       | `NumpyNegMixin`    |
| `__abs__`    | `LaxAbsMixin`       | `NumpyAbsMixin`    |
| `__invert__` | _(not implemented)_ | `NumpyInvertMixin` |

---

## Rounding operators

These implement the Python numeric protocol methods called by `round()`,
`math.trunc()`, `math.floor()`, and `math.ceil()`.

| Operator    | Lax mixin       | NumPy mixin       |
| ----------- | --------------- | ----------------- |
| `__round__` | `LaxRoundMixin` | `NumpyRoundMixin` |
| `__trunc__` | `LaxTruncMixin` | `NumpyTruncMixin` |
| `__floor__` | `LaxFloorMixin` | `NumpyFloorMixin` |
| `__ceil__`  | `LaxCeilMixin`  | `NumpyCeilMixin`  |

!!! note "Rounding mode" `LaxRoundMixin` defaults to
`RoundingMethod.AWAY_FROM_ZERO` (rounds ties away from zero, e.g. `0.5 → 1`,
`-0.5 → -1`). `NumpyRoundMixin` uses banker's rounding (rounds ties to even,
e.g. `0.5 → 0`, `2.5 → 2`), which matches `numpy.round`. Override the
`_ROUNDING_METHOD` class variable on `LaxRoundMixin` to change the behaviour.

---

## Container operators

| Operator          | Lax mixin            | NumPy mixin            |
| ----------------- | -------------------- | ---------------------- |
| `__len__`         | `LaxLenMixin`        | `NumpyLenMixin`        |
| `__length_hint__` | `LaxLengthHintMixin` | `NumpyLengthHintMixin` |

`LaxLenMixin` reads the length from `self.shape[0]` (returns `0` for scalars).
`NumpyLenMixin` uses `jax.numpy.shape` to obtain the shape first.

---

## Copy operators

| Operator       | Lax mixin           | NumPy mixin          |
| -------------- | ------------------- | -------------------- |
| `__copy__`     | _(not implemented)_ | `NumpyCopyMixin`     |
| `__deepcopy__` | _(not implemented)_ | `NumpyDeepCopyMixin` |

Both copy operations are backed by `jax.numpy.copy` and return a new array, not
a new instance of the custom type.
