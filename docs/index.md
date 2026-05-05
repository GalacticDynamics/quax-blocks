# quax-blocks

Building blocks for [Quax](https://github.com/nstarman/quax) classes.

[`quax`](https://docs.kidger.site/quax/) enables JAX + multiple dispatch +
custom array-ish objects. `quax-blocks` provides ready-made mixins for the most
common operator patterns, so you don't have to hand-write every JAX primitive
dispatch.

## Installation

```bash
pip install quax-blocks
```

## Quick start

Define a minimal `quax.ArrayValue` subclass, then mix in the operators you need:

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import quax
from jaxtyping import Array

import quax_blocks as qb


class MyArray(
    qb.NumpyBinaryOpsMixin,  # +, -, *, /, //, %, **, @, bitwise …
    qb.NumpyComparisonMixin,  # ==, !=, <, <=, >, >=
    qb.NumpyUnaryMixin,  # +x, -x, abs(x)
    qb.NumpyRoundMixin,  # round(x)
    qb.NumpyLenMixin,  # len(x)
    quax.ArrayValue,
):
    data: Array

    def aval(self) -> jax.core.ShapedArray:
        return jax.core.ShapedArray(self.data.shape, self.data.dtype)

    def materialise(self) -> Array:
        return self.data


x = MyArray(jnp.array([1.0, 2.0, 3.0]))
y = MyArray(jnp.array([10.0, 20.0, 30.0]))

print(x + y)  # Array([11., 22., 33.], dtype=float32)
print(x < y)  # Array([ True,  True,  True], dtype=bool)
print(-x)  # Array([-1., -2., -3.], dtype=float32)
print(len(x))  # 3

# Fully compatible with JAX transformations
add = quax.quaxify(lambda a, b: a + b)
print(jax.jit(add)(x, y))  # Array([11., 22., 33.], dtype=float32)
print(jax.vmap(add)(x, y))  # Array([11., 22., 33.], dtype=float32)
```

## Mixin groups

`quax-blocks` organises its mixins into five groups:

| Group                                               | What it provides                                         |
| --------------------------------------------------- | -------------------------------------------------------- |
| [Comparison](guides/mixins.md#comparison-operators) | `==`, `!=`, `<`, `<=`, `>`, `>=`                         |
| [Binary](guides/mixins.md#binary-operators)         | `+`, `-`, `*`, `/`, `//`, `%`, `**`, `@`, bitwise ops    |
| [Unary](guides/mixins.md#unary-operators)           | `+x`, `-x`, `~x`, `abs(x)`                               |
| [Rounding](guides/mixins.md#rounding-operators)     | `round()`, `math.trunc()`, `math.floor()`, `math.ceil()` |
| [Container](guides/mixins.md#container-operators)   | `len()`, `operator.length_hint()`                        |
| [Copy](guides/mixins.md#copy-operators)             | `copy.copy()`, `copy.deepcopy()`                         |

Each group comes in two flavours — **Lax** (using `jax.lax` primitives directly)
and **NumPy** (using `jax.numpy`). See the
[Lax vs NumPy guide](guides/mixins.md#lax-vs-numpy) for when to prefer each.

## See also

- [Getting started guide](guides/getting_started.md) — step-by-step walkthrough
- [Mixin reference](guides/mixins.md) — full table of every mixin
- [Quax](https://docs.kidger.site/quax/) — the base library
- [Quaxed](https://github.com/GalacticDynamics/quaxed) — pre-quaxified JAX
  functions
