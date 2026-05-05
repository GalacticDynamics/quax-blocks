# Getting started

This guide walks through building a custom `quax.ArrayValue` type step by step,
showing how `quax-blocks` mixins remove the boilerplate.

## Prerequisites

You should be familiar with [Quax](https://docs.kidger.site/quax/) — in
particular the concepts of `ArrayValue`, `aval()`, and `materialise()`.

## 1. A minimal `ArrayValue` without mixins

With plain Quax, every Python operator you want to support requires a
hand-written primitive dispatch:

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import quax
from jax import lax
from jaxtyping import Array


class MyArray(quax.ArrayValue):
    data: Array

    def aval(self) -> jax.core.ShapedArray:
        return jax.core.ShapedArray(self.data.shape, self.data.dtype)

    def materialise(self) -> Array:
        return self.data


@quax.register(lax.add_p)
def _(x: MyArray, y: MyArray):
    return MyArray(x.data + y.data)


@quax.register(lax.neg_p)
def _(x: MyArray):
    return MyArray(-x.data)


# … one dispatch per operation …
```

This becomes tedious quickly — especially when you need the reflected variants
(`__radd__`, `__rsub__`, …) and both the `jax.lax` and `jax.numpy` code paths.

## 2. The same type with `quax-blocks`

`quax-blocks` ships these dispatches as ready-made mixin classes. Inherit from
the mixins you need:

```python
import equinox as eqx
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

    def aval(self) -> jax.core.ShapedArray:
        return jax.core.ShapedArray(self.data.shape, self.data.dtype)

    def materialise(self) -> Array:
        return self.data


x = MyArray(jnp.array([1.0, 2.0, 3.0]))
y = MyArray(jnp.array([4.0, 5.0, 6.0]))

print(x + y)  # Array([5., 7., 9.], dtype=float32)
print(x - y)  # Array([-3., -3., -3.], dtype=float32)
print(x < y)  # Array([ True,  True,  True], dtype=bool)
print(-x)  # Array([-1., -2., -3.], dtype=float32)
```

## 3. Selecting individual operators

The aggregate mixins (`NumpyBinaryOpsMixin`, `LaxUnaryMixin`, …) are convenient,
but you can also pick exactly the operators you want:

```python
import quax
from jaxtyping import Array

import quax_blocks as qb


class AddableArray(
    qb.NumpyBothAddMixin,  # __add__ + __radd__
    qb.NumpyNegMixin,  # __neg__
    quax.ArrayValue,
):
    data: Array
    ...
```

`BothXxxMixin` is a shorthand that includes both the forward (`__xxx__`) and
reflected (`__rxxx__`) variants. You can also include only one:

```python
class ForwardOnlyAdd(qb.NumpyAddMixin, quax.ArrayValue):
    ...  # supports `obj + other` but not `other + obj`
```

## 4. Type parameters

Many mixins are generic over the _other_ operand type `T` and the _return_ type
`R`:

```python
from jaxtyping import Array
from quax_blocks import AbstractVal, NumpyAddMixin


class Val(AbstractVal, NumpyAddMixin[object, Array]):
    v: Array
```

These parameters are **optional** — leaving them off gives the defaults
`T = object` and `R = bool` (or the appropriate default for each mixin). The
parameters are only used for static type-checking and do not affect runtime
behaviour.

## 5. The `__eq__` special case

Equinox's `Module` base class provides its own `__eq__` method that tests
structural equality. If you want element-wise JAX comparison instead, you must
explicitly assign the method from the mixin:

```python
from jaxtyping import Array, Bool
from quax_blocks import AbstractVal, LaxEqMixin


class Val(AbstractVal, LaxEqMixin[object, Bool[Array, "..."]]):
    v: Array
    __eq__ = LaxEqMixin.__eq__  # required to override Equinox's __eq__


x = Val(jnp.array([1, 2, 3]))
print(x == 2)  # Array([False,  True, False], dtype=bool)
```

## 6. JAX compatibility

Mixins work transparently with JAX transformations:

```python
import jax
import jax.numpy as jnp
import quax

add_fn = quax.quaxify(lambda a, b: a + b)

x = MyArray(jnp.array([1.0, 2.0, 3.0]))
y = MyArray(jnp.array([4.0, 5.0, 6.0]))

# JIT compilation
print(jax.jit(add_fn)(x, y))

# Vectorisation
print(jax.vmap(add_fn)(x, y))
```

## Next steps

- [Mixin reference](mixins.md) — every available mixin with its operator and
  backend
- [API reference](../api/quax_blocks.md) — full docstring reference
