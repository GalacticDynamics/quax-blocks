"""Unary operations for array-ish objects."""

# fmt: off
__all__ = [
    "LaxUnaryMixin", "NumpyUnaryMixin",
    # ----------
    "LaxPosMixin", "NumpyPosMixin",  # __pos__
    "LaxNegMixin", "NumpyNegMixin",  # __neg__
    "LaxInvertMixin", "NumpyInvertMixin",  # __invert__
    "LaxAbsMixin", "NumpyAbsMixin",  # __abs__
]
# fmt: on

from typing import TYPE_CHECKING, Any, Generic
from typing_extensions import TypeVar

import optype as opt

# `quaxed`'s annotations describe the plain-JAX signatures it wraps (`ArrayLike`
# in, `Array` out) and cannot express quax's runtime dispatch, under which an
# `ArrayValue` flows through and comes back out. Type-checking the mixins
# against those signatures produces hundreds of false positives, so the modules
# are given a permissive type at check time and imported normally at runtime.
# `test_quaxed_names_exist` guards the function names this gives up on.
if TYPE_CHECKING:
    qlax: Any
    qnp: Any
else:
    import quaxed.lax as qlax
    import quaxed.numpy as qnp

T = TypeVar("T")
R = TypeVar("R", default=bool)

# -----------------------------------------------
# `__pos__`


class LaxPosMixin:
    """Mixin for ``__pos__`` method returning ``self`` unchanged.

    `jax.lax` has no ``pos`` primitive; unary plus is the identity, so this
    returns ``self`` directly. See `NumpyPosMixin` for the
    `jax.numpy.positive` version.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, LaxPosMixin

    >>> class Val(AbstractVal, LaxPosMixin):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> +x
    Val(v=i32[3])

    """

    def __pos__(self: opt.CanPosSelf) -> opt.CanPosSelf:
        return self  # TODO: more robust implementation


class NumpyPosMixin:
    """Mixin for ``__pos__`` method using quaxified `jax.numpy.positive`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, NumpyPosMixin

    >>> class Val(AbstractVal, NumpyPosMixin):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> +x
    Val(v=i32[3])

    """

    def __pos__(self) -> opt.CanPosSelf:
        return qnp.positive(self)


# -----------------------------------------------
# `__neg__`


class LaxNegMixin(Generic[R]):
    """Mixin for ``__neg__`` method using quaxified `jax.lax.neg`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, LaxNegMixin

    >>> class Val(AbstractVal, LaxNegMixin[Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> -x
    Array([-1, -2, -3], dtype=int32)

    """

    def __neg__(self) -> R:
        return qlax.neg(self)


class NumpyNegMixin(Generic[R]):
    """Mixin for ``__neg__`` method using quaxified `jax.numpy.negative`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, NumpyNegMixin

    >>> class Val(AbstractVal, NumpyNegMixin[Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> -x
    Array([-1, -2, -3], dtype=int32)

    """

    def __neg__(self) -> R:
        return qnp.negative(self)


# -----------------------------------------------
# `__invert__`


class LaxInvertMixin(Generic[R]):
    """Mixin for ``__invert__`` method using quaxified `jax.lax.bitwise_not`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, LaxInvertMixin

    >>> class Val(AbstractVal, LaxInvertMixin[Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> ~x
    Array([-2, -3, -4], dtype=int32)

    """

    def __invert__(self) -> R:
        return qlax.bitwise_not(self)


class NumpyInvertMixin(Generic[R]):
    """Mixin for ``__invert__`` method using quaxified `jax.numpy.invert`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, NumpyInvertMixin

    >>> class Val(AbstractVal, NumpyInvertMixin[Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> ~x
    Array([-2, -3, -4], dtype=int32)

    """

    def __invert__(self) -> R:
        return qnp.invert(self)


# -----------------------------------------------
# `__abs__`


class LaxAbsMixin(Generic[R]):
    """Mixin for ``__abs__`` method using quaxified `jax.lax.abs`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, LaxAbsMixin

    >>> class Val(AbstractVal, LaxAbsMixin[Array]):
    ...     v: Array

    >>> x = Val(jnp.array([-1, -2, -3]))
    >>> abs(x)
    Array([1, 2, 3], dtype=int32)

    """

    def __abs__(self) -> R:
        return qlax.abs(self)


class NumpyAbsMixin(Generic[R]):
    """Mixin for ``__abs__`` method using quaxified `jax.numpy.abs`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, NumpyAbsMixin

    >>> class Val(AbstractVal, NumpyAbsMixin[Array]):
    ...     v: Array

    >>> x = Val(jnp.array([-1, -2, -3]))
    >>> abs(x)
    Array([1, 2, 3], dtype=int32)

    """

    def __abs__(self) -> R:
        return qnp.abs(self)


# ===============================================
# Combined Mixins


class LaxUnaryMixin(LaxPosMixin, LaxNegMixin[R], LaxAbsMixin[R], LaxInvertMixin[R]):
    """Combined mixin for unary operations using quaxified `jax.lax`."""


class NumpyUnaryMixin(
    NumpyPosMixin, NumpyNegMixin[R], NumpyAbsMixin[R], NumpyInvertMixin[R]
):
    """Combined mixin for unary operations using quaxified `jax.numpy`."""
