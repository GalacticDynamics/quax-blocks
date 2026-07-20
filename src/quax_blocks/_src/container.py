"""Container operations for Array-ish objects."""

# fmt: off
__all__ = [
    "LaxLenMixin", "NumpyLenMixin",  # __len__
    "LaxLengthHintMixin", "NumpyLengthHintMixin",  # __length_hint__
    "NumpyGetItemMixin",  # __getitem__
]
# fmt: on

import operator
from typing import TYPE_CHECKING, Any, Generic, Protocol, runtime_checkable
from typing_extensions import TypeVar

import quax

# Permissive type at check time (quaxed's annotations can't model quax's
# dispatch); imported normally at runtime. See tests/test_quaxed_names.py.
if TYPE_CHECKING:
    qnp: Any
else:
    import quaxed.numpy as qnp

R = TypeVar("R", default=Any)

#: Quaxified indexing, built once and reused (see ``NumpyGetItemMixin``).
_quax_getitem = quax.quaxify(operator.getitem)


@runtime_checkable
class HasShape(Protocol):
    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the object."""
        # Protocol member needs a body; `...` is canonical (pylint's
        # unnecessary-ellipsis is a false positive for protocols).
        ...  # pylint: disable=unnecessary-ellipsis


# -----------------------------------------------
# `__len__`


class LaxLenMixin:
    """Mixin for ``__len__`` reading the leading axis from ``self.shape``.

    Neither `jax.lax` nor `jax.numpy` has a ``len`` function; the length is
    taken from the shape directly. Returns ``0`` for scalars (unlike NumPy,
    which raises ``TypeError`` for 0-d arrays).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, LaxLenMixin

    >>> class Val(AbstractVal, LaxLenMixin):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> len(x)
    3

    >>> x = Val(jnp.array(1))
    >>> len(x)
    0

    """

    def __len__(self: HasShape) -> int:
        return self.shape[0] if self.shape else 0


class NumpyLenMixin:
    """Mixin for ``__len__`` using quaxified `jax.numpy.shape`.

    `jax.numpy` has no ``len`` function; the shape is obtained first and the
    leading axis returned. Returns ``0`` for scalars (unlike NumPy, which
    raises ``TypeError`` for 0-d arrays).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, NumpyLenMixin

    >>> class Val(AbstractVal, NumpyLenMixin):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> len(x)
    3

    >>> x = Val(jnp.array(1))
    >>> len(x)
    0

    """

    def __len__(self) -> int:
        shape = qnp.shape(self)
        return shape[0] if shape else 0


# -----------------------------------------------
# `__length_hint__`


class LaxLengthHintMixin:
    """Mixin for ``__length_hint__`` reading the leading axis from ``self.shape``.

    Neither `jax.lax` nor `jax.numpy` has a ``length_hint`` function; the hint
    is taken from the shape directly.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, LaxLengthHintMixin

    >>> class Val(AbstractVal, LaxLengthHintMixin):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> x.__length_hint__()
    3

    >>> x = Val(jnp.array(0))
    >>> x.__length_hint__()
    0

    """

    def __length_hint__(self: HasShape) -> int:
        return self.shape[0] if self.shape else 0


class NumpyLengthHintMixin:
    """Mixin for ``__length_hint__`` using quaxified `jax.numpy.shape`.

    `jax.numpy` has no ``length_hint`` function; the shape is obtained first
    and the leading axis returned.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, NumpyLengthHintMixin

    >>> class Val(AbstractVal, NumpyLengthHintMixin):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> x.__length_hint__()
    3

    >>> x = Val(jnp.array(1))
    >>> x.__length_hint__()
    0

    """

    def __length_hint__(self) -> int:
        shape = qnp.shape(self)
        return shape[0] if shape else 0


# -----------------------------------------------
# `__getitem__`


class NumpyGetItemMixin(Generic[R]):
    """Mixin for ``__getitem__`` using NumPy-style indexing, via `quax.quaxify`.

    Indexing goes through `quax.quaxify`, so a custom `quax.ArrayValue` with a
    registered gather rule handles it; otherwise the value is materialised and
    indexed, as with the other mixins.

    !!! note
        There is no `Lax` counterpart: `jax.lax` has no general indexing
        primitive (only `jax.lax.slice`, `jax.lax.dynamic_slice` and
        `jax.lax.gather`, none of which accept a Python index expression).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, NumpyGetItemMixin

    >>> class Val(AbstractVal, NumpyGetItemMixin[Array]):
    ...     v: Array

    >>> x = Val(jnp.array([10, 20, 30]))
    >>> x[0]
    Array(10, dtype=int32)

    Slices and fancy indexing work too:

    >>> x[1:]
    Array([20, 30], dtype=int32)

    >>> x[jnp.array([0, 2])]
    Array([10, 30], dtype=int32)

    """

    def __getitem__(self, key: object) -> R:
        return _quax_getitem(self, key)
