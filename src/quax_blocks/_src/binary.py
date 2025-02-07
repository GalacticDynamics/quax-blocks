"""Binary operations for Array-ish objects."""

# fmt: off
__all__ = [
    "LaxBinaryOpsMixin", "NumpyBinaryOpsMixin",
    # ========== Float Operations ==========
    "LaxMathMixin", "NumpyMathMixin",
    # ----- add -----
    "LaxBothAddMixin", "NumpyBothAddMixin",
    "LaxAddMixin", "NumpyAddMixin",  # __add__
    "LaxRAddMixin", "NumpyRAddMixin",  # __radd__
    # ----- sub -----
    "LaxBothSubMixin", "NumpyBothSubMixin",
    "LaxSubMixin", "NumpySubMixin",  # __sub__
    "LaxRSubMixin", "NumpyRSubMixin",  # __rsub__
    # ----- mul -----
    "LaxBothMulMixin", "NumpyBothMulMixin",
    "LaxMulMixin", "NumpyMulMixin",  # __mul__
    "LaxRMulMixin", "NumpyRMulMixin",  # __rmul__
    # ---- matmul -----
    "LaxBothMatMulMixin", "NumpyBothMatMulMixin",
    "LaxMatMulMixin", "NumpyMatMulMixin",  # __matmul__
    "LaxRMatMulMixin", "NumpyRMatMulMixin",  # __rmatmul__
    # ----- truediv -----
    "LaxBothTrueDivMixin", "NumpyBothTrueDivMixin",
    "LaxTrueDivMixin", "NumpyTrueDivMixin",  # __truediv__
    "LaxRTrueDivMixin", "NumpyRTrueDivMixin",  # __rtruediv__
    # ----- floordiv -----
    "LaxBothFloorDivMixin", "NumpyBothFloorDivMixin",
    "LaxFloorDivMixin", "NumpyFloorDivMixin",  # __floordiv__
    "LaxRFloorDivMixin", "NumpyRFloorDivMixin",  # __rfloordiv__
    # ----- mod -----
    "LaxBothModMixin", "NumpyBothModMixin",
    "LaxModMixin", "NumpyModMixin",  # __mod__
    "LaxRModMixin", "NumpyRModMixin",  # __rmod__
    # ----- divmod -----
    "NumpyBothDivModMixin",
    # "LaxDivModMixin",
    "NumpyDivModMixin",
    # "LaxRDivModMixin",
    "NumpyRDivModMixin",
    # ----- pow -----
    "LaxBothPowMixin", "NumpyBothPowMixin",
    "LaxPowMixin", "NumpyPowMixin",  # __pow__
    "LaxRPowMixin", "NumpyRPowMixin",  # __rpow__
    # ========== Bitwise Operations ==========
    "LaxBitwiseMixin", "NumpyBitwiseMixin",
    # ----- lshift -----
    "LaxBothLShiftMixin", "NumpyBothLShiftMixin",
    "LaxLShiftMixin", "NumpyLShiftMixin",  # __lshift__
    "LaxRLShiftMixin", "NumpyRLShiftMixin",  # __rlshift__
    # ----- rshift -----
    "LaxBothRShiftMixin", "NumpyBothRShiftMixin",
    "LaxRShiftMixin", "NumpyRShiftMixin",  # __rshift__
    "LaxRRShiftMixin", "NumpyRRShiftMixin",  # __rrshift__
    # ----- and -----
    "LaxBothAndMixin", "NumpyBothAndMixin",
    "LaxAndMixin", "NumpyAndMixin",  # __and__
    "LaxRAndMixin", "NumpyRAndMixin",  # __rand__
    # ----- xor -----
    "LaxBothXorMixin", "NumpyBothXorMixin",
    "LaxXorMixin", "NumpyXorMixin",  # __xor__
    "LaxRXorMixin", "NumpyRXorMixin",  # __rxor__
    # ----- or -----
    "LaxBothOrMixin", "NumpyBothOrMixin",
    "LaxOrMixin", "NumpyOrMixin",  # __or__
    "LaxROrMixin", "NumpyROrMixin", # __ror__
]
# fmt: on

from typing import Generic, Literal
from typing_extensions import TypeVar

import quaxed.lax as qlax
import quaxed.numpy as qnp
from plum import NotFoundLookupError

T = TypeVar("T")
R = TypeVar("R", default=bool)


# ===============================================
# Add

# -------------------------------------
# `__add__`


class LaxAddMixin(Generic[T, R]):
    """Mixin for ``__add__`` method using quaxified `jax.lax.add`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, LaxAddMixin

    >>> class Val(AbstractVal, LaxAddMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> x + x
    Array([2, 4, 6], dtype=int32)

    """

    def __add__(self, other: T) -> R:
        try:
            return qlax.add(self, other)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


class NumpyAddMixin(Generic[T, R]):
    """Mixin for ``__add__`` method using quaxified `jax.numpy.add`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, NumpyAddMixin

    >>> class Val(AbstractVal, NumpyAddMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> x + x
    Array([2, 4, 6], dtype=int32)

    """

    def __add__(self, other: T) -> R:
        try:
            return qnp.add(self, other)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


# -------------------------------------
# `__radd__`


class LaxRAddMixin(Generic[T, R]):
    """Mixin for ``__radd__`` method using quaxified `jax.lax.add`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, LaxRAddMixin

    >>> class Val(AbstractVal, LaxRAddMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> 1 + x
    Array([2, 3, 4], dtype=int32)

    """

    def __radd__(self, other: T) -> R:
        try:
            return qlax.add(other, self)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


class NumpyRAddMixin(Generic[T, R]):
    """Mixin for ``__radd__`` method using quaxified `jax.numpy.add`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, NumpyRAddMixin

    >>> class Val(AbstractVal, NumpyRAddMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> 1 + x
    Array([2, 3, 4], dtype=int32)

    """

    def __radd__(self, other: T) -> R:
        try:
            return qnp.add(other, self)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


# -------------------------------------


class LaxBothAddMixin(LaxAddMixin[T, R], LaxRAddMixin[T, R]):
    pass


class NumpyBothAddMixin(NumpyAddMixin[T, R], NumpyRAddMixin[T, R]):
    pass


# ===============================================
# Subtraction

# -------------------------------------
# `__sub__`


class LaxSubMixin(Generic[T, R]):
    """Mixin for ``__sub__`` method using quaxified `jax.lax.sub`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, LaxSubMixin

    >>> class Val(AbstractVal, LaxSubMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> x - x
    Array([0, 0, 0], dtype=int32)

    """

    def __sub__(self, other: T) -> R:
        try:
            return qlax.sub(self, other)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


class NumpySubMixin(Generic[T, R]):
    """Mixin for ``__sub__`` method using quaxified `jax.numpy.sub`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, NumpySubMixin

    >>> class Val(AbstractVal, NumpySubMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> x - x
    Array([0, 0, 0], dtype=int32)

    """

    def __sub__(self, other: T) -> R:
        try:
            return qnp.subtract(self, other)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


# -------------------------------------
# `__rsub__`


class LaxRSubMixin(Generic[T, R]):
    """Mixin for ``__rsub__`` method using quaxified `jax.lax.sub`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, LaxRSubMixin

    >>> class Val(AbstractVal, LaxRSubMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> 1 - x
    Array([ 0, -1, -2], dtype=int32)

    """

    def __rsub__(self, other: T) -> R:
        try:
            return qlax.sub(other, self)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


class NumpyRSubMixin(Generic[T, R]):
    """Mixin for ``__rsub__`` method using quaxified `jax.numpy.sub`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, NumpyRSubMixin

    >>> class Val(AbstractVal, NumpyRSubMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> 1 - x
    Array([ 0, -1, -2], dtype=int32)

    """

    def __rsub__(self, other: T) -> R:
        try:
            return qnp.subtract(other, self)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


# -------------------------------------


class LaxBothSubMixin(LaxSubMixin[T, R], LaxRSubMixin[T, R]):
    pass


class NumpyBothSubMixin(NumpySubMixin[T, R], NumpyRSubMixin[T, R]):
    pass


# ===============================================

# -------------------------------------
# `__mul__`


class LaxMulMixin(Generic[T, R]):
    """Mixin for ``__mul__`` method using quaxified `jax.lax.mul`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, LaxMulMixin

    >>> class Val(AbstractVal, LaxMulMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> x * x
    Array([1, 4, 9], dtype=int32)

    """

    def __mul__(self, other: T) -> R:
        try:
            return qlax.mul(self, other)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


class NumpyMulMixin(Generic[T, R]):
    """Mixin for ``__mul__`` method using quaxified `jax.numpy.mul`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, NumpyMulMixin

    >>> class Val(AbstractVal, NumpyMulMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> x * x
    Array([1, 4, 9], dtype=int32)

    """

    def __mul__(self, other: T) -> R:
        try:
            return qnp.multiply(self, other)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


# -------------------------------------
# `__rmul__`


class LaxRMulMixin(Generic[T, R]):
    """Mixin for ``__rmul__`` method using quaxified `jax.lax.mul`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, LaxRMulMixin

    >>> class Val(AbstractVal, LaxRMulMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> 2 * x
    Array([2, 4, 6], dtype=int32)

    """

    def __rmul__(self, other: T) -> R:
        try:
            return qlax.mul(other, self)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


class NumpyRMulMixin(Generic[T, R]):
    """Mixin for ``__rmul__`` method using quaxified `jax.numpy.mul`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, NumpyRMulMixin

    >>> class Val(AbstractVal, NumpyRMulMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> 2 * x
    Array([2, 4, 6], dtype=int32)

    """

    def __rmul__(self, other: T) -> R:
        try:
            return qnp.multiply(other, self)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


# -------------------------------------


class LaxBothMulMixin(LaxMulMixin[T, R], LaxRMulMixin[T, R]):
    pass


class NumpyBothMulMixin(NumpyMulMixin[T, R], NumpyRMulMixin[T, R]):
    pass


# ===============================================

# -------------------------------------
# `__matmul__`


class LaxMatMulMixin(Generic[T, R]):
    """Mixin for ``__matmul__`` method using quaxified `jax.lax.matmul`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, LaxMatMulMixin

    >>> class Val(AbstractVal, LaxMatMulMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([[1, 2], [3, 4]]))
    >>> y = Val(jnp.array([[5, 6], [7, 8]]))
    >>> x @ y
    Array([[19, 22],
           [43, 50]], dtype=int32)

    """

    def __matmul__(self, other: T) -> R:
        try:
            return qlax.dot(self, other)  # TODO: is this the right operator?
        except (TypeError, NotFoundLookupError):
            return NotImplemented


class NumpyMatMulMixin(Generic[T, R]):
    """Mixin for ``__matmul__`` method using quaxified `jax.numpy.matmul`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, NumpyMatMulMixin

    >>> class Val(AbstractVal, NumpyMatMulMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([[1, 2], [3, 4]]))
    >>> y = Val(jnp.array([[5, 6], [7, 8]]))
    >>> x @ y
    Array([[19, 22],
           [43, 50]], dtype=int32)

    """

    def __matmul__(self, other: T) -> R:
        try:
            return qnp.matmul(self, other)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


# -------------------------------------


class LaxRMatMulMixin(Generic[T, R]):
    """Mixin for ``__rmatmul__`` method using quaxified `jax.lax.matmul`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, LaxRMatMulMixin

    >>> class Val(AbstractVal, LaxRMatMulMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([[1, 2], [3, 4]]))
    >>> y = jnp.array([[5, 6], [7, 8]])
    >>> y @ x
    Array([[23, 34],
           [31, 46]], dtype=int32)

    """

    def __rmatmul__(self, other: T) -> R:
        try:
            return qlax.dot(other, self)  # TODO: is this the right operator?
        except (TypeError, NotFoundLookupError):
            return NotImplemented


class NumpyRMatMulMixin(Generic[T, R]):
    """Mixin for ``__rmatmul__`` method using quaxified `jax.numpy.matmul`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, NumpyRMatMulMixin

    >>> class Val(AbstractVal, NumpyRMatMulMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([[1, 2], [3, 4]]))
    >>> y = jnp.array([[5, 6], [7, 8]])
    >>> y @ x
    Array([[23, 34],
           [31, 46]], dtype=int32)

    """

    def __rmatmul__(self, other: T) -> R:
        try:
            return qnp.matmul(other, self)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


# -------------------------------------


class LaxBothMatMulMixin(LaxMatMulMixin[T, R], LaxRMatMulMixin[T, R]):
    pass


class NumpyBothMatMulMixin(NumpyMatMulMixin[T, R], NumpyRMatMulMixin[T, R]):
    pass


# ===============================================
# Float Division

# -------------------------------------
# `__truediv__`


class LaxTrueDivMixin(Generic[T, R]):
    """Mixin for ``__truediv__`` method using quaxified `jax.lax.truediv`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, LaxTrueDivMixin

    >>> class Val(AbstractVal, LaxTrueDivMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> x / 2  # Note: integer division
    Array([0, 1, 1], dtype=int32)

    """

    def __truediv__(self, other: T) -> R:
        try:
            return qlax.div(self, other)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


class NumpyTrueDivMixin(Generic[T, R]):
    """Mixin for ``__truediv__`` method using quaxified `jax.numpy.true_divide`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, NumpyTrueDivMixin

    >>> class Val(AbstractVal, NumpyTrueDivMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> x / 2
    Array([0.5, 1. , 1.5], dtype=float32)

    """

    def __truediv__(self, other: T) -> R:
        try:
            return qnp.true_divide(self, other)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


# -------------------------------------
# `__rtruediv__`


class LaxRTrueDivMixin(Generic[T, R]):
    """Mixin for ``__rtruediv__`` method using quaxified `jax.lax.truediv`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, LaxRTrueDivMixin

    >>> class Val(AbstractVal, LaxRTrueDivMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> 2 / x
    Array([2, 1, 0], dtype=int32)

    """

    def __rtruediv__(self, other: T) -> R:
        try:
            return qlax.div(other, self)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


class NumpyRTrueDivMixin(Generic[T, R]):
    """Mixin for ``__rtruediv__`` method using quaxified `jax.numpy.true_divide`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, NumpyRTrueDivMixin

    >>> class Val(AbstractVal, NumpyRTrueDivMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> 2 / x
    Array([2. , 1. , 0.6666667], dtype=float32)

    """

    def __rtruediv__(self, other: T) -> R:
        try:
            return qnp.true_divide(other, self)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


# -------------------------------------


class LaxBothTrueDivMixin(LaxTrueDivMixin[T, R], LaxRTrueDivMixin[T, R]):
    pass


class NumpyBothTrueDivMixin(NumpyTrueDivMixin[T, R], NumpyRTrueDivMixin[T, R]):
    pass


# ===============================================
# Integer Division

# -------------------------------------
# Floor division


class LaxFloorDivMixin(Generic[T, R]):
    """Mixin for ``__floordiv__`` method using quaxified `jax.lax`.

    Note that lax does not have a floor division function, so this is
    ``lax.floor(lax.div(x, y))``.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, LaxFloorDivMixin

    >>> class Val(AbstractVal, LaxFloorDivMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1., 2, 3]))
    >>> x // 2.
    Array([0., 1., 1.], dtype=float32)

    """

    def __floordiv__(self, other: T) -> R:
        try:
            return qlax.floor(qlax.div(self, other))
        except (TypeError, NotFoundLookupError):
            return NotImplemented


class NumpyFloorDivMixin(Generic[T, R]):
    """Mixin for ``__floordiv__`` method using quaxified `jax.numpy.floor_divide`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, NumpyFloorDivMixin

    >>> class Val(AbstractVal, NumpyFloorDivMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> x // 2
    Array([0, 1, 1], dtype=int32)

    """

    def __floordiv__(self, other: T) -> R:
        try:
            return qnp.floor_divide(self, other)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


# -------------------------------------
# `__rfloordiv__`


class LaxRFloorDivMixin(Generic[T, R]):
    """Mixin for ``__rfloordiv__`` method using quaxified `jax.lax`.

    Note that lax does not have a floor division function, so this is
    ``lax.floor(lax.div(x, y))``.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, LaxRFloorDivMixin

    >>> class Val(AbstractVal, LaxRFloorDivMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1., 2, 3]))
    >>> 2. // x
    Array([2., 1., 0.], dtype=float32)

    """

    def __rfloordiv__(self, other: T) -> R:
        try:
            return qlax.floor(qlax.div(other, self))
        except (TypeError, NotFoundLookupError):
            return NotImplemented


class NumpyRFloorDivMixin(Generic[T, R]):
    """Mixin for ``__rfloordiv__`` method using quaxified `jax.numpy.floor_divide`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, NumpyRFloorDivMixin

    >>> class Val(AbstractVal, NumpyRFloorDivMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> 2 // x
    Array([2, 1, 0], dtype=int32)

    """

    def __rfloordiv__(self, other: T) -> R:
        try:
            return qnp.floor_divide(other, self)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


# -------------------------------------


class LaxBothFloorDivMixin(LaxFloorDivMixin[T, R], LaxRFloorDivMixin[T, R]):
    pass


class NumpyBothFloorDivMixin(NumpyFloorDivMixin[T, R], NumpyRFloorDivMixin[T, R]):
    pass


# ===============================================
# Modulus

# -------------------------------------
# `__mod__`


class LaxModMixin(Generic[T, R]):
    """Mixin for ``__mod__`` method using quaxified `jax.lax.rem`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, LaxModMixin

    >>> class Val(AbstractVal, LaxModMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> x % 2
    Array([1, 0, 1], dtype=int32)

    """

    def __mod__(self, other: T) -> R:
        try:
            return qlax.rem(self, other)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


class NumpyModMixin(Generic[T, R]):
    """Mixin for ``__mod__`` method using quaxified `jax.numpy.mod`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, NumpyModMixin

    >>> class Val(AbstractVal, NumpyModMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> x % 2
    Array([1, 0, 1], dtype=int32)

    """

    def __mod__(self, other: T) -> R:
        try:
            return qnp.mod(self, other)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


# -------------------------------------
# `__rmod__`


class LaxRModMixin(Generic[T, R]):
    """Mixin for ``__rmod__`` method using quaxified `jax.lax.rem`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, LaxRModMixin

    >>> class Val(AbstractVal, LaxRModMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> 2 % x
    Array([0, 0, 2], dtype=int32)

    """

    def __rmod__(self, other: T) -> R:
        try:
            return qlax.rem(other, self)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


class NumpyRModMixin(Generic[T, R]):
    """Mixin for ``__rmod__`` method using quaxified `jax.numpy.mod`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, NumpyRModMixin

    >>> class Val(AbstractVal, NumpyRModMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> 2 % x
    Array([0, 0, 2], dtype=int32)

    """

    def __rmod__(self, other: T) -> R:
        try:
            return qnp.mod(other, self)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


# -------------------------------------


class LaxBothModMixin(LaxModMixin[T, R], LaxRModMixin[T, R]):
    pass


class NumpyBothModMixin(NumpyModMixin[T, R], NumpyRModMixin[T, R]):
    pass


# ===============================================
# Divmod

# -------------------------------------
# `__divmod__`


# TODO: a jax.lax.divmod equivalent?


class NumpyDivModMixin(Generic[T, R]):
    """Mixin for ``__divmod__`` method using quaxified `jax.numpy.divmod`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, NumpyDivModMixin

    >>> class Val(AbstractVal, NumpyDivModMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([5, 7, 9]))
    >>> divmod(x, 2)
    (Array([2, 3, 4], dtype=int32), Array([1, 1, 1], dtype=int32))

    """

    def __divmod__(self, other: T) -> R:
        try:
            return qnp.divmod(self, other)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


# -------------------------------------
# `__rdivmod__`


# TODO: a jax.lax.divmod equivalent?


class NumpyRDivModMixin(Generic[T, R]):
    """Mixin for ``__rdivmod__`` method using quaxified `jax.numpy.divmod`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, NumpyRDivModMixin

    >>> class Val(AbstractVal, NumpyRDivModMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([5, 7, 9]))
    >>> divmod(20, x)
    (Array([4, 2, 2], dtype=int32), Array([0, 6, 2], dtype=int32))

    """

    def __rdivmod__(self, other: T) -> R:
        try:
            return qnp.divmod(other, self)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


# -------------------------------------


class NumpyBothDivModMixin(NumpyDivModMixin[T, R], NumpyRDivModMixin[T, R]):
    pass


# ===============================================
# Power

# -------------------------------------
# `__pow__`


class LaxPowMixin(Generic[T, R]):
    """Mixin for ``__pow__`` method using quaxified `jax.lax.pow`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, LaxPowMixin

    >>> class Val(AbstractVal, LaxPowMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1., 2, 3]))  # must be floating
    >>> x ** 2
    Array([1., 4., 9.], dtype=float32)

    """

    def __pow__(self, other: T) -> R:
        try:
            return qlax.pow(self, other)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


class NumpyPowMixin(Generic[T, R]):
    """Mixin for ``__pow__`` method using quaxified `jax.numpy.pow`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, NumpyPowMixin

    >>> class Val(AbstractVal, NumpyPowMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> x ** 2
    Array([1, 4, 9], dtype=int32)

    """

    def __pow__(self, other: T) -> R:
        try:
            return qnp.power(self, other)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


# -------------------------------------
# `__rpow__`


class LaxRPowMixin(Generic[T, R]):
    """Mixin for ``__rpow__`` method using quaxified `jax.lax.pow`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, LaxRPowMixin

    >>> class Val(AbstractVal, LaxRPowMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> 2. ** x
    Array([2., 4., 8.], dtype=float32)

    """

    def __rpow__(self, other: T) -> R:
        try:
            return qlax.pow(other, self)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


class NumpyRPowMixin(Generic[T, R]):
    """Mixin for ``__rpow__`` method using quaxified `jax.numpy.pow`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, NumpyRPowMixin

    >>> class Val(AbstractVal, NumpyRPowMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> 2 ** x
    Array([2, 4, 8], dtype=int32)

    """

    def __rpow__(self, other: T) -> R:
        try:
            return qnp.power(other, self)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


# -------------------------------------


class LaxBothPowMixin(LaxPowMixin[T, R], LaxRPowMixin[T, R]):
    pass


class NumpyBothPowMixin(NumpyPowMixin[T, R], NumpyRPowMixin[T, R]):
    pass


# ===============================================
# Left Shift

# -------------------------------------
# `__lshift__`


class LaxLShiftMixin(Generic[T, R]):
    """Mixin for ``__lshift__`` method using quaxified `jax.lax.lshift`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, LaxLShiftMixin

    >>> class Val(AbstractVal, LaxLShiftMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> x << 1
    Array([2, 4, 6], dtype=int32)

    """

    def __lshift__(self, other: T) -> R:
        try:
            return qlax.shift_left(self, other)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


class NumpyLShiftMixin(Generic[T, R]):
    """Mixin for ``__lshift__`` method using quaxified `jax.numpy.lshift`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, NumpyLShiftMixin

    >>> class Val(AbstractVal, NumpyLShiftMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> x << 1
    Array([2, 4, 6], dtype=int32)

    """

    def __lshift__(self, other: T) -> R:
        try:
            return qnp.left_shift(self, other)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


# -------------------------------------
# `__rlshift__`


class LaxRLShiftMixin(Generic[T, R]):
    """Mixin for ``__rlshift__`` method using quaxified `jax.lax.lshift`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, LaxRLShiftMixin

    >>> class Val(AbstractVal, LaxRLShiftMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> 1 << x
    Array([2, 4, 8], dtype=int32)

    """

    def __rlshift__(self, other: T) -> R:
        try:
            return qlax.shift_left(other, self)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


class NumpyRLShiftMixin(Generic[T, R]):
    """Mixin for ``__rlshift__`` method using quaxified `jax.numpy.lshift`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, NumpyRLShiftMixin

    >>> class Val(AbstractVal, NumpyRLShiftMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> 1 << x
    Array([2, 4, 8], dtype=int32)

    """

    def __rlshift__(self, other: T) -> R:
        try:
            return qnp.left_shift(other, self)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


# -------------------------------------


class LaxBothLShiftMixin(LaxLShiftMixin[T, R], LaxRLShiftMixin[T, R]):
    pass


class NumpyBothLShiftMixin(NumpyLShiftMixin[T, R], NumpyRLShiftMixin[T, R]):
    pass


# ===============================================
# Right Shift

# -------------------------------------
# `__rshift__`


class LaxRShiftMixin(Generic[T, R]):
    """Mixin for ``__rshift__`` method using quaxified `jax.lax.shift_right`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, LaxRShiftMixin

    >>> class Val(AbstractVal, LaxRShiftMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([2, 4, 8]))
    >>> x >> 1
    Array([1, 2, 4], dtype=int32)

    """

    _RIGHT_SHIFT_LOGICAL: Literal[True, False] = True

    def __rshift__(self, other: T) -> R:
        try:
            return qlax.cond(
                self._RIGHT_SHIFT_LOGICAL,
                qlax.shift_right_logical,
                qlax.shift_right_arithmetic,
                self,
                other,
            )
        except (TypeError, NotFoundLookupError):
            return NotImplemented


class NumpyRShiftMixin(Generic[T, R]):
    """Mixin for ``__rshift__`` method using quaxified `jax.numpy.rshift`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, NumpyRShiftMixin

    >>> class Val(AbstractVal, NumpyRShiftMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([2, 4, 8]))
    >>> x >> 1
    Array([1, 2, 4], dtype=int32)

    """

    def __rshift__(self, other: T) -> R:
        try:
            return qnp.right_shift(self, other)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


# -------------------------------------
# `__rrshift__`


class LaxRRShiftMixin(Generic[T, R]):
    """Mixin for ``__rrshift__`` method using quaxified `jax.lax.rshift`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, LaxRRShiftMixin

    >>> class Val(AbstractVal, LaxRRShiftMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([2, 4, 8]))
    >>> 16 >> x
    Array([4, 1, 0], dtype=int32)

    """

    _RIGHT_SHIFT_LOGICAL: Literal[True, False] = True

    def __rrshift__(self, other: T) -> R:
        try:
            return qlax.cond(
                self._RIGHT_SHIFT_LOGICAL,
                qlax.shift_right_logical,
                qlax.shift_right_arithmetic,
                other,
                self,
            )
        except (TypeError, NotFoundLookupError):
            return NotImplemented


class NumpyRRShiftMixin(Generic[T, R]):
    """Mixin for ``__rrshift__`` method using quaxified `jax.numpy.rshift`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, NumpyRRShiftMixin

    >>> class Val(AbstractVal, NumpyRRShiftMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([2, 4, 8]))
    >>> 16 >> x
    Array([4, 1, 0], dtype=int32)

    """

    def __rrshift__(self, other: T) -> R:
        try:
            return qnp.right_shift(other, self)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


# -------------------------------------


class LaxBothRShiftMixin(LaxRShiftMixin[T, R], LaxRRShiftMixin[T, R]):
    pass


class NumpyBothRShiftMixin(NumpyRShiftMixin[T, R], NumpyRRShiftMixin[T, R]):
    pass


# ===============================================
# And

# -------------------------------------
# `__and__`


class LaxAndMixin(Generic[T, R]):
    """Mixin for ``__and__`` method using quaxified `jax.lax.and_`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, LaxAndMixin

    >>> class Val(AbstractVal, LaxAndMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> x & 1
    Array([1, 0, 1], dtype=int32)

    """

    def __and__(self, other: T) -> R:
        try:
            return qlax.bitwise_and(self, other)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


class NumpyAndMixin(Generic[T, R]):
    """Mixin for ``__and__`` method using quaxified `jax.numpy.and_`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, NumpyAndMixin

    >>> class Val(AbstractVal, NumpyAndMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> x & 1
    Array([1, 0, 1], dtype=int32)

    """

    def __and__(self, other: T) -> R:
        try:
            return qnp.bitwise_and(self, other)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


# -------------------------------------
# `__rand__`


class LaxRAndMixin(Generic[T, R]):
    """Mixin for ``__rand__`` method using quaxified `jax.lax.and_`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, LaxRAndMixin

    >>> class Val(AbstractVal, LaxRAndMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> 1 & x
    Array([1, 0, 1], dtype=int32)

    """

    def __rand__(self, other: T) -> R:
        try:
            return qlax.bitwise_and(other, self)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


class NumpyRAndMixin(Generic[T, R]):
    """Mixin for ``__rand__`` method using quaxified `jax.numpy.and_`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, NumpyRAndMixin

    >>> class Val(AbstractVal, NumpyRAndMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> 1 & x
    Array([1, 0, 1], dtype=int32)

    """

    def __rand__(self, other: T) -> R:
        try:
            return qnp.bitwise_and(other, self)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


# -------------------------------------


class LaxBothAndMixin(LaxAndMixin[T, R], LaxRAndMixin[T, R]):
    pass


class NumpyBothAndMixin(NumpyAndMixin[T, R], NumpyRAndMixin[T, R]):
    pass


# ===============================================
# Xor


# -------------------------------------
# `__xor__`


class LaxXorMixin(Generic[T, R]):
    """Mixin for ``__xor__`` method using quaxified `jax.lax.xor`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, LaxXorMixin

    >>> class Val(AbstractVal, LaxXorMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> x ^ 1
    Array([0, 3, 2], dtype=int32)

    """

    def __xor__(self, other: T) -> R:
        try:
            return qlax.bitwise_xor(self, other)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


class NumpyXorMixin(Generic[T, R]):
    """Mixin for ``__xor__`` method using quaxified `jax.numpy.xor`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, NumpyXorMixin

    >>> class Val(AbstractVal, NumpyXorMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> x ^ 1
    Array([0, 3, 2], dtype=int32)

    """

    def __xor__(self, other: T) -> R:
        try:
            return qnp.bitwise_xor(self, other)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


# -------------------------------------
# `__rxor__`


class LaxRXorMixin(Generic[T, R]):
    """Mixin for ``__rxor__`` method using quaxified `jax.lax.xor`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, LaxRXorMixin

    >>> class Val(AbstractVal, LaxRXorMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> 1 ^ x
    Array([0, 3, 2], dtype=int32)

    """

    def __rxor__(self, other: T) -> R:
        try:
            return qlax.bitwise_xor(other, self)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


class NumpyRXorMixin(Generic[T, R]):
    """Mixin for ``__rxor__`` method using quaxified `jax.numpy.xor`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, NumpyRXorMixin

    >>> class Val(AbstractVal, NumpyRXorMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> 1 ^ x
    Array([0, 3, 2], dtype=int32)

    """

    def __rxor__(self, other: T) -> R:
        try:
            return qnp.bitwise_xor(other, self)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


# -------------------------------------


class LaxBothXorMixin(LaxXorMixin[T, R], LaxRXorMixin[T, R]):
    pass


class NumpyBothXorMixin(NumpyXorMixin[T, R], NumpyRXorMixin[T, R]):
    pass


# ================================================
# Or

# -------------------------------------
# `__or__`


class LaxOrMixin(Generic[T, R]):
    """Mixin for ``__or__`` method using quaxified `jax.lax.or_`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, LaxOrMixin

    >>> class Val(AbstractVal, LaxOrMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> x | 1
    Array([1, 3, 3], dtype=int32)

    """

    def __or__(self, other: T) -> R:
        try:
            return qlax.bitwise_or(self, other)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


class NumpyOrMixin(Generic[T, R]):
    """Mixin for ``__or__`` method using quaxified `jax.numpy.or_`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, NumpyOrMixin

    >>> class Val(AbstractVal, NumpyOrMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> x | 1
    Array([1, 3, 3], dtype=int32)

    """

    def __or__(self, other: T) -> R:
        try:
            return qnp.bitwise_or(self, other)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


# ================================================
# Ror

# -------------------------------------
# `__ror__`


class LaxROrMixin(Generic[T, R]):
    """Mixin for ``__ror__`` method using quaxified `jax.lax.or_`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, LaxROrMixin

    >>> class Val(AbstractVal, LaxROrMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> 1 | x
    Array([1, 3, 3], dtype=int32)

    """

    def __ror__(self, other: T) -> R:
        try:
            return qlax.bitwise_or(other, self)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


class NumpyROrMixin(Generic[T, R]):
    """Mixin for ``__ror__`` method using quaxified `jax.numpy.or_`.

    Examples:
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, NumpyROrMixin

    >>> class Val(AbstractVal, NumpyROrMixin[object, Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> 1 | x
    Array([1, 3, 3], dtype=int32)

    """

    def __ror__(self, other: T) -> R:
        try:
            return qnp.bitwise_or(other, self)
        except (TypeError, NotFoundLookupError):
            return NotImplemented


# -------------------------------------


class LaxBothOrMixin(LaxOrMixin[T, R], LaxROrMixin[T, R]):
    pass


class NumpyBothOrMixin(NumpyOrMixin[T, R], NumpyROrMixin[T, R]):
    pass


# =================================================


class LaxMathMixin(
    LaxBothAddMixin[T, R],  # __add__, __radd__
    LaxBothSubMixin[T, R],  # __sub__, __rsub__
    LaxBothMulMixin[T, R],  # __mul__, __rmul__
    LaxBothMatMulMixin[T, R],  # __matmul__, __rmatmul__
    LaxBothTrueDivMixin[T, R],  # __truediv__, __rtruediv__
    LaxBothFloorDivMixin[T, R],  # __floordiv__, __rfloordiv__
    LaxBothModMixin[T, R],  # __mod__, __rmod__
    # TODO: divmod
    LaxBothPowMixin[T, R],  # __pow__, __rpow__
):
    pass


class NumpyMathMixin(
    NumpyBothAddMixin[T, R],  # __add__, __radd__
    NumpyBothSubMixin[T, R],  # __sub__, __rsub__
    NumpyBothMulMixin[T, R],  # __mul__, __rmul__
    NumpyBothMatMulMixin[T, R],  # __matmul__, __rmatmul__
    NumpyBothTrueDivMixin[T, R],  # __truediv__, __rtruediv__
    NumpyBothFloorDivMixin[T, R],  # __floordiv__, __rfloordiv__
    NumpyBothModMixin[T, R],  # __mod__, __rmod__
    NumpyBothDivModMixin[T, R],  # __divmod__, __rdivmod__
    NumpyBothPowMixin[T, R],  # __pow__, __rpow__
):
    pass


class LaxBitwiseMixin(
    LaxBothLShiftMixin[T, R],  # __lshift__, __rlshift__
    LaxBothRShiftMixin[T, R],  # __rshift__, __rrshift__
    LaxBothAndMixin[T, R],  # __and__, __rand__
    LaxBothXorMixin[T, R],  # __xor__, __rxor__
    LaxBothOrMixin[T, R],  # __or__, __ror__
):
    pass


class NumpyBitwiseMixin(
    NumpyBothLShiftMixin[T, R],  # __lshift__, __rlshift__
    NumpyBothRShiftMixin[T, R],  # __rshift__, __rrshift__
    NumpyBothAndMixin[T, R],  # __and__, __rand__
    NumpyBothXorMixin[T, R],  # __xor__, __rxor__
    NumpyBothOrMixin[T, R],  # __or__, __ror__
):
    pass


# =================================================


class LaxBinaryOpsMixin(LaxMathMixin[T, R], LaxBitwiseMixin[T, R]):
    pass


class NumpyBinaryOpsMixin(NumpyMathMixin[T, R], NumpyBitwiseMixin[T, R]):
    pass
