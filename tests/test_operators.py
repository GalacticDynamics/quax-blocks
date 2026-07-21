"""Behavioural tests for the operator mixins.

The mixins' docstrings each carry one happy-path doctest. This module covers
what those cannot: reflected-operand dispatch, the ``NotImplemented`` fallback
paths, dtype edge cases, and -- most importantly -- the documented places where
the ``Lax`` flavour deliberately diverges from Python/NumPy semantics, so that
those divergences cannot change silently.
"""

import copy
import operator
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array

import quax_blocks as qb


def make(*mixins: type, v: Array) -> Any:
    """Build a `quax.ArrayValue` mixing in ``mixins`` and holding ``v``."""
    cls = type("Val", (qb.AbstractVal, *mixins), {"__annotations__": {"v": Array}})
    return cls(v)


I1 = jnp.array([4, 5, 6])
I2 = jnp.array([2, 2, 2])
F1 = jnp.array([4.0, 5.0, 6.0])
F2 = jnp.array([2.0, 2.0, 2.0])


# ===================================================================
# Forward binary operators


@pytest.mark.parametrize(
    ("lax_mixin", "numpy_mixin", "op"),
    [
        (qb.LaxAddMixin, qb.NumpyAddMixin, operator.add),
        (qb.LaxSubMixin, qb.NumpySubMixin, operator.sub),
        (qb.LaxMulMixin, qb.NumpyMulMixin, operator.mul),
    ],
)
def test_arithmetic_matches_numpy(
    lax_mixin: type, numpy_mixin: type, op: Callable[[Any, Any], Any]
) -> None:
    """Lax and NumPy flavours agree with each other and with NumPy itself."""
    expected = op(np.asarray(I1), np.asarray(I2))
    for mixin in (lax_mixin, numpy_mixin):
        got = op(make(mixin, v=I1), I2)
        np.testing.assert_array_equal(np.asarray(got), expected)


@pytest.mark.parametrize(
    ("lax_mixin", "numpy_mixin", "op"),
    [
        (qb.LaxAndMixin, qb.NumpyAndMixin, operator.and_),
        (qb.LaxOrMixin, qb.NumpyOrMixin, operator.or_),
        (qb.LaxXorMixin, qb.NumpyXorMixin, operator.xor),
        (qb.LaxLShiftMixin, qb.NumpyLShiftMixin, operator.lshift),
    ],
)
def test_bitwise_matches_numpy(
    lax_mixin: type, numpy_mixin: type, op: Callable[[Any, Any], Any]
) -> None:
    """Bitwise operators agree with NumPy on integer operands."""
    expected = op(np.asarray(I1), np.asarray(I2))
    for mixin in (lax_mixin, numpy_mixin):
        got = op(make(mixin, v=I1), I2)
        np.testing.assert_array_equal(np.asarray(got), expected)


# ===================================================================
# Reflected operators


@pytest.mark.parametrize(
    ("mixin", "op", "forward", "reflected"),
    [
        (qb.LaxRSubMixin, operator.sub, "__sub__", "__rsub__"),
        (qb.NumpyRSubMixin, operator.sub, "__sub__", "__rsub__"),
        (qb.LaxRAddMixin, operator.add, "__add__", "__radd__"),
        (qb.NumpyRAddMixin, operator.add, "__add__", "__radd__"),
        (qb.NumpyRTrueDivMixin, operator.truediv, "__truediv__", "__rtruediv__"),
    ],
)
def test_reflected_operand_order(
    mixin: type, op: Callable[[Any, Any], Any], forward: str, reflected: str
) -> None:
    """``op(other, val)`` computes ``other OP self``, not the other way round."""
    val = make(mixin, v=I2)
    expected = op(np.asarray(I1), np.asarray(I2))

    # Premise: the reflected method is reached only because the left operand
    # declines. Assert it, so this test cannot silently stop exercising the
    # `__r*__` path if jax ever starts handling ArrayValue operands directly.
    assert getattr(type(I1), forward)(I1, val) is NotImplemented

    # Through the operator -- the real dispatch path a user hits ...
    np.testing.assert_allclose(np.asarray(op(I1, val)), expected)

    # ... and directly, which pins the operand order regardless of dispatch.
    np.testing.assert_allclose(np.asarray(getattr(val, reflected)(I1)), expected)


def test_both_operands_declining_raises_clean_typeerror() -> None:
    """When both sides decline, Python raises a normal TypeError.

    The left operand defers, Python then tries the mixin's ``__rsub__``, which
    cannot handle a non-array operand and returns ``NotImplemented`` too. The
    fallback chain must terminate in the standard error rather than leaking a
    library-internal exception.
    """

    class Uncooperative:
        """Always defers, forcing Python to try the reflected operation."""

        def __sub__(self, other: object) -> Any:
            return NotImplemented

    with pytest.raises(TypeError, match="unsupported operand type"):
        _ = Uncooperative() - make(qb.NumpyRSubMixin, v=I1)  # type: ignore[operator]


# ===================================================================
# NotImplemented guards


@pytest.mark.parametrize(
    ("mixin", "method"),
    [
        (qb.LaxAddMixin, "__add__"),
        (qb.NumpyAddMixin, "__add__"),
        (qb.LaxSubMixin, "__sub__"),
        (qb.NumpySubMixin, "__sub__"),
        (qb.LaxLtMixin, "__lt__"),
        (qb.NumpyLtMixin, "__lt__"),
        (qb.LaxGtMixin, "__gt__"),
        (qb.NumpyGtMixin, "__gt__"),
        (qb.LaxGeMixin, "__ge__"),
        (qb.NumpyGeMixin, "__ge__"),
    ],
)
def test_unsupported_operand_returns_notimplemented(mixin: type, method: str) -> None:
    """Operators return NotImplemented (not raise) for non-array operands.

    Returning ``NotImplemented`` is what lets Python fall back to the other
    operand's reflected method and ultimately raise a good ``TypeError``.
    """
    val = make(mixin, v=I1)
    assert getattr(val, method)("not-an-array") is NotImplemented


# ===================================================================
# Comparisons


@pytest.mark.parametrize(
    ("lax_mixin", "numpy_mixin", "op"),
    [
        (qb.LaxLtMixin, qb.NumpyLtMixin, operator.lt),
        (qb.LaxLeMixin, qb.NumpyLeMixin, operator.le),
        (qb.LaxGtMixin, qb.NumpyGtMixin, operator.gt),
        (qb.LaxGeMixin, qb.NumpyGeMixin, operator.ge),
    ],
)
def test_comparisons_match_numpy(
    lax_mixin: type, numpy_mixin: type, op: Callable[[Any, Any], Any]
) -> None:
    """Comparison operators agree with NumPy, in both flavours."""
    expected = op(np.asarray(I1), np.asarray(I2))
    for mixin in (lax_mixin, numpy_mixin):
        got = op(make(mixin, v=I1), I2)
        np.testing.assert_array_equal(np.asarray(got), expected)


def test_eq_requires_explicit_assignment() -> None:
    """Equinox defines __eq__, so the mixin must be assigned explicitly."""
    cls = type(
        "Val",
        (qb.AbstractVal, qb.NumpyEqMixin),
        {"__annotations__": {"v": Array}, "__eq__": qb.NumpyEqMixin.__eq__},
    )
    np.testing.assert_array_equal(np.asarray(cls(I1) == I1), [True, True, True])


# ===================================================================
# Documented Lax-vs-NumPy divergences
#
# These lock in behaviour that `docs/guides/mixins.md` documents. They are not
# endorsements -- they exist so the divergence cannot change unnoticed.


def test_lax_truediv_is_integer_division_on_ints() -> None:
    """Lax `/` truncates on integers; NumPy `/` is true division."""
    lax_got = make(qb.LaxTrueDivMixin, v=I1) / I2
    np.testing.assert_array_equal(np.asarray(lax_got), [2, 2, 3])
    assert np.issubdtype(np.asarray(lax_got).dtype, np.integer)

    numpy_got = make(qb.NumpyTrueDivMixin, v=I1) / I2
    np.testing.assert_allclose(np.asarray(numpy_got), [2.0, 2.5, 3.0])
    assert np.issubdtype(np.asarray(numpy_got).dtype, np.floating)


def test_lax_mod_takes_sign_of_dividend() -> None:
    """Lax `%` is a C-style remainder; NumPy/Python `%` follows the divisor."""
    a, b = jnp.array([-7, 7]), jnp.array([3, -3])

    lax_got = make(qb.LaxModMixin, v=a) % b
    np.testing.assert_array_equal(np.asarray(lax_got), [-1, 1])  # sign of dividend

    numpy_got = make(qb.NumpyModMixin, v=a) % b
    np.testing.assert_array_equal(np.asarray(numpy_got), [2, -2])  # sign of divisor
    assert [-7 % 3, 7 % -3] == [2, -2]  # ... which is what Python does


@pytest.mark.parametrize(
    ("mixin", "op"),
    [
        (qb.LaxPowMixin, operator.pow),  # jax.lax.pow rejects int dtypes
        (qb.LaxFloorDivMixin, operator.floordiv),  # jax.lax.floor rejects int dtypes
    ],
)
def test_lax_float_only_ops_reject_integers(
    mixin: type, op: Callable[[Any, Any], Any]
) -> None:
    """These lax primitives are float-only, so integer operands fail."""
    with pytest.raises(TypeError):
        op(make(mixin, v=I1), I2)


@pytest.mark.parametrize(
    ("mixin", "op", "expected"),
    [
        (qb.LaxPowMixin, operator.pow, [16.0, 25.0, 36.0]),
        (qb.LaxFloorDivMixin, operator.floordiv, [2.0, 2.0, 3.0]),
    ],
)
def test_lax_float_only_ops_work_on_floats(
    mixin: type, op: Callable[[Any, Any], Any], expected: list[float]
) -> None:
    """The same operators work once given floating-point operands."""
    got = op(make(mixin, v=F1), F2)
    np.testing.assert_allclose(np.asarray(got), expected)


# ===================================================================
# divmod


def test_numpy_divmod_returns_pair() -> None:
    """Divmod returns a (quotient, remainder) tuple matching NumPy."""
    got = divmod(make(qb.NumpyDivModMixin, v=jnp.array([7, 8])), jnp.array([3, 3]))
    assert isinstance(got, tuple)
    assert len(got) == 2
    exp_q, exp_r = np.divmod(np.array([7, 8]), 3)
    np.testing.assert_array_equal(np.asarray(got[0]), exp_q)
    np.testing.assert_array_equal(np.asarray(got[1]), exp_r)


# ===================================================================
# Unary / rounding


@pytest.mark.parametrize(
    ("lax_mixin", "numpy_mixin", "op"),
    [
        (qb.LaxNegMixin, qb.NumpyNegMixin, operator.neg),
        (qb.LaxAbsMixin, qb.NumpyAbsMixin, abs),
    ],
)
def test_unary_matches_numpy(
    lax_mixin: type, numpy_mixin: type, op: Callable[[Any], Any]
) -> None:
    """Unary operators agree with NumPy, in both flavours."""
    neg = jnp.array([-1, 2, -3])
    expected = op(np.asarray(neg))
    for mixin in (lax_mixin, numpy_mixin):
        np.testing.assert_array_equal(np.asarray(op(make(mixin, v=neg))), expected)


def test_lax_and_numpy_round_use_different_tie_breaking() -> None:
    """Lax rounds ties away from zero; NumPy uses banker's rounding."""
    ties = jnp.array([0.5, 1.5, 2.5])
    lax_got = round(make(qb.LaxRoundMixin, v=ties))
    np.testing.assert_allclose(np.asarray(lax_got), [1.0, 2.0, 3.0])

    numpy_got = round(make(qb.NumpyRoundMixin, v=ties))
    np.testing.assert_allclose(np.asarray(numpy_got), [0.0, 2.0, 2.0])


# ===================================================================
# Container / copy


@pytest.mark.parametrize("mixin", [qb.LaxLenMixin, qb.NumpyLenMixin])
def test_len(mixin: type) -> None:
    """len() reads the leading axis and returns a Python int."""
    got = len(make(mixin, v=I1))
    assert got == 3
    assert isinstance(got, int)


@pytest.mark.parametrize("mixin", [qb.LaxLenMixin, qb.NumpyLenMixin])
def test_len_of_scalar_is_zero_unlike_numpy(mixin: type) -> None:
    """len() of a 0-d value returns 0, where NumPy raises TypeError.

    A documented deviation; locked in so it cannot change silently.
    """
    assert len(make(mixin, v=jnp.array(1))) == 0
    with pytest.raises(TypeError):
        len(np.array(1))


def test_copy_returns_an_array_not_the_wrapper() -> None:
    """copy() materialises to an Array rather than preserving the type.

    Documented in `docs/guides/mixins.md` ("return a new array, not a new
    instance of the custom type"); locked in so it cannot change silently.
    """
    val = make(qb.NumpyCopyMixin, v=I1)
    copied = copy.copy(val)
    assert not isinstance(copied, type(val))
    np.testing.assert_array_equal(np.asarray(copied), np.asarray(I1))
