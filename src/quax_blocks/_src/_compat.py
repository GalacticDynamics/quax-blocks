"""Compatibility shims across supported dependency versions."""

__all__ = ["DISPATCH_ERRORS"]

from importlib.metadata import version

from packaging.version import Version
from plum import NotFoundLookupError

#: `quax` < 0.3.5 raises `AssertionError` out of its dispatch machinery where
#: later versions raise `TypeError` / `NotFoundLookupError`. The operator mixins
#: rely on those exceptions to return `NotImplemented`, so on older `quax` the
#: guard misses and the operator raises instead of deferring to the other
#: operand.
#:
#: Why the floor is not simply raised past 0.3.5: the released 0.3.5 and 0.3.6
#: cannot be imported against the declared `jax` floor (0.7.2) -- they reference
#: `jax._src.literals.TypedInt`, which only exists in `jax` >= 0.8.0 -- so the
#: `quax` that resolves at the floor is one of the older, `AssertionError`
#: -raising versions. That import bug is fixed upstream (quax#166 / #167); once
#: a fixed `quax` is released, raise the floor to it and delete this shim.
#: See issue #46.
#:
#: `AssertionError` is added only on the affected versions, so on a modern
#: `quax` a genuine assertion failure still propagates instead of being
#: silently converted into `NotImplemented`.
_QUAX_RAISES_ASSERTION: bool = Version(version("quax")) < Version("0.3.5")

DISPATCH_ERRORS: tuple[type[Exception], ...] = (
    (TypeError, NotFoundLookupError, AssertionError)
    if _QUAX_RAISES_ASSERTION
    else (TypeError, NotFoundLookupError)
)
