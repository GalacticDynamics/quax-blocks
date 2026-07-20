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
#: This is not a hypothetical: `quax` >= 0.3.5 cannot be imported against the
#: declared `jax` floor (0.7.2) -- it uses `jax._src.literals.TypedInt`, which
#: only exists in newer `jax` -- so the versions resolved at the floor are
#: exactly the ones that raise `AssertionError`. See issue #46.
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
