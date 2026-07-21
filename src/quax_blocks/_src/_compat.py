"""Compatibility shims across supported dependency versions."""

__all__ = ["DISPATCH_ERRORS"]

from importlib.metadata import version

from packaging.version import Version
from plum import NotFoundLookupError

#: On a dispatch miss (an operand `quax` cannot handle), the operator mixins
#: rely on `quax` raising `TypeError` / `NotFoundLookupError` so they can return
#: `NotImplemented` and let Python try the reflected operand. On `jax` < 0.9.2
#: that miss instead surfaces as an `AssertionError`: `quax`'s
#: `_default_process` ends in a bare ``assert False`` (quax `_core.py`), and on
#: those older `jax` dispatch paths the unhandled primitive reaches it, whereas
#: `jax` >= 0.9.2 rejects the argument with a `TypeError` before `quax` is
#: consulted. The trigger is the `jax` version, not the `quax` version -- a
#: fixed `quax` (0.3.7, the declared floor) still asserts under `jax` 0.7.2.
#:
#: The `jax` floor is deliberately kept at 0.7.2 (see issue #46), so the
#: affected range 0.7.2 -- 0.9.1 is supported and the guard must catch
#: `AssertionError` there. `AssertionError` is added only on that range, so on
#: `jax` >= 0.9.2 a genuine assertion failure still propagates instead of being
#: silently converted into `NotImplemented`. Once the `jax` floor is raised past
#: 0.9.2, drop `AssertionError` and delete this shim.
_JAX_RAISES_ASSERTION: bool = Version(version("jax")) < Version("0.9.2")

DISPATCH_ERRORS: tuple[type[Exception], ...] = (
    (TypeError, NotFoundLookupError, AssertionError)
    if _JAX_RAISES_ASSERTION
    else (TypeError, NotFoundLookupError)
)
