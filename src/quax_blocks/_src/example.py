"""Arrayish."""

__all__: list[str] = ["AbstractVal"]

from typing import Any

import equinox as eqx
import jax
from jaxtyping import Array
from quax import ArrayValue


class AbstractVal(ArrayValue):  # type: ignore[misc]
    """ABC for example arrayish object."""

    #: The array.
    v: eqx.AbstractVar[Array]

    def aval(self) -> Any:
        # `jax.typeof` is the public accessor; `jax.core` is the legacy path
        # JAX has been progressively deprecating.
        return jax.typeof(self.v)

    def materialise(self) -> Array:
        return self.v
