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
        return jax.typeof(self.v)  # public accessor; `jax.core` is deprecated

    def materialise(self) -> Array:
        return self.v
