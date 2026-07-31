"""Construction that skips ``__init__`` for Array-ish objects."""

__all__ = ["SupportsUncheckedMake"]

import dataclasses
from typing import Any, NamedTuple, Self

#: What `dataclasses` uses to mark "no default was given".
_MISSING = dataclasses.MISSING


class _Spec(NamedTuple):
    """What `__make__` needs to know about a class, computed once."""

    #: Field names, for comparing against the given keys in one operation.
    names: frozenset[str]
    #: The same names in **declaration order**. Filling defaults by iterating a
    #: set would be arbitrary and, because string hashing is randomised per
    #: process, would vary between runs: side-effecting default factories would
    #: run in an unpredictable order and errors would name an arbitrary field.
    order: tuple[str, ...]
    #: Plain defaults, by name.
    defaults: dict[str, Any]
    #: Default factories, by name.
    factories: dict[str, Any]


#: Per-class field spec. A plain `dict` rather than `functools.cache`: one
#: lookup instead of a wrapper call on a hot path, and the strong reference to
#: the class costs nothing, since registering a class as a pytree node already
#: keeps it alive for the life of the process.
_SPECS: dict[type, _Spec] = {}


def _field_spec(cls: type[Any], /) -> _Spec:
    """Return the field spec for *cls*, computing it on first use."""
    spec = _SPECS.get(cls)
    if spec is None:
        spec = _SPECS[cls] = _compute_field_spec(cls)
    return spec


def _compute_field_spec(cls: type[Any], /) -> _Spec:
    """Read *cls*'s dataclass fields, rejecting what cannot be made."""
    # `__abstractvars__` / `__abstractclassvars__` are equinox's; read by name so
    # this module stays stdlib-only, and absent on a plain `abc` class.
    abstract = sorted(
        {
            *getattr(cls, "__abstractmethods__", ()),
            *getattr(cls, "__abstractvars__", ()),
            *getattr(cls, "__abstractclassvars__", ()),
        }
    )
    if abstract:
        msg = f"cannot make {cls.__name__!r}: it is abstract in {abstract}"
        raise TypeError(msg)

    fields = dataclasses.fields(cls)  # raises TypeError if not a dataclass
    return _Spec(
        names=frozenset(f.name for f in fields),
        order=tuple(f.name for f in fields),
        defaults={f.name: f.default for f in fields if f.default is not _MISSING},
        factories={
            f.name: f.default_factory
            for f in fields
            if f.default_factory is not _MISSING
        },
    )


def _complete(cls: type[Any], given: dict[str, Any], /) -> dict[str, Any]:
    """Fill defaults into *given*, or say precisely what is wrong with it.

    Walks the fields in declaration order, so factories run in the order the
    ordinary constructor would run them and the error names every missing field
    in a stable order.
    """
    spec = _field_spec(cls)

    if unknown := given.keys() - spec.names:
        msg = f"{cls.__name__} has no field(s) {sorted(unknown)}"
        raise TypeError(msg)

    fields: dict[str, Any] = {}
    missing: list[str] = []
    for name in spec.order:
        if name in given:
            fields[name] = given[name]
        elif name in spec.defaults:
            fields[name] = spec.defaults[name]
        elif name in spec.factories:
            fields[name] = spec.factories[name]()
        else:
            missing.append(name)

    if missing:
        msg = f"{cls.__name__} field(s) {missing} have no default, so must be given"
        raise TypeError(msg)
    return fields


class SupportsUncheckedMake:
    """Mixin adding `__make__`: construction that skips ``__init__``.

    A dataclass-based pytree -- an `equinox.Module`, hence any `quax.ArrayValue`
    -- carries its fields in ``__dict__`` and is rebuilt from them directly when
    JAX unflattens it. `__make__` does the same thing on purpose: it writes the
    fields and returns, so field converters, ``__post_init__`` and
    ``__check_init__`` never run.

    That is worth having where a validating constructor is provably redundant.
    A check that lowers to `equinox.error_if` costs a conditional and two
    custom-calls on *every* construction, which is pure overhead for an internal
    caller whose result cannot violate the invariant as a matter of arithmetic.

    It is also the dangerous option, which is what the name says. `__make__` will
    build an object that the ordinary constructor would reject. Reach for it
    only where the invariant is a theorem, and say which theorem at the call
    site.

    Fields are passed **by name**, so a subclass adding a field cannot silently
    change what a positional argument means. Fields with defaults may be
    omitted.

    Ruff's ``PLW3201`` objects to invented dunders (it is preview-only), but it
    fires where one is *defined*, not where it is called -- so a class that only
    calls `__make__` need do nothing. A class that wants a positional signature,
    or just a shorter name, should bind a normal one rather than redefine the
    dunder; see the last two examples.

    Examples
    --------
    >>> import equinox as eqx
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax_blocks import AbstractVal, SupportsUncheckedMake

    >>> class NonNegative(AbstractVal, SupportsUncheckedMake):
    ...     v: Array
    ...     name: str = eqx.field(default="x", static=True)
    ...
    ...     def __check_init__(self):
    ...         if jnp.any(self.v < 0):
    ...             raise ValueError("must be non-negative")

    The constructor enforces the invariant:

    >>> try: NonNegative(jnp.asarray([-1.0]))
    ... except ValueError as e: print(e)
    must be non-negative

    `__make__` does not -- it never runs the check:

    >>> NonNegative.__make__(v=jnp.asarray([-1.0]))
    NonNegative(v=f32[1])

    Otherwise the result is indistinguishable from a constructed one, down to
    the pytree structure, so it composes with `jax.jit` and friends identically:

    >>> import jax.tree as jt
    >>> ok = jnp.asarray([1.0])
    >>> jt.structure(NonNegative.__make__(v=ok)) == jt.structure(NonNegative(ok))
    True

    Omitted fields take their defaults, and a name that is not a field is an
    error rather than a stray attribute:

    >>> NonNegative.__make__(v=ok).name
    'x'

    >>> NonNegative.__make__(v=ok, name="y")
    NonNegative(v=f32[1], name='y')

    >>> try: NonNegative.__make__(v=ok, unit="m")
    ... except TypeError as e: print(e)
    NonNegative has no field(s) ['unit']

    A base class can wrap `__make__` in a friendlier positional signature under
    a name of its own. Calling the dunder is not what ``PLW3201`` flags, so this
    is clean under that rule where redefining `__make__` would not be:

    >>> class Val(NonNegative):
    ...     @classmethod
    ...     def _mk(cls, v, /, name="v"):
    ...         return cls.__make__(v=v, name=name)

    >>> Val._mk(jnp.asarray([-1.0]), name="z")
    Val(v=f32[1], name='z')

    To keep the keyword signature and only shorten the name, re-bind the
    classmethod *object* -- ``cls`` then follows the class it is called on:

    >>> class Short(NonNegative):
    ...     _mk = SupportsUncheckedMake.__dict__["__make__"]

    >>> class Shorter(Short):
    ...     pass

    >>> type(Shorter._mk(v=ok)).__name__
    'Shorter'

    Two spellings that look right and are not. ``_mk = __make__`` in a class
    body raises `NameError`: the inherited name is not in scope there. And
    ``_mk = SupportsUncheckedMake.__make__`` binds to the mixin itself, so it
    tries to build a `SupportsUncheckedMake` rather than your class.

    """

    @classmethod
    def __make__(cls, **fields: Any) -> Self:
        """Return an instance holding *fields*, without running ``__init__``.

        Parameters
        ----------
        **fields
            One per dataclass field, by name. Fields with a default (or a
            default factory) may be omitted; anything else must be given.
            Values are stored as passed -- field converters do not run, so a
            value that a converter would have normalised must arrive already
            normalised.

        Returns
        -------
        Self
            An instance of ``cls``. No invariant of ``cls`` has been checked.

        Raises
        ------
        TypeError
            If ``cls`` is abstract or not a dataclass, if a name is not a
            field, or if a field without a default is missing.

        """
        spec = _field_spec(cls)
        if fields.keys() != spec.names:
            fields = _complete(cls, fields)  # already in declaration order
        else:
            # Order it, so `vars()` matches a constructed instance rather than
            # however the caller happened to write the keywords.
            fields = {name: fields[name] for name in spec.order}

        self = object.__new__(cls)
        # Install the dict rather than `self.__dict__.update(fields)`: reading
        # `self.__dict__` materialises the instance dict (CPython keeps the
        # values in a hidden array until something asks), and then copies into
        # it. Installing the one we just built skips both, and skips more of
        # them the more fields there are -- 1.8x on two fields, 2.4x on six.
        # Safe because that dict is ours: `**fields` is fresh per call, and
        # `_complete` returns a new dict.
        object.__setattr__(self, "__dict__", fields)
        return self
