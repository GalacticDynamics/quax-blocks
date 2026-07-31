"""Test `quax_blocks.SupportsUncheckedMake`."""

import abc
import dataclasses
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree as jt
import pytest
from jaxtyping import Array

from quax_blocks import AbstractVal, SupportsUncheckedMake


class Checked(AbstractVal, SupportsUncheckedMake):
    """A value whose constructor enforces an invariant."""

    v: Array
    name: str = eqx.field(default="x", static=True)

    def __check_init__(self) -> None:
        """Reject a negative value."""
        if jnp.any(self.v < 0):
            msg = "must be non-negative"
            raise ValueError(msg)


class Converted(AbstractVal, SupportsUncheckedMake):
    """A value with a field converter."""

    v: Array = eqx.field(converter=jnp.asarray)


class Bundle(AbstractVal, SupportsUncheckedMake):
    """A value whose payload is a container, not a single leaf."""

    v: Array
    parts: dict[str, Array] = dataclasses.field(default_factory=dict)


class Required(AbstractVal, SupportsUncheckedMake):
    """A value with two fields, neither defaulted."""

    v: Array
    other: Array


OK = jnp.asarray([1.0, 2.0])


class TestIndistinguishableFromTheConstructor:
    """Whatever `__make__` returns must be a first-class instance."""

    def test_type_and_fields(self) -> None:
        """Same type, same field set, same values."""
        made, built = Checked.__make__(v=OK), Checked(OK)
        assert type(made) is type(built) is Checked
        assert made.__dict__.keys() == built.__dict__.keys()
        assert jnp.array_equal(made.v, built.v)
        assert made.name == built.name

    def test_repr(self) -> None:
        """A made instance prints as a built one."""
        assert repr(Checked.__make__(v=OK)) == repr(Checked(OK))

    def test_pytree_structure(self) -> None:
        """Equal treedefs, so it composes with jit/vmap/tree_map identically."""
        assert jt.structure(Checked.__make__(v=OK)) == jt.structure(Checked(OK))

    def test_roundtrips_through_jit(self) -> None:
        """It flattens and unflattens like any other module."""
        out = jax.jit(lambda x: x)(Checked.__make__(v=OK))
        assert type(out) is Checked
        assert jnp.array_equal(out.v, OK)

    def test_static_field_stays_static(self) -> None:
        """A static field belongs to the treedef, not the leaves."""
        leaves, treedef = jt.flatten(Checked.__make__(v=OK, name="y"))
        assert len(leaves) == 1
        assert treedef != jt.structure(Checked.__make__(v=OK, name="z"))


class TestSkipsInit:
    """The point of the mixin, and its whole danger."""

    def test_skips_check_init(self) -> None:
        """The value the constructor rejects is the one `__make__` returns."""
        bad = jnp.asarray([-1.0])
        with pytest.raises(ValueError, match="must be non-negative"):
            Checked(bad)
        assert jnp.array_equal(Checked.__make__(v=bad).v, bad)

    def test_skips_the_converter(self) -> None:
        """Documented: values are stored as passed, so they must arrive ready."""
        assert isinstance(Converted([1.0, 2.0]).v, jax.Array)
        assert Converted.__make__(v=[1.0, 2.0]).v == [1.0, 2.0]  # the list itself


class TestFields:
    """Fields are passed by name; defaults fill in; mistakes are loud."""

    def test_default_is_filled(self) -> None:
        """An omitted field takes its default."""
        assert Checked.__make__(v=OK).name == "x"

    def test_default_factory_is_called_each_time(self) -> None:
        """A factory default is not shared between instances."""
        first, second = Bundle.__make__(v=OK), Bundle.__make__(v=OK)
        assert first.parts == {}
        assert first.parts is not second.parts

    def test_given_value_wins_over_default(self) -> None:
        """An explicit value overrides the default."""
        assert Checked.__make__(v=OK, name="y").name == "y"

    def test_unknown_field_raises(self) -> None:
        """A name that is not a field is an error, not a stray attribute."""
        with pytest.raises(TypeError, match=r"has no field\(s\) \['unit'\]"):
            Checked.__make__(v=OK, unit="m")

    def test_missing_field_without_default_raises(self) -> None:
        """A field with no default must be given."""
        with pytest.raises(TypeError, match=r"field\(s\) \['other'\] have no default"):
            Required.__make__(v=OK)

    def test_every_missing_field_is_named_in_declaration_order(self) -> None:
        """Not just whichever a set happened to yield first.

        Set iteration depends on string hashing, which is randomised per
        process, so reporting one arbitrary name would vary between runs.
        """

        class Three(AbstractVal, SupportsUncheckedMake):
            """Two of these have no default."""

            v: Array
            zzz: Array
            aaa: Array

        with pytest.raises(
            TypeError, match=r"field\(s\) \['zzz', 'aaa'\] have no default"
        ):
            Three.__make__(v=OK)

    def test_factories_run_in_declaration_order(self) -> None:
        """The order the ordinary constructor would run them in."""
        calls: list[str] = []

        def record(name: str) -> Callable[[], Array]:
            def factory() -> Array:
                calls.append(name)
                return OK

            return factory

        class Ordered(AbstractVal, SupportsUncheckedMake):
            """Factory names deliberately unsorted and unlike field order."""

            v: Array
            mmm: Array = eqx.field(default_factory=record("mmm"))
            aaa: Array = eqx.field(default_factory=record("aaa"))
            zzz: Array = eqx.field(default_factory=record("zzz"))

        Ordered.__make__(v=OK)
        assert calls == ["mmm", "aaa", "zzz"]

    def test_fields_land_in_declaration_order(self) -> None:
        """`vars()` matches a constructed instance, not the caller's kwargs."""
        made = Checked.__make__(name="y", v=OK)  # kwargs reversed
        assert list(vars(made)) == list(vars(Checked(OK, "y")))

    def test_abstract_method_raises(self) -> None:
        """An `abc.abstractmethod` still blocks instantiation."""

        class Abstract(AbstractVal, SupportsUncheckedMake):
            v: Array

            @abc.abstractmethod
            def thing(self) -> None: ...

        with pytest.raises(TypeError, match=r"abstract in \['thing'\]"):
            Abstract.__make__(v=OK)

    def test_abstract_var_raises(self) -> None:
        """`object.__new__` sidesteps equinox's own abstractness check."""

        class MissingVar(AbstractVal, SupportsUncheckedMake):
            """Never provides `AbstractVal.v`."""

        with pytest.raises(TypeError, match=r"abstract in \['v'\]"):
            MissingVar.__make__(v=OK)


class TestContainerPayload:
    """A dynamic field need not be a single leaf.

    This is what a treedef cached on ``(cls, static fields)`` cannot do: the
    structure here depends on the *keys* of ``parts``, which are payload, not
    class metadata.
    """

    @pytest.mark.parametrize(
        ("parts", "n_leaves"),
        [({}, 1), ({"a": OK}, 2), ({"a": OK, "b": OK}, 3)],
    )
    def test_structure_follows_the_payload(
        self, parts: dict[str, Array], n_leaves: int
    ) -> None:
        """Leaf count tracks the container's contents."""
        assert jt.structure(Bundle.__make__(v=OK, parts=parts)).num_leaves == n_leaves

    def test_matches_the_constructor(self) -> None:
        """Same structure as the ordinary constructor gives."""
        parts = {"a": OK, "b": OK}
        made, built = Bundle.__make__(v=OK, parts=parts), Bundle(OK, parts)
        assert jt.structure(made) == jt.structure(built)


class TestSubclassWrapper:
    """A base may wrap `__make__` under a name of its own.

    Documented as the way to stay clear of ruff's ``PLW3201``, which fires
    where a custom dunder is *defined*, not where it is called.
    """

    def test_wrapper_under_a_plain_name(self) -> None:
        """The recommended recipe: a normal classmethod calling the dunder."""

        class Sugar(Checked):
            @classmethod
            def _mk(cls, v: Array, /, name: str = "sugar") -> "Sugar":
                return cls.__make__(v=v, name=name)

        out = Sugar._mk(jnp.asarray([-1.0]))  # noqa: SLF001
        assert type(out) is Sugar
        assert out.name == "sugar"

    def test_wrapper_binds_to_the_calling_subclass(self) -> None:
        """A subclass of the wrapper builds itself, not its parent."""

        class Sugar(Checked):
            @classmethod
            def _mk(cls, v: Array, /) -> "Sugar":
                return cls.__make__(v=v)

        class Sweeter(Sugar):
            pass

        assert type(Sweeter._mk(OK)) is Sweeter  # noqa: SLF001

    def test_rebinding_the_classmethod_object(self) -> None:
        """The other recipe: alias the descriptor, keeping the keyword API."""

        class Short(Checked):
            _mk = SupportsUncheckedMake.__dict__["__make__"]

        class Shorter(Short):
            pass

        assert type(Short._mk(v=OK)) is Short  # noqa: SLF001
        assert type(Shorter._mk(v=OK)) is Shorter  # noqa: SLF001

    def test_super_call_still_works(self) -> None:
        """Overriding the dunder itself remains legal, just PLW3201-flagged."""

        class Sugar(Checked):
            @classmethod
            def __make__(cls, v: Array, /, name: str = "sugar") -> "Sugar":
                return super().__make__(v=v, name=name)

        assert Sugar.__make__(OK).name == "sugar"

    def test_the_two_traps(self) -> None:
        """Spellings that look right: one is a NameError, one binds the mixin."""
        with pytest.raises(NameError):
            exec(  # noqa: S102
                "class Bad(Checked):\n    _mk = __make__\n",
                {"Checked": Checked},
            )

        class BoundToTheMixin(Checked):
            _mk = SupportsUncheckedMake.__make__

        with pytest.raises(TypeError, match="must be called with a dataclass"):
            BoundToTheMixin._mk(v=OK)  # noqa: SLF001

    def test_subclass_adding_a_field(self) -> None:
        """The keyword API cannot be broken by a subclass adding a field."""

        class Extra(Checked):
            extra: Array = eqx.field(default_factory=lambda: OK)

        assert jnp.array_equal(Extra.__make__(v=OK).extra, OK)
        assert jt.structure(Extra.__make__(v=OK)) == jt.structure(Extra(OK))
