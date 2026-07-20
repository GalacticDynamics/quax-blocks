"""Guard the `quaxed` function names the type checker no longer sees.

The mixin modules give `quaxed.lax` / `quaxed.numpy` a permissive type at
check time (see the note beside their imports), because quaxed's annotations
describe plain-JAX signatures and cannot express quax's dispatch. That trade
costs static validation of the *names* themselves, so a typo like
``qnp.subtractt`` would no longer be a type error.

This test recovers exactly that: every ``qnp.<name>`` / ``qlax.<name>``
referenced in the source must exist in the corresponding quaxed module.
"""

import re
from pathlib import Path

import pytest
import quaxed.lax as qlax
import quaxed.numpy as qnp

SRC = Path(__file__).resolve().parents[1] / "src" / "quax_blocks" / "_src"
_CALL = re.compile(r"\b(qnp|qlax)\.([A-Za-z_][A-Za-z0-9_]*)")
_MODULES = {"qnp": qnp, "qlax": qlax}


def _referenced_names() -> list[tuple[str, str, str]]:
    """Return every (file, alias, attribute) referenced across the source."""
    found = {
        (path.name, alias, attr)
        for path in sorted(SRC.glob("*.py"))
        for alias, attr in _CALL.findall(path.read_text())
    }
    return sorted(found)


def test_source_references_some_quaxed_names() -> None:
    """Sanity-check the scanner itself, so the test cannot silently pass."""
    refs = _referenced_names()
    assert len(refs) > 20, f"scanner found suspiciously few references: {refs}"


@pytest.mark.parametrize(("filename", "alias", "attr"), _referenced_names())
def test_quaxed_names_exist(filename: str, alias: str, attr: str) -> None:
    """Every quaxed attribute used in the source actually exists."""
    module = _MODULES[alias]
    assert hasattr(module, attr), (
        f"{filename} references `{alias}.{attr}`, which does not exist in "
        f"{module.__name__}. The mixin modules type quaxed permissively, so "
        f"this is not caught by the type checker."
    )
