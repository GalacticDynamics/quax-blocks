#!/usr/bin/env -S uv run --script  # noqa: EXE003
# /// script
#    dependencies = ["nox", "nox_uv"]
# ///
"""Nox setup."""

import shutil
from pathlib import Path

import nox
from nox_uv import session

nox.needs_version = ">=2024.3.2"
nox.options.default_venv_backend = "uv"

DIR = Path(__file__).parent.resolve()

# =============================================================================
# Comprehensive sessions


@session(uv_groups=["lint", "test", "docs"], reuse_venv=True, default=True)
def all(s: nox.Session, /) -> None:
    """Run all default sessions."""
    lint(s)
    test(s)
    docs(s)


# =============================================================================
# Linting


@session(uv_groups=["lint", "build"], reuse_venv=True)
def lint(s: nox.Session, /) -> None:
    """Run the linter."""
    precommit(s)  # reuse pre-commit session
    pylint(s)  # reuse pylint session


@session(uv_groups=["lint", "build"], reuse_venv=True)
def precommit(s: nox.Session, /) -> None:
    """Run pre-commit."""
    s.run("prek", "run", "--all-files", *s.posargs)


@session(uv_groups=["lint", "build"], reuse_venv=True)
def pylint(s: nox.Session, /) -> None:
    """Run PyLint."""
    s.run("pylint", "quax-blocks", *s.posargs)


# =============================================================================
# Testing


@session(uv_groups=["test"], reuse_venv=True)
def test(s: nox.Session, /) -> None:
    """Run the unit and regular tests."""
    s.notify("pytest", posargs=s.posargs)


# =============================================================================
# Documentation


@session(uv_groups=["docs"], reuse_venv=True)
def docs(s: nox.Session, /) -> None:
    """Build the docs. Pass "--serve" to serve. Pass "-b linkcheck" to check links."""
    s.run("zensical", "build", *s.posargs)


# =============================================================================
# Build


@session(uv_groups=["build"])
def build(s: nox.Session, /) -> None:
    """Build an SDist and wheel."""
    build_path = DIR.joinpath("build")
    if build_path.exists():
        shutil.rmtree(build_path)

    s.run("python", "-m", "build")


# =============================================================================

if __name__ == "__main__":
    nox.main()
