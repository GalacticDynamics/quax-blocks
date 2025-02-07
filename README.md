<h1 align='center'> quax-blocks </h1>
<h3 align="center">Building blocks for <code>Quax</code> classes</h3>

<p align="center">
    <a href="https://pypi.org/project/quax-blocks/"> <img alt="PyPI: quax-blocks" src="https://img.shields.io/pypi/v/quax-blocks?style=flat" /> </a>
    <a href="https://pypi.org/project/quax-blocks/"> <img alt="PyPI versions: quax-blocks" src="https://img.shields.io/pypi/pyversions/quax-blocks" /> </a>
    <a href="https://quax-blocks.readthedocs.io/en/"> <img alt="ReadTheDocs" src="https://img.shields.io/badge/read_docs-here-orange" /> </a>
    <a href="https://pypi.org/project/quax-blocks/"> <img alt="quax-blocks license" src="https://img.shields.io/github/license/GalacticDynamics/quax-blocks" /> </a>
</p>
<p align="center">
    <a href="https://github.com/GalacticDynamics/quax-blocks/actions/workflows/ci.yml"> <img alt="CI status" src="https://github.com/GalacticDynamics/quax-blocks/actions/workflows/ci.yml/badge.svg?branch=main" /> </a>
    <a href="https://quax-blocks.readthedocs.io/en/"> <img alt="ReadTheDocs" src="https://readthedocs.org/projects/quax-blocks/badge/?version=latest" /> </a>
    <a href="https://codecov.io/gh/GalacticDynamics/quax-blocks"> <img alt="codecov" src="https://codecov.io/gh/GalacticDynamics/quax-blocks/graph/badge.svg" /> </a>
    <a href="https://scientific-python.org/specs/spec-0000/"> <img alt="ruff" src="https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038" /> </a>
    <a href="https://docs.astral.sh/ruff/"> <img alt="ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json" /> </a>
    <a href="https://pre-commit.com"> <img alt="pre-commit" src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit" /> </a>
</p>

---

[`quax`](https://docs.kidger.site/quax/) brings enables JAX to work with custom
array-ish objects. This library provides the building blocks, like comparison
operators, for building `quax`-compatible classes.

## Installation

[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

```bash
pip install quax-blocks
```

## Documentation

[![Read The Docs](https://img.shields.io/badge/read_docs-here-orange)](https://unxt.readthedocs.io/en/)

### Quick Start

```python
import quax
import quax_blocks


class MyClass(quax_blocks.LaxAdd, quax.ArrayValue):
    pass
```

## Development

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]
[![codecov][codecov-badge]][codecov-link]
[![SPEC 0 — Minimum Supported Dependencies][spec0-badge]][spec0-link]
[![pre-commit][pre-commit-badge]][pre-commit-link]
[![ruff][ruff-badge]][ruff-link]

We welcome contributions!

## Citation

[![DOI][zenodo-badge]][zenodo-link]

If you found this library to be useful and want to support the development and
maintenance of lower-level utility libraries for the scientific community,
please consider citing this work.

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/GalacticDynamics/quax-blocks/workflows/CI/badge.svg
[actions-link]:             https://github.com/GalacticDynamics/quax-blocks/actions
[codecov-badge]:            https://codecov.io/gh/GalacticDynamics/quax-blocks/graph/badge.svg?token=9G19ONVD3U
[codecov-link]:             https://codecov.io/gh/GalacticDynamics/quax-blocks
[pre-commit-badge]:         https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit
[pre-commit-link]:          https://pre-commit.com
[pypi-link]:                https://pypi.org/project/quax-blocks/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/quax-blocks
[pypi-version]:             https://img.shields.io/pypi/v/quax-blocks
[rtd-badge]:                https://readthedocs.org/projects/quax-blocks/badge/?version=latest
[rtd-link]:                 https://quax-blocks.readthedocs.io/en/latest/?badge=latest
[ruff-badge]:               https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json
[ruff-link]:                https://docs.astral.sh/ruff/
[spec0-badge]:              https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038
[spec0-link]:               https://scientific-python.org/specs/spec-0000/
[zenodo-badge]:             https://zenodo.org/badge/732262318.svg
[zenodo-link]:              https://zenodo.org/doi/10.5281/zenodo.10850521

<!-- prettier-ignore-end -->
