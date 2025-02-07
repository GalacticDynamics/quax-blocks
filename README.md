<h1 align='center'> quax-blocks </h1>
<h3 align="center">Building blocks for <code>Quax</code> classes</h3>

<p align="center">
    <a href="https://pypi.org/project/quax-blocks/"> <img alt="PyPI: quax-blocks" src="https://img.shields.io/pypi/v/quax-blocks?style=flat" /> </a>
    <a href="https://pypi.org/project/quax-blocks/"> <img alt="PyPI versions: quax-blocks" src="https://img.shields.io/pypi/pyversions/quax-blocks" /> </a>
    <a href="https://pypi.org/project/quax-blocks/"> <img alt="quax-blocks license" src="https://img.shields.io/github/license/GalacticDynamics/quax-blocks" /> </a>
</p>
<p align="center">
    <a href="https://github.com/GalacticDynamics/quax-blocks/actions/workflows/ci.yml"> <img alt="CI status" src="https://github.com/GalacticDynamics/quax-blocks/actions/workflows/ci.yml/badge.svg?branch=main" /> </a>
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

### Rich Comparison Operators

| Comparison Operator | Lax Operator         | NumPy Operator         |
| ------------------- | -------------------- | ---------------------- |
| All Comparisons     | `LaxComparisonMixin` | `NumpyComparisonMixin` |
| `__eq__`            | `LaxEqMixin`         | `NumpyEqMixin`         |
| `__ne__`            | `LaxNeMixin`         | `NumpyNeMixin`         |
| `__lt__`            | `LaxLtMixin`         | `NumpyLtMixin`         |
| `__le__`            | `LaxLeMixin`         | `NumpyLeMixin`         |
| `__gt__`            | `LaxGtMixin`         | `NumpyGtMixin`         |
| `__ge__`            | `LaxGeMixin`         | `NumpyGeMixin`         |

### Binary Operators

| Binary Operator        | Lax Operator        | NumPy Operator        |
| ---------------------- | ------------------- | --------------------- |
| All Binary Operations  | `LaxBinaryOpsMixin` | `NumpyBinaryOpsMixin` |
| All Float Operations   | `LaxMathMixin`      | `NumpyMathMixin`      |
| `__add__`              | `LaxAddMixin`       | `NumpyAddMixin`       |
| `__radd__`             | `LaxRAddMixin`      | `NumpyRAddMixin`      |
| `__sub__`              | `LaxSubMixin`       | `NumpySubMixin`       |
| `__rsub__`             | `LaxRSubMixin`      | `NumpyRSubMixin`      |
| `__mul__`              | `LaxMulMixin`       | `NumpyMulMixin`       |
| `__rmul__`             | `LaxRMulMixin`      | `NumpyRMulMixin`      |
| `__matmul__`           | `LaxMatMulMixin`    | `NumpyMatMulMixin`    |
| `__rmatmul__`          | `LaxRMatMulMixin`   | `NumpyRMatMulMixin`   |
| `__truediv__`          | `LaxTrueDivMixin`   | `NumpyTrueDivMixin`   |
| `__rtruediv__`         | `LaxRTrueDivMixin`  | `NumpyRTrueDivMixin`  |
| `__floordiv__`         | `LaxFloorDivMixin`  | `NumpyFloorDivMixin`  |
| `__rfloordiv__`        | `LaxRFloorDivMixin` | `NumpyRFloorDivMixin` |
| `__mod__`              | `LaxModMixin`       | `NumpyModMixin`       |
| `__rmod__`             | `LaxRModMixin`      | `NumpyRModMixin`      |
| `__divmod__`           | Not Implemented     | `NumpyDivModMixin`    |
| `__rdivmod__`          | Not Implemented     | `NumpyRDivModMixin`   |
| `__pow__`              | `LaxPowMixin`       | `NumpyPowMixin`       |
| `__rpow__`             | `LaxRPowMixin`      | `NumpyRPowMixin`      |
| All Bitwise Operations | `LaxBitwiseMixin`   | `NumpyBitwiseMixin`   |
| `__lshift__`           | `LaxLShiftMixin`    | `NumpyLShiftMixin`    |
| `__rlshift__`          | `LaxRLShiftMixin`   | `NumpyRLShiftMixin`   |
| `__rshift__`           | `LaxRShiftMixin`    | `NumpyRShiftMixin`    |
| `__rrshift__`          | `LaxRRShiftMixin`   | `NumpyRRShiftMixin`   |
| `__and__`              | `LaxAndMixin`       | `NumpyAndMixin`       |
| `__rand__`             | `LaxRAndMixin`      | `NumpyRAndMixin`      |
| `__xor__`              | `LaxXorMixin`       | `NumpyXorMixin`       |
| `__rxor__`             | `LaxRXorMixin`      | `NumpyRXorMixin`      |
| `__or__`               | `LaxOrMixin`        | `NumpyOrMixin`        |
| `__ror__`              | `LaxROrMixin`       | `NumpyROrMixin`       |

### Unary Operators

| Unary Operator       | Lax Operator    | NumPy Operator     |
| -------------------- | --------------- | ------------------ |
| All Unary Operations | `LaxUnaryMixin` | `NumpyUnaryMixin`  |
| `__pos__`            | `LaxPosMixin`   | `NumpyPosMixin`    |
| `__neg__`            | `LaxNegMixin`   | `NumpyNegMixin`    |
| `__abs__`            | `LaxAbsMixin`   | `NumpyAbsMixin`    |
| `__invert__`         | Not Implemented | `NumpyInvertMixin` |

### Rounding Operators

| Unary Operator | Lax Operator    | NumPy Operator    |
| -------------- | --------------- | ----------------- |
| `__round__`    | `LaxRoundMixin` | `NumpyRoundMixin` |
| `__trunc__`    | `LaxTruncMixin` | `NumpyTruncMixin` |
| `__floor__`    | `LaxFloorMixin` | `NumpyFloorMixin` |
| `__ceil__`     | `LaxCeilMixin`  | `NumpyCeilMixin`  |

### Containers

| Container Operator | Lax Operator         | NumPy Operator         |
| ------------------ | -------------------- | ---------------------- |
| `__len__`          | `LaxLenMixin`        | `NumpyLenMixin`        |
| `__length_hint__`  | `LaxLengthHintMixin` | `NumpyLengthHintMixin` |

### Copy Operators

| Copy Operator  | NumPy Operator       |
| -------------- | -------------------- |
| `__copy__`     | `NumpyCopyMixin`     |
| `__deepcopy__` | `NumpyDeepCopyMixin` |

## Development

[![Actions Status][actions-badge]][actions-link]
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
[ruff-badge]:               https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json
[ruff-link]:                https://docs.astral.sh/ruff/
[spec0-badge]:              https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038
[spec0-link]:               https://scientific-python.org/specs/spec-0000/
[zenodo-badge]:             https://zenodo.org/badge/732262318.svg
[zenodo-link]:              https://zenodo.org/doi/10.5281/zenodo.10850521

<!-- prettier-ignore-end -->
