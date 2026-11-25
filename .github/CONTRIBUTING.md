See the [Scientific Python Developer Guide][spc-dev-intro] for a detailed
description of best practices for developing scientific packages.

[spc-dev-intro]: https://learn.scientific-python.org/development/

# Quick development

The fastest way to start with development is to use nox. If you don't have nox,
you can use `pipx run nox` to run it without installing, or `pipx install nox`.
If you don't have pipx (pip for applications), then you can install with
`pip install pipx` (the only case were installing an application with regular
pip is reasonable). If you use macOS, then pipx and nox are both in brew, use
`brew install pipx nox`.

To use, run `nox`. This will lint and test using every installed version of
Python on your system, skipping ones that are not installed. You can also run
specific jobs:

```console
$ nox -s lint  # Lint only
$ nox -s tests  # Python tests
$ nox -s build  # Make an SDist and wheel
```

Nox handles everything for you, including setting up an temporary virtual
environment for each run.

# Setting up a development environment manually

You can set up a development environment by running:

```bash
python3 -m venv .venv
source ./.venv/bin/activate
pip install -v -e .[dev]
```

If you have the
[Python Launcher for Unix](https://github.com/brettcannon/python-launcher), you
can instead do:

```bash
py -m venv .venv
py -m install -v -e .[dev]
```

# Post setup

You should prepare pre-commit, which will help you by checking that commits pass
required checks:

```bash
pip install pre-commit # or brew install pre-commit on macOS
pre-commit install # Will install a pre-commit hook into the git repo
```

You can also/alternatively run `pre-commit run` (changes only) or
`pre-commit run --all-files` to check even without installing the hook.

# The code graph

`graphify-out/graph.json` is a
[graphify](https://github.com/Graphify-Labs/graphify) map of this repository:
its symbols and the relationships between them. AI assistants query it instead
of grepping, and you can too. Extraction is local tree-sitter parsing — no model
call, and nothing leaves your machine — and every edge is marked `EXTRACTED`
(explicit in the source) or `INFERRED` (resolved by graphify), so you can tell
what was read from what was deduced.

The graph is committed, and the commit checks rebuild it. Install the pinned
release and register the merge driver once per clone, so two branches that both
changed the graph merge cleanly:

```bash
uv tool install "graphifyy==0.9.119"
PY="$(sed -n '1s/^#!//p' "$(command -v graphify)")"
git config merge.graphify.name "graphify graph.json union merge"
git config merge.graphify.driver "\"$PY\" -m graphify merge-driver %O %A %B"
```

The merge driver cannot be committed — git refuses to take a merge-driver
command from a tracked file — so every clone runs those three lines once.

Then query it:

```bash
graphify query "how are comparison mixins dispatched"  # a scoped subgraph
graphify explain "LaxComparisonMixin"                  # one symbol and its neighbours
graphify path "LaxEqMixin" "DISPATCH_ERRORS"           # how two things connect
open graphify-out/graph.html                           # the clickable graph
```

Day to day:

- The rebuild runs with your other commit checks. Like a formatter, it rebuilds
  and stops the commit; `git add graphify-out/graph.json` and commit again.
- Commit without rebuilding: `SKIP=graphify-update git commit ...`.
- If a rebuild is ever refused because the graph would shrink — you deleted code
  — use `GRAPHIFY_FORCE=1 git commit ...`. It usually just shrinks and says
  nothing.
- If the graph ever conflicts in a merge, rebuild rather than edit it:
  `rm graphify-out/graph.json && graphify update . --force && git add graphify-out/graph.json`.
- A changed `graph.json` in a pull request is generated output. It is one line,
  collapsed by GitHub, and is not reviewed.
- The rebuild may rewrite the graph the first time you commit in a working
  environment, and the weekly `Refresh code graph` job may rewrite it back. That
  is expected: graphify's output depends on which files are on disk, so a built
  checkout — one where `src/quax_blocks/_version.py` exists because you
  installed the package — and a fresh clone disagree about community numbering
  even though the symbols are identical. Stage it and commit.

# Testing

Use pytest to run the unit checks:

```bash
pytest
```

# Coverage

Use pytest-cov to generate coverage reports:

```bash
pytest --cov=quax-blocks
```

# Pre-commit

This project uses pre-commit for all style checking. While you can run it with
nox, this is such an important tool that it deserves to be installed on its own.
Install pre-commit and run:

```bash
pre-commit run -a
```

to check all files.
