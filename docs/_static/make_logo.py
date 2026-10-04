# /// script
# requires-python = ">=3.11"
# dependencies = ["resvg-py"]
# ///
"""Draw the quax-blocks logo: quax's duck, wearing a stack of blocks.

quax-blocks' duck carries a white badge on its body with three blue blocks
stacked into a pyramid, for the blocks quax classes are built from.
The duck is quax's logo, Material Design Icons' "duck", drawn from its own path,
so it is the same duck quax's docs show. The shapes are vector, so the logo is
written as an SVG, sharp at any size; for a bitmap, name a .png and give its
size::

    uv run docs/_static/make_logo.py                     # favicon.svg
    uv run docs/_static/make_logo.py --size 2048 big.png
"""

import argparse
from pathlib import Path

# Material Design Icons' "duck", by Pictogrammers, under the Apache License 2.0
# (https://pictogrammers.com/library/mdi/), in its 24-unit grid. It is drawn at
# twice that, in the 48-unit grid the badge below is placed in.
DUCK = (
    "M8.5,5A1.5,1.5 0 0,0 7,6.5A1.5,1.5 0 0,0 8.5,8"
    "A1.5,1.5 0 0,0 10,6.5A1.5,1.5 0 0,0 8.5,5M10,2A5,5 0 0,1 15,7"
    "C15,8.7 14.15,10.2 12.86,11.1C14.44,11.25 16.22,11.61 18,12.5"
    "C21,14 22,12 22,12C22,12 21,21 15,21H9C9,21 4,21 4,16"
    "C4,13 7,12 6,10C2,10 2,6.5 2,6.5C3,7 4.24,7 5,6.65"
    "C5.19,4.05 7.36,2 10,2Z"
)
BLACK, WHITE, BLUE = "#000000", "#ffffff", "#1473f0"
BADGE = ((19.6, 31.2), 8.4)  # the white disc on the duck's body: centre, radius
BLOCK, SPACE = 4.4, 0.7  # a block's side, and the space between blocks
ROUNDING = 0.7  # a block's corner radius

SVG = """\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" width="512" height="512">
  <path d="{duck}" transform="scale(2)" fill="{black}"/>
  <circle cx="{bx:g}" cy="{by:g}" r="{br:g}" fill="{white}"/>
{blocks}
</svg>
"""


def svg() -> str:
    """Return the logo as SVG text."""
    (bx, by), br = BADGE
    step = BLOCK + SPACE
    # Two blocks side by side, and one on top, centred on the badge.
    centres = ((bx - step / 2, by + step / 2), (bx + step / 2, by + step / 2),
               (bx, by - step / 2))  # fmt: skip
    blocks = "\n".join(
        f'  <rect x="{x - BLOCK / 2:g}" y="{y - BLOCK / 2:g}" width="{BLOCK:g}"'
        f' height="{BLOCK:g}" rx="{ROUNDING:g}" fill="{BLUE}"/>'
        for x, y in centres
    )
    return SVG.format(
        duck=DUCK, black=BLACK, white=WHITE, bx=bx, by=by, br=br, blocks=blocks
    )


def main() -> None:
    """Parse the command line and save the logo."""
    parser = argparse.ArgumentParser(
        description="Draw the quax-blocks logo: quax's duck, wearing blocks."
    )
    parser.add_argument(
        "out",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("favicon.svg"),
        help="output file, SVG or PNG by its extension (default: favicon.svg)",
    )
    parser.add_argument(
        "--size", type=int, default=512, help="pixels per side, for a PNG"
    )
    args = parser.parse_args()

    if args.out.suffix == ".svg":
        args.out.write_text(svg())
    else:
        import resvg_py  # noqa: PLC0415  # only a PNG needs a renderer

        png = resvg_py.svg_to_bytes(svg_string=svg(), width=args.size)
        args.out.write_bytes(bytes(png))


if __name__ == "__main__":
    main()
