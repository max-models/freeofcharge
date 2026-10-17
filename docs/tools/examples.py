"""Build Full examples pages from Python scripts and their Matplotlib figures."""

from __future__ import annotations

import ast
import json
import os
import re
import runpy
import shutil
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1]

# The MDX escaping of the theme's notebook tool (installed with the npm packages)
sys.path.insert(
    0, str(DOCS / "node_modules" / "@max-models" / "starlight-theme" / "tools")
)
from notebooks import sanitize_markdown  # noqa: E402

ROOT = DOCS.parent
SOURCES = ROOT / "examples"
CONTENT = DOCS / "src" / "content" / "docs" / "full-examples"
PUBLIC = DOCS / "public" / "full-examples"


def source_fence(code: str) -> str:
    """Wrap the complete source without truncating long scripts."""
    ticks = "```"
    while ticks in code:
        ticks += "`"
    return f"{ticks}python\n{code.rstrip()}\n{ticks}"


def build_example(source: Path, order: int) -> tuple[str, str]:
    """Run one script and publish its docstring, source, and open figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    slug = re.sub(r"[^a-z0-9]+", "-", source.stem.lower()).strip("-")
    code = source.read_text()
    docstring = ast.get_docstring(ast.parse(code))
    if not docstring:
        raise ValueError(f"{source}: a module docstring is required")
    title, _, description = docstring.partition("\n")
    title = title.strip()
    description = description.strip()
    summary = description.split("\n\n", 1)[0].replace("\n", " ") or title
    output = PUBLIC / slug
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True)
    plt.close("all")
    previous = Path.cwd()
    try:
        os.chdir(source.parent)
        runpy.run_path(str(source), run_name="__main__")
    finally:
        os.chdir(previous)
    plots = []
    for index, number in enumerate(plt.get_fignums(), start=1):
        filename = f"plot-{index}.png"
        plt.figure(number).savefig(output / filename, dpi=150, bbox_inches="tight")
        plots.append(
            f"![Plot {index} from {title}](/template-python/full-examples/{slug}/{filename})"
        )
    plt.close("all")
    page = [
        "---",
        f"title: {json.dumps(title)}",
        f"description: {json.dumps(summary)}",
        "sidebar:",
        f"  order: {order}",
        "---",
        "",
        sanitize_markdown(description) if description else "",
        "",
        "## Python source",
        "",
        source_fence(code),
        "",
        "## Plots",
        "",
        "\n\n".join(plots) if plots else "This example does not produce plots.",
        "",
    ]
    CONTENT.mkdir(parents=True, exist_ok=True)
    (CONTENT / f"{slug}.mdx").write_text("\n".join(page))
    return title, summary


def main() -> None:
    """Regenerate all Full examples pages and assets."""
    shutil.rmtree(CONTENT, ignore_errors=True)
    shutil.rmtree(PUBLIC, ignore_errors=True)
    examples = sorted(SOURCES.glob("*.py"))
    listing = []
    for order, source in enumerate(examples, start=1):
        title, summary = build_example(source, order)
        slug = re.sub(r"[^a-z0-9]+", "-", source.stem.lower()).strip("-")
        listing.append(
            f"- [{title}](/template-python/full-examples/{slug}/): {summary}"
        )
        print(f"[examples] {source} -> {CONTENT / slug}.mdx")
    CONTENT.mkdir(parents=True, exist_ok=True)
    (CONTENT / "index.md").write_text(
        "---\ntitle: Full examples\ndescription: Complete Python scripts and their plots.\n"
        "sidebar:\n  order: 0\n---\n\n"
        "Complete Python examples are listed below. Each page shows the script and its plots.\n\n"
        + "\n".join(listing)
        + "\n"
    )


if __name__ == "__main__":
    main()
