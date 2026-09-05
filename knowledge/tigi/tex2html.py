"""Convert tigi_v1.1.tex to a self-contained HTML file (MathML-based).

The document is a plain article without tables/figures/bibliography
styles, so a small purpose-built converter is sufficient.  LaTeX math
is converted with latex2mathml, and the result is rendered to PDF with
a headless browser.
"""

from __future__ import annotations

import html
import re
import sys
from dataclasses import dataclass, field

from latex2mathml.converter import convert as math_to_mathml


SRC = "tigi_v1.1.tex"
OUT = "tigi_v1.1.html"


# --------------------------------------------------------------------------
# Blocks produced by the small line parser
# --------------------------------------------------------------------------


@dataclass
class TextBlock:
    text: str


@dataclass
class HeadingBlock:
    level: int  # 1..3 (LaTeX section/subsection/subsubsection)
    title: str


@dataclass
class RuninBlock:
    title: str  # LaTeX \paragraph


@dataclass
class ListBlock:
    kind: str  # 'ul' | 'ol'
    items: list  # list of Blocks


@dataclass
class BibBlock:
    items: list  # list of (key, Blocks)


@dataclass
class AbstractBlock:
    content: "Blocks"


class Blocks:
    def __init__(self):
        self.items: list = []
        self.buf: list[str] = []


# --------------------------------------------------------------------------
# Math extraction and text conversion
# --------------------------------------------------------------------------

_MACRO_RE = re.compile(r"\\(textbf|emph|textit|texttt)\{([^{}]*)\}")
_INLINE_MATH_RE = re.compile(
    r"(?<!\\)\$(?!\$)([^\n$]*?)(?<!\\)\$(?!\$)"
)


def _extract_math(raw: str) -> tuple[str, list[str]]:
    """Replace LaTeX math with placeholders; return (text, mathml list)."""

    maths: list[str] = []

    def display_repl(match: re.Match) -> str:
        tex = match.group(1).strip()
        idx = len(maths)
        try:
            tex = tex.replace(r"\arg\max", r"\mathrm{arg\,max}")
            tex = tex.replace(r"\text{arg}\min", r"\mathrm{arg\,min}")
            mml = math_to_mathml(tex, display="block")
        except Exception:
            mml = html.escape(tex)
        maths.append(f'<span class="eq">{mml}</span>')
        return f"\x00M{idx}\x00"

    def inline_repl(match: re.Match) -> str:
        tex = match.group(1).strip()
        idx = len(maths)
        try:
            tex = tex.replace(r"\arg\max", r"\mathrm{arg\,max}")
            tex = tex.replace(r"\text{arg}\min", r"\mathrm{arg\,min}")
            mml = math_to_mathml(tex, display="inline")
        except Exception:
            mml = html.escape(tex)
        maths.append(mml)
        return f"\x00M{idx}\x00"

    raw = re.sub(r"\\\[\s*(.*?)\s*\\\]", display_repl, raw, flags=re.S)
    raw = re.sub(r"\$\$(.*?)\$\$", display_repl, raw, flags=re.S)
    raw = _INLINE_MATH_RE.sub(inline_repl, raw)
    return raw, maths


def text_to_html(raw: str) -> str:
    """Convert ordinary LaTeX paragraph/heading text into HTML."""

    raw, maths = _extract_math(raw)

    # LaTeX escapes that may appear in ordinary prose.
    raw = raw.replace(r"\%", "%")
    raw = raw.replace(r"\&", "&")
    raw = raw.replace(r"\_", "_")
    raw = raw.replace(r"\#", "#")
    raw = raw.replace(r"\$", "$")
    raw = raw.replace(r"~", "\u00a0")

    # Font/emphasis macros (contents never nest in this document).
    tags = {
        "textbf": "strong",
        "emph": "em",
        "textit": "i",
        "texttt": "code",
    }
    while True:
        match = _MACRO_RE.search(raw)
        if match is None:
            break
        cmd, inner = match.group(1), match.group(2)
        tag = tags[cmd]
        raw = (
            raw[: match.start()]
            + f"\x01{tag}\x01"
            + text_to_html(inner)
            + f"\x01/{tag}\x01"
            + raw[match.end() :]
        )
    escaped = html.escape(raw, quote=False)

    for tag in ("strong", "em", "i", "code"):
        escaped = escaped.replace(f"\x01{tag}\x01", f"<{tag}>")
        escaped = escaped.replace(f"\x01/{tag}\x01", f"</{tag}>")

    for idx, mml in enumerate(maths):
        escaped = escaped.replace(f"\x00M{idx}\x00", mml)
    return escaped.strip()


def paragraph_to_html(raw: str) -> str:
    return f"<p>{text_to_html(raw)}</p>"


# --------------------------------------------------------------------------
# Line parser
# --------------------------------------------------------------------------

_SECTION_RE = re.compile(
    r"\\(section|subsection|subsubsection)\{(.*?)\}\s*$"
)
_PARAGRAPH_RE = re.compile(r"\\(paragraph)\{(.*?)\}\s*$")
_BEGIN_RE = re.compile(
    r"\\begin\{(abstract|itemize|enumerate|thebibliography)\}(?:\{[^{}]*\})?\s*$"
)
_END_RE = re.compile(r"\\end\{(abstract|itemize|enumerate|thebibliography)\}\s*$")
_ITEM_RE = re.compile(r"\\item\s?(.*)$", re.S)
_BIBITEM_RE = re.compile(r"\\bibitem\{([^}]*)\}\s?(.*)$", re.S)


def parse_body(body: str) -> Blocks:
    root = Blocks()
    active = root
    env_stack = []

    def flush(blocks: Blocks | None) -> None:
        if blocks is not None and blocks.buf:
            blocks.items.append(TextBlock(" ".join(blocks.buf).strip()))
            blocks.buf = []

    def current_text_blocks() -> Blocks | None:
        """Where ordinary prose lines should go right now."""
        if env_stack:
            top = env_stack[-1]
            if top["kind"] in ("list", "bib") and top["item"] is not None:
                return top["item"]
            return None  # list/bib before the first item: ignore stray lines
        return active

    for line_no, line in enumerate(body.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            flush(current_text_blocks())
            continue

        m = _SECTION_RE.match(stripped)
        if m:
            flush(current_text_blocks())
            level = {
                "section": 1,
                "subsection": 2,
                "subsubsection": 3,
            }[m.group(1)]
            active.items.append(HeadingBlock(level, m.group(2)))
            continue

        m = _PARAGRAPH_RE.match(stripped)
        if m:
            flush(current_text_blocks())
            active.items.append(RuninBlock(m.group(2)))
            continue

        m = _BEGIN_RE.match(stripped)
        if m:
            kind = m.group(1)
            flush(current_text_blocks())
            if kind == "abstract":
                content = Blocks()
                active.items.append(AbstractBlock(content))
                env_stack.append({"kind": "abstract", "parent": active})
                active = content
            elif kind in ("itemize", "enumerate"):
                node = ListBlock("ul" if kind == "itemize" else "ol", [])
                active.items.append(node)
                env_stack.append(
                    {
                        "kind": "list",
                        "tag": "ul" if kind == "itemize" else "ol",
                        "node": node,
                        "item": None,
                        "parent": active,
                    }
                )
            else:  # thebibliography
                node = BibBlock([])
                active.items.append(node)
                env_stack.append(
                    {"kind": "bib", "node": node, "item": None, "parent": active}
                )
            continue

        m = _END_RE.match(stripped)
        if m:
            if not env_stack:
                raise RuntimeError(
                    f"unmatched \\end{{{m.group(1)}}} at body line {line_no}"
                )
            flush(current_text_blocks())
            env = env_stack.pop()
            active = env["parent"]
            if env["kind"] == "abstract":
                flush(env.get("content", None))
            elif env["kind"] == "list":
                if env["item"] is not None:
                    flush(env["item"])
            elif env["kind"] == "bib":
                if env["item"] is not None:
                    flush(env["item"])
            continue

        if stripped == r"\maketitle":
            continue

        if env_stack and env_stack[-1]["kind"] in ("list", "bib"):
            top = env_stack[-1]
            if top["kind"] == "list":
                m = _ITEM_RE.match(line)
                if m:
                    if top["item"] is not None:
                        flush(top["item"])
                    item = Blocks()
                    top["node"].items.append(item)
                    top["item"] = item
                    rest = m.group(1).strip()
                    if rest:
                        item.buf.append(rest)
                    continue
            else:  # bib
                m = _BIBITEM_RE.match(line)
                if m:
                    if top["item"] is not None:
                        flush(top["item"])
                    item = Blocks()
                    top["node"].items.append((m.group(1), item))
                    top["item"] = item
                    rest = m.group(2).strip()
                    if rest:
                        item.buf.append(rest)
                    continue

        blocks = current_text_blocks()
        if blocks is not None:
            blocks.buf.append(stripped)

    flush(current_text_blocks())
    flush(active)
    return root


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def render_blocks(
    blocks: Blocks, *, inside_li: bool = False, nums: dict | None = None
) -> str:
    if nums is None:
        nums = {"sec": 0, "sub": 0, "subsub": 0}
    out: list[str] = []
    items = blocks.items
    i = 0
    while i < len(items):
        item = items[i]

        if isinstance(item, TextBlock):
            out.append(paragraph_to_html(item.text))

        elif isinstance(item, HeadingBlock):
            tag = {1: "h1", 2: "h2", 3: "h3"}[item.level]
            if item.level == 1:
                nums["sec"] += 1
                nums["sub"] = 0
                nums["subsub"] = 0
                number = str(nums["sec"])
            elif item.level == 2:
                nums["sub"] += 1
                nums["subsub"] = 0
                number = f"{nums['sec']}.{nums['sub']}"
            else:
                nums["subsub"] += 1
                number = f"{nums['sec']}.{nums['sub']}.{nums['subsub']}"
            out.append(
                f"<{tag}><span class=\"secnum\">{number}. </span>"
                f"{text_to_html(item.title)}</{tag}>"
            )

        elif isinstance(item, RuninBlock):
            title_html = text_to_html(item.title)
            nxt = items[i + 1] if i + 1 < len(items) else None
            if isinstance(nxt, TextBlock):
                body = text_to_html(nxt.text)
                out.append(f"<p><strong class=\"runin\">{title_html}</strong> {body}</p>")
                i += 1
            else:
                out.append(f"<h4>{title_html}</h4>")

        elif isinstance(item, ListBlock):
            tag = item.kind
            inner: list[str] = []
            for li_blocks in item.items:
                inner.append(
                    f"<li>{render_blocks(li_blocks, inside_li=True, nums=None)}</li>"
                )
            out.append(f"<{tag}>{''.join(inner)}</{tag}>")

        elif isinstance(item, BibBlock):
            inner = []
            for key, li_blocks in item.items:
                body = render_blocks(li_blocks, nums=None).strip()
                inner.append(
                    f'<li id="ref-{html.escape(key, quote=True)}">'
                    f"<span class=\"refnum\">[{html.escape(key)}]</span> "
                    f"{body}</li>"
                )
            out.append(f'<ol class="references">{"".join(inner)}</ol>')

        elif isinstance(item, AbstractBlock):
            inner = render_blocks(item.content, nums=None)
            out.append(f'<div class="abstract"><div class="abstract-title">Abstract</div>{inner}</div>')

        i += 1
    return "\n".join(out)


# --------------------------------------------------------------------------
# Document assembly
# --------------------------------------------------------------------------


CSS = r"""
@page {
  size: A4;
  margin: 25.4mm;
}

html {
  font-size: 12pt;
}

body {
  font-family: "Times New Roman", "Nimbus Roman", Georgia, serif;
  line-height: 1.42;
  color: #000;
  text-align: justify;
  hyphens: auto;
}

.titleblock {
  text-align: center;
  margin-bottom: 0.9em;
}
.titleblock h1 {
  font-size: 16.5pt;
  line-height: 1.3;
  margin: 0 0 0.2em;
}
.titleblock h1::before {
  content: none;
  counter-increment: none;
}

.abstract {
  font-size: 10.8pt;
  margin: 1em 3em 1.4em;
}
.abstract .abstract-title {
  text-align: center;
  font-weight: bold;
  margin-bottom: 0.4em;
}

h1, h2, h3, h4 {
  font-family: "Times New Roman", "Nimbus Roman", Georgia, serif;
  text-align: left;
  font-weight: bold;
}
h1 {
  font-size: 13.5pt;
  margin: 1.05em 0 0.55em;
}
h2 {
  font-size: 12.5pt;
  margin: 0.85em 0 0.4em;
}
h3 {
  font-size: 12pt;
  margin: 0.7em 0 0.35em;
}
h4 {
  font-size: 12pt;
  margin: 0.7em 0 0.35em;
}

p {
  margin: 0 0 0.55em;
}

ul, ol {
  margin: 0.25em 0 0.7em;
}
li {
  margin: 0 0 0.3em;
}
li > ul, li > ol {
  margin-top: 0.3em;
}

.eq {
  display: block;
  text-align: center;
  margin: 0.45em 0;
  font-size: 0.98em;
}

math {
  font-family: "Cambria Math", "STIX Two Math", "Latin Modern Math", "Times New Roman", serif;
}

ol.references {
  list-style: none;
  padding-left: 0;
  margin: 0.4em 0;
}
ol.references li {
  margin-bottom: 0.45em;
  padding-left: 2.4em;
  text-indent: -2.4em;
}
ol.references li > p {
  display: inline;
  margin: 0;
}
.refnum {
  font-weight: normal;
}

.runin {
  font-weight: bold;
}
"""


def build_html(tex_src: str) -> str:
    title_m = re.search(r"\\title\{(.*?)\}", tex_src, flags=re.S)
    title = title_m.group(1).strip() if title_m else ""

    start = tex_src.index(r"\begin{document}")
    end = tex_src.index(r"\end{document}")
    body_tex = tex_src[start + len(r"\begin{document}") : end]
    root = parse_body(body_tex)
    body_html = render_blocks(root)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>{CSS}</style>
</head>
<body>
<div class="titleblock">
<h1>{text_to_html(title)}</h1>
</div>
{body_html}
</body>
</html>
"""


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    tex = open(SRC, encoding="utf-8").read()
    out_html = build_html(tex)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(out_html)
    print(f"wrote {OUT} ({len(out_html)} bytes)")


if __name__ == "__main__":
    main()
