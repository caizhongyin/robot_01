"""Compile a LaTeX source file into PDF with pdflatex.

Usage:
    python latex2pdf.py [source.tex]

If no source file is given, ``tigi_v1.2.tex`` in this script's directory is
compiled.  The generated PDF (plus .aux/.log) is written next to the .tex file.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TEX_FILE = "tigi_v1.2.tex"


def find_pdflatex() -> str | None:
    """Return a usable pdflatex executable, or None if none is found."""
    found = shutil.which("pdflatex")
    if found:
        return found

    # Common Windows install locations (checked when pdflatex is not on PATH).
    candidates = [
        r"C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe",
        r"C:\Program Files (x86)\MiKTeX\miktex\bin\x64\pdflatex.exe",
        r"D:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe",
        r"C:\texlive\bin\windows\pdflatex.exe",
    ]
    user_profile = os.environ.get("USERPROFILE") or os.environ.get("HOME")
    if user_profile:
        candidates.insert(
            0,
            os.path.join(
                user_profile,
                "AppData",
                "Local",
                "Programs",
                "MiKTeX",
                "miktex",
                "bin",
                "x64",
                "pdflatex.exe",
            ),
        )

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def run_pdflatex(pdflatex: str, tex_path: Path) -> None:
    """Run one pdflatex pass; raise SystemExit on failure with log hints."""
    cmd = [
        pdflatex,
        f"-jobname={tex_path.stem}",
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-output-directory={tex_path.parent}",
        str(tex_path),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(tex_path.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        print(proc.stdout[-4000:])
        print(
            f"pdflatex failed (exit code {proc.returncode}). "
            f"See the log tail above or {tex_path.stem}.log for details.",
            file=sys.stderr,
        )
        raise SystemExit(proc.returncode)


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEX_FILE
    tex_path = (SCRIPT_DIR / source).resolve()
    if not tex_path.is_file():
        raise SystemExit(f"Source file not found: {tex_path}")

    pdflatex = find_pdflatex()
    if pdflatex is None:
        raise SystemExit(
            "pdflatex was not found. Install a TeX distribution (e.g. MiKTeX or "
            "TeX Live) or add its bin directory to PATH and try again."
        )

    print(f"pdflatex: {pdflatex}")
    print(f"source  : {tex_path}")

    # Two passes so \ref / \label cross-references are resolved.
    for pass_no in (1, 2):
        print(f"pass {pass_no}/2 ...")
        run_pdflatex(pdflatex, tex_path)

    pdf_path = tex_path.with_suffix(".pdf")
    if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
        raise SystemExit(
            f"Compilation finished but no PDF was produced at {pdf_path}."
        )
    print(f"done   : {pdf_path} ({pdf_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
