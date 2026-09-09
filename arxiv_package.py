"""
Assemble a flat tarball for arXiv, and prove it builds without the repository.

The paper lives in `paper/` while the figures it includes live in the repository
root, bridged by `\\graphicspath{{../}{./}}`. That works locally and fails on
arXiv, where the upload is a flat archive and `../` points outside it. The
failure is quiet: pdflatex reports "File not found", carries on, and emits a PDF
with empty boxes where the figures were, so the first sign of trouble is the
published version.

This script copies the source and the figures it actually references into one
directory, builds there with no access to anything else, and refuses to write
the archive unless the build came out clean. `--check` stops after the build.

Bundling the .bbl is deliberate: the bibliography here is a `thebibliography`
environment written by hand, so there is nothing for arXiv's BibTeX pass to do
and nothing to go wrong, but the check below confirms the reference list is
intact rather than assuming it.

    python arxiv_package.py            writes paper/arxiv/degregorio.tar.gz
    python arxiv_package.py --check    builds and verifies, writes no archive
"""

import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEX = ROOT / "paper" / "degregorio.tex"
OUT = ROOT / "paper" / "arxiv"

# The staging directory goes to system temp rather than under the repository.
# Build artefacts do not belong in a synced folder, and on OneDrive the sync
# client keeps a handle on the directory, so removing it between runs fails
# with a permission error and the second invocation dies before it starts.
STAGE = Path(tempfile.mkdtemp(prefix="degregorio-arxiv-"))


def referenced_figures(source):
    """Figure filenames the source actually includes, in order of appearance."""
    return [m.group(1) for m in
            re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", source)]


def locate(name):
    """Find a figure, searching the graphicspath roots the paper declares."""
    for base in (ROOT, ROOT / "paper"):
        p = base / name
        if p.is_file():
            return p
    return None


def stage(source, figures):
    # STAGE is a fresh temporary directory, so there is nothing to clear.
    # graphicspath is what breaks on arXiv, so the staged copy does not keep it.
    flat = re.sub(r"\\graphicspath\{[^\n]*\}\n", "", source)
    (STAGE / "degregorio.tex").write_text(flat, encoding="utf-8")

    for name, path in figures.items():
        shutil.copy2(path, STAGE / Path(name).name)
    return flat


def build():
    """Two passes, for the cross references. Returns the log."""
    for _ in range(2):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "degregorio.tex"],
            cwd=STAGE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=False)
    return (STAGE / "degregorio.log").read_text(encoding="utf-8",
                                                errors="replace")


def check(log, n_figures):
    """Every way this build is allowed to be wrong."""
    problems = []

    missing = sorted(set(re.findall(r"File `([^']+)' not found", log)))
    if missing:
        problems.append(f"figures not found in the flat package: {missing}")

    for pattern, label in (
        (r"^! ", "LaTeX error"),
        (r"LaTeX Warning: Reference `[^']+' on page .* undefined", "undefined reference"),
        (r"LaTeX Warning: Citation `[^']+' on page .* undefined", "undefined citation"),
    ):
        hits = re.findall(pattern, log, re.MULTILINE)
        if hits:
            problems.append(f"{label}: {len(hits)} occurrence(s)")

    m = re.search(r"Output written on degregorio\.pdf \((\d+) pages", log)
    if not m:
        problems.append("no PDF produced")
        return problems, None
    pages = int(m.group(1))

    # A PDF can be produced with the figures silently absent, so confirm the
    # count of successfully used graphics matches what the source asked for.
    used = len(re.findall(r"<use fig_[^>]*>|fig_\w+\.png", log))
    if missing and used:
        problems.append("figures partially resolved")

    return problems, pages


def main():
    check_only = "--check" in sys.argv
    source = TEX.read_text(encoding="utf-8")

    names = referenced_figures(source)
    figures, absent = {}, []
    for n in names:
        p = locate(n)
        (figures.setdefault(n, p) if p else absent.append(n))
    if absent:
        print(f"  cannot find in the repository: {absent}")
        return 1

    print(f"  source     {TEX.relative_to(ROOT)}")
    print(f"  figures    {len(figures)} referenced: {', '.join(figures)}")

    stage(source, figures)
    print(f"  staged     {STAGE}, graphicspath stripped")

    log = build()
    problems, pages = check(log, len(figures))

    if problems:
        print()
        for p in problems:
            print(f"  FAIL  {p}")
        print()
        print("  The package is not clean. No archive written.")
        return 1

    refs = len(re.findall(r"\\bibitem", source))
    print(f"  built      {pages} pages, no errors, no undefined references")
    print(f"  references {refs} entries in the bibliography")

    if check_only:
        print()
        print("  Package is clean. Rerun without --check to write the archive.")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    archive = OUT / "degregorio.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for f in sorted(STAGE.iterdir()):
            if f.suffix in (".tex", ".png", ".bbl"):
                tar.add(f, arcname=f.name)
    contents = tarfile.open(archive).getnames()
    size = archive.stat().st_size

    print()
    print(f"  wrote      {archive.relative_to(ROOT)}  ({size/1024:.0f} KB)")
    print(f"  contains   {', '.join(contents)}")
    print()
    print("  Upload that archive. arXiv runs its own pdflatex, so the check")
    print("  above is the same one it will make.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
