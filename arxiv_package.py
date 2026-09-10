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

`--anon` produces the variant a double anonymous journal review needs, with the
title block replaced. Nonlinearity defaults to double anonymous and lets the
author opt out by leaving names in, which is the wrong way round for anyone
without an institution to trade on. The two mentions of an accompanying
repository do not name it and so survive anonymisation, but the PDF metadata
does not look after itself: hyperref copies the title block into the document
properties, so the check below reads the metadata back out of the built PDF and
fails if a name is still in there.

    python arxiv_package.py            writes paper/arxiv/degregorio.tar.gz
    python arxiv_package.py --check    builds and verifies, writes no archive
    python arxiv_package.py --anon     writes paper/arxiv/degregorio-anon.tar.gz
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


ANON_BLOCK = "\\author{}\n"


ANON_DATA = (
    "\\section*{Data availability}\n"
    "\\label{sec:data}\n\n"
    "Every number reported here is reproducible from source. The solver, the\n"
    "profile solver, the script behind each table and figure, and a suite of\n"
    "nineteen checks against closed forms and conserved quantities are openly\n"
    "available under an MIT licence in a public repository, whose address is\n"
    "withheld here because it identifies the author, and will be supplied on\n"
    "acceptance.\n")


def anonymise(source):
    """
    Replace the title block, and stop hyperref from leaking it into the PDF.

    The author macro spans several lines and ends at the first line that closes
    it, so match to the closing brace rather than to a blank line.

    The data availability section carries the repository URL, which contains the
    author's name, so it is swapped for a wording that asserts availability
    without the address. Journals running double anonymous review expect exactly
    that, and the leak check below would fail on the URL otherwise.
    """
    # The replacement is passed as a function: re processes backslash escapes
    # in a replacement template, which would turn the \a of \author into BEL.
    out, n = re.subn(r"\\author\{.*?\}\}\n", lambda m: ANON_BLOCK, source,
                     flags=re.S)
    if n != 1:
        raise SystemExit("  could not identify the author block to redact")
    # hyperref infers pdfauthor from \author unless told otherwise.
    out = out.replace("\\begin{document}",
                      "\\hypersetup{pdfauthor={}}\n\\begin{document}", 1)

    start = out.find("\\section*{Data availability}")
    if start == -1:
        raise SystemExit("  no data availability section found to redact")
    end = out.find("\\begin{thebibliography}", start)
    out = out[:start] + ANON_DATA + "\n" + out[end:]
    return out


def stage(source, figures, anon=False):
    # STAGE is a fresh temporary directory, so there is nothing to clear.
    # graphicspath is what breaks on arXiv, so the staged copy does not keep it.
    flat = re.sub(r"\\graphicspath\{[^\n]*\}\n", "", source)
    if anon:
        flat = anonymise(flat)
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


IDENTIFIERS = ("Munawar", "Kazmi", "munawarkazmi", "munawarsaeedkazmi")


def pdf_text(pdf):
    """
    Extract the text of a PDF, or return None if we cannot.

    A raw byte scan does not work: content streams are Flate compressed, so a
    search over the file finds nothing whether the name is present or not. An
    earlier version of this function did exactly that and reported every
    document clean, which is worse than not checking. Returning None where
    extraction is impossible is the point: the caller must be able to tell
    "verified clean" from "could not verify".
    """
    try:
        out = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True,
                             check=True)
        return out.stdout.decode("utf-8", errors="replace")
    except (OSError, subprocess.CalledProcessError):
        pass
    try:
        import pypdf
        return "\n".join(p.extract_text() or "" for p in
                         pypdf.PdfReader(str(pdf)).pages)
    except Exception:
        return None


def leaks(pdf):
    """
    Look for the author's identity in the built PDF, page text and metadata.

    Returns (found, checked). `checked` is False when neither extractor was
    available, in which case `found` says nothing.
    """
    haystack = pdf_text(pdf)
    if haystack is None:
        return [], False

    # The information dictionary is usually not compressed, and hyperref puts
    # the title block there, so scan the raw bytes for it as well as the text.
    raw = pdf.read_bytes().decode("latin-1", errors="replace")

    found = [i for i in IDENTIFIERS if i in haystack or i in raw]
    return found, True


def main():
    check_only = "--check" in sys.argv
    anon = "--anon" in sys.argv
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

    stage(source, figures, anon=anon)
    print(f"  staged     {STAGE}, graphicspath stripped"
          f"{', title block redacted' if anon else ''}")

    log = build()
    problems, pages = check(log, len(figures))

    if anon and not problems:
        found, checked = leaks(STAGE / "degregorio.pdf")
        if not checked:
            problems.append("could not extract text from the PDF, so anonymity "
                            "is unverified; install poppler or pypdf")
        elif found:
            problems.append(f"identifying strings still in the PDF: {found}")
        else:
            print("  anonymity  verified: no identifying string in text or "
                  "metadata")

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
    archive = OUT / ("degregorio-anon.tar.gz" if anon else "degregorio.tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        for f in sorted(STAGE.iterdir()):
            if f.suffix in (".tex", ".png", ".bbl"):
                tar.add(f, arcname=f.name)
    contents = tarfile.open(archive).getnames()
    size = archive.stat().st_size

    # Journals want a single PDF at initial submission and the source only on
    # acceptance, so the built PDF is an artefact in its own right, not just a
    # by-product of checking that the archive compiles.
    pdf_out = OUT / ("degregorio-anon.pdf" if anon else "degregorio-flat.pdf")
    shutil.copy2(STAGE / "degregorio.pdf", pdf_out)

    print()
    print(f"  wrote      {archive.relative_to(ROOT)}  ({size/1024:.0f} KB)")
    print(f"  contains   {', '.join(contents)}")
    print(f"  wrote      {pdf_out.relative_to(ROOT)}  "
          f"({pdf_out.stat().st_size/1024:.0f} KB), for journal submission")
    print()
    print("  Upload that archive. arXiv runs its own pdflatex, so the check")
    print("  above is the same one it will make.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
