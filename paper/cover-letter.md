# Cover letter

Target journal: **Physica D: Nonlinear Phenomena** (Elsevier), submitted via
Editorial Manager.

Previously desk rejected by Nonlinearity as NON-111406 on 10 September 2026,
without review, on the grounds that the manuscript did not make a sufficient
contribution to the literature. Nonlinearity was a defensible first choice, the
family having been introduced there, and it does carry this literature
currently. The lesson was not about venue. The old cover letter described the
contribution as "deliberately narrow" in its second paragraph, which handed the
editor the conclusion. Every attribution has been kept and that verdict has
been removed, from the letter and from the introduction and Limitations
section. See the introduction's paragraph beginning "The consequences reach
past the decimal places" for the case that was previously missing.

Submission notes:

- **Physica D uses single anonymized review.** Referees see the author. Upload
  `paper/degregorio.pdf`, the named build, not the anonymised one. The
  anonymised build exists for any double anonymous venue and is produced by
  `python arxiv_package.py --anon`.
- Because referees see the author, the Data availability section carries the
  repository URL, and the reproducibility is visible to them rather than
  confined to the cover letter. The `--anon` build strips that URL
  automatically and substitutes a wording that withholds it, so both variants
  stay correct.
- Elsevier asks for source files rather than a single PDF at submission for
  some journals; check what Editorial Manager requests. If it wants LaTeX,
  `paper/arxiv/degregorio.tar.gz` from `python arxiv_package.py` is a flat,
  verified package.
- The abstract is 296 words. Physica D publishes no hard limit, so this is
  comfortable.
- Open access is optional. Decline it; the subscription route is free.

The letter itself is `paper/cover-letter.tex`, which compiles to the one page
PDF the portal asks for:

    cd paper && pdflatex cover-letter.tex

It carries the author's name, the ORCID and the repository URL, which is
correct for a single anonymized venue and for the editor in any case.
