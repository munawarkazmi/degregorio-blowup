# Cover letter

Target journal: **Nonlinearity** (IOP Publishing / London Mathematical Society).

Chosen because the family itself was introduced there: Okamoto, Sakajo and
Wunsch, *On a generalization of the Constantin, Lax and Majda equation*,
Nonlinearity **21** (2008) 2447, which is `osw2008` in the bibliography. The
editorial board has handled this exact problem. Publication on the subscription
route is free of charge; the article publication charge applies only if gold
open access is chosen after acceptance.

Submission notes, not part of the letter:

- Upload `paper/arxiv/degregorio-anon.pdf`, built by
  `python arxiv_package.py --anon`. IOP wants a single PDF at initial
  submission and asks for source only on acceptance, which is when
  `degregorio-anon.tar.gz` becomes the relevant file. Nonlinearity defaults to
  double anonymous review, and the anonymised build is the one that belongs in
  the portal.
- The letter goes to the editor rather than the referees, so it carries the
  author's name even under double anonymous review.
- The abstract is 296 words against their limit of 300.
- The cover letter must be uploaded as a file. The portal has no box to paste
  it into.
- Data availability: choose the statement saying all data are included within
  the article. Every number quoted appears in a table, and that option needs
  no URL, where naming the repository in the manuscript would undo the
  anonymisation. The editor learns about the repository from the letter.

The letter itself is `paper/cover-letter.tex`, which compiles to the PDF the
portal asks for. It is kept as LaTeX rather than duplicated here so there is
one source of truth:

    cd paper && pdflatex cover-letter.tex

It carries the author's name, the ORCID and the repository URL. That is
correct: the cover letter goes to the editor, not to the referees, so it can
say things the anonymised manuscript cannot.
