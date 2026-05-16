# Third-Party Notices

This file documents third-party code adapted into this repository, with
upstream attribution preserved. Transitive dependencies installed via
`pip` are governed by their own licenses (see `pyproject.toml` for the
canonical list).

---

## Summary

| Upstream | Local path | Upstream license |
|---|---|---|
| sklearn-compat — single-file compat shim | `src/tabpfn_extensions/misc/sklearn_compat.py` | BSD-3-Clause |

---

## Per-upstream notices

### sklearn-compat — compatibility shim

**Upstream:** https://github.com/sklearn-compat/sklearn-compat
**Local path:** `src/tabpfn_extensions/misc/sklearn_compat.py` (single-file vendored distribution; ~1000 lines)
**License:** BSD-3-Clause
**Copyright:** Copyright (c) 2024, Guillaume Lemaitre (per the upstream `LICENSE`)
**Modifications:** None of significance; vendored verbatim from upstream version 0.1.5 to avoid an extra runtime dependency. The single-file format is the distribution model sklearn-compat itself encourages for downstream users. File is kept as close to upstream as possible — both `ruff` formatting and `mypy` type-checking are suppressed at the top of the file.

---

## Adding new entries

When vendoring or adapting third-party code:

1. Preserve any upstream per-file copyright and license header verbatim. If the upstream does not ship a per-file header, add an attribution block citing the upstream URL, copyright holder, and SPDX license identifier.
2. When vendoring a whole directory of upstream code, also vendor the upstream `LICENSE` / `NOTICE` file alongside it. For single-file adaptations, the in-file attribution plus the entry in this NOTICE file is sufficient.
3. Add a row to the summary table and a per-upstream notice to this file, including the upstream copyright line when one is published.
