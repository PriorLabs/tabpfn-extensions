# Releasing tabpfn-extensions

Maintainer-facing guide to cutting a release. Contributors don't need
to read this — see `changelog/README.md` for the fragment convention.

The release pipeline is fully automated from a single workflow
dispatch. There are two manual checkpoints: reviewing the
auto-generated release PR, and approving the `pypi` deployment.

## Releasing

1. **Pick the base commit.** Find the full 40-character SHA on `main`
   you want to release from (usually the tip):

   ```bash
   git rev-parse origin/main
   ```

2. **Trigger `Create Release PR`.** Actions tab → *Create Release PR*
   → *Run workflow*:
   - `version`: the new version, no `v` prefix (e.g. `0.5.0`)
   - `base_ref`: the 40-char SHA from step 1
   - `base_branch`: leave as `main`

   The workflow opens a PR on `release/v<version>` with:
   - `pyproject.toml` `[project].version` bumped
   - A new section in `CHANGELOG.md` assembled by Towncrier from
     fragments in `changelog/`
   - Those fragments deleted

3. **Review and merge the release PR.** Sanity-check the version bump
   and the changelog. Merge with a regular merge commit (not squash —
   the tag will point at the merge commit).

4. **Auto-tag fires.** `release-tag-on-merge.yml` creates and pushes
   `v<version>` on the merge commit.

5. **TestPyPI publish + install-verify** run automatically on the tag
   push. The install-verify step `pip install`s the freshly published
   package from TestPyPI in a clean venv and imports it.

6. **Approve the `pypi` deployment.** Once TestPyPI is green, the
   `publish-pypi` job pauses waiting for an environment approval. Go
   to the workflow run, click *Review deployments*, approve `pypi`.

7. **PyPI publish + GitHub Release** run. The GitHub Release uses the
   new CHANGELOG section as its body.

8. **Spot-check the published artifact.** In a fresh venv:

   ```bash
   pip install tabpfn-extensions==<version>
   python -c "import tabpfn_extensions; print(tabpfn_extensions.__version__)"
   ```

## Workflows reference

| File | Trigger | Purpose |
|---|---|---|
| `.github/workflows/release-create-pr.yml` | `workflow_dispatch` | Opens the release PR |
| `.github/workflows/release-tag-on-merge.yml` | PR merge of `release/v*` | Pushes the version tag |
| `.github/workflows/release-publish.yml` | Tag push `v*` | Builds, TestPyPI + verify, gated PyPI publish, GitHub Release |
| `.github/workflows/check-changelog.yml` | Every PR | Enforces a changelog fragment exists |

## Versioning

Single source of truth: `[project].version` in `pyproject.toml`.
`release-create-pr.yml` writes it; `release-publish.yml` cross-checks
it against the tag and refuses to publish on mismatch. SemVer applies.

## Pre-releases

The pipeline accepts PEP 440 pre-release versions (`0.5.0rc1`,
`0.5.0a1`, etc.). PyPI marks them as pre-releases automatically and
`pip install` won't pick them up without `--pre`. Good for end-to-end
testing the pipeline without committing to a stable version.

## Recovering from a failed run

| Failure point | Recovery |
|---|---|
| `Create Release PR` fails before pushing the branch | Re-run the workflow. No cleanup needed. |
| `Create Release PR` fails after pushing the branch | Delete the `release/v*` branch on GitHub, then re-run. |
| Release PR opened but version/changelog wrong | Close the PR, delete the branch, fix fragments on `main`, re-run `Create Release PR`. Do **not** hand-edit the release branch. |
| `release-tag-on-merge` fails | Re-run from the Actions UI. If the tag was partially created locally but not pushed, the workflow's idempotency check handles it. |
| `publish-testpypi` fails | Investigate (most common: TestPyPI propagation lag, dependency conflict in install-verify). Re-run *publish-testpypi* job; the build artifact is retained for 7 days. |
| `publish-pypi` fails after approval | If artifacts already uploaded to PyPI, do **not** re-upload — yank the bad release via the PyPI UI and cut a new version. PyPI does not allow re-uploads of the same version. |
| GitHub Release creation fails | Create it manually via `gh release create v<version> --notes-file <(...)` using the CHANGELOG section. |

## Yanking a release

If a published release is broken, yank it (don't delete — that breaks
existing pins):

- PyPI: project page → *Manage* → *Releases* → *Options* → *Yank*
- TestPyPI: same flow at test.pypi.org

Yanked versions remain installable by exact pin but are excluded from
resolution. Follow up with a fixed release.
