# Release notes

Curated, user-facing notes for each toolkit version live here as one file per
version. The release workflow uses the file that matches the package version /
git tag as the GitHub Release body and **fails if it is missing or empty**.

## Convention

| Path | Role |
| --- | --- |
| `docs/releases/<X.Y.Z>.md` | Notes for tag `v<X.Y.Z>` (same version as `pyproject.toml`) |
| [`docs/releases/certification-matrix.md`](certification-matrix.md) | Generated public certification matrix (from `docs/certifications/*.json`) |
| [GitHub Releases](https://github.com/defai-digital/axquant/releases) | Public history for every published tag |

Do **not** dump commit messages. Write short Fixed / Changed / Added bullets an
operator can act on (regenerate artifacts, host id renames, install path).

## On every release

1. Bump `project.version` and `axquant.__version__`.
2. Add `docs/releases/<new-version>.md` with the curated notes (same commit as
   the version bump).
3. Tag `v<new-version>` and push; CI extracts that file into the Release.

Older versions stay on GitHub Releases only. You may delete or omit older
`docs/releases/*.md` files after a tag is published; the packaging test only
requires a non-empty file for the **current** package version.
