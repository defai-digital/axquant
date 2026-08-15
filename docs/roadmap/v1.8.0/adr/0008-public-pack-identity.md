# ADR-0008 — Public pack identity and immutable editions

- **Status:** accepted
- **Date:** 2026-08-14
- **Release:** AXQuant 1.8.0
- **Supersedes:** MP-named Hub repository enforcement in `claims.py`
- **References:** `src/axquant/naming.py`, `src/axquant/model_card.py`

## Context

Users split on public names:

- **Camp 1.** Drop `4bit` / `6bit`. One repo per base, maybe an edition
  suffix: `AX-Qwen3.8-27B-MLX-AXQ` or `…-v3`.
- **Camp 2.** Keep `4bit` / `6bit`. People search those strings and pick a
  memory-versus-quality SKU.

The toolkit already has two identities:

| Identity | Producer | Example |
| --- | --- | --- |
| Product-class SKU | `model_name()` | `AX-Qwen3.6-27B-MLX-AXQ-4bit-MTP` |
| Measured-BPW claim | `certified_mixed_precision_name()` | `AX-Qwen3.6-27B-MLX-AXQ-MP-5p42bpw-MTP` |

The Hub catalog, model-card renderer, and published certificates use the
class SKU. Flagship public-claim construction currently *requires* the MP
string as `public_repository` (`claims.py`). Those two contracts disagree.

A `4bit` pack is already not uniform 4-bit. Qwen 3.6 27B AXQ 4-bit MTP
measures about 5.42 main BPW. Some bases collapse: a distinct 4-bit sibling
would not shrink the download, so only `6bit` is published.

Codex (`gpt-5.6-sol`) and Qoder (`Qwen3.8-Max`) independently reviewed the
camps. Both rejected Camp 1 as a rename of live certified URLs. Both
rejected putting measured BPW in the repository name (it churns when an
edition's BPW moves). Codex additionally rejected an unversioned alias
repo; Hub collections already provide family discovery.

## Decision

**Two-level identity, based on Camp 2, with Camp 1's honesty rules made
normative.**

> The repository name says which product lane to choose.
> The certificate says exactly what was measured.

### Canonical repository

```text
AutomatosX/AX-<Base>-MLX-AXQ-<4bit|6bit|8bit|2bit|3bit>[-MTP]
```

- Class comes from the **requested budget lane**, not from measured BPW.
- No measured BPW and no edition in the repo name.
- `-MTP` means MTP assets are packaged. It is never a speed claim.
- Created once per `(base, class, mtp-state)`.

### Canonical claim

```text
<exact repo>@<immutable commit>
artifact edition vN
target_class = 4bit | 6bit | …
measured_main_bpw = <unrounded machine value>
```

Display rounding is decimal half-up to two places (`naming.py`). Manifests
keep the full value and remain independently recomputable from bytes and
logical parameters.

### Where each string lives

| Surface | Content |
| --- | --- |
| Hub repository | Class SKU. Never MP. Never edition. |
| Hub tag | Immutable `vN` for a certified edition. `main` is a convenience pointer only. |
| Toolkit git tag | SemVer of the toolkit (`v1.8.0`), unrelated to pack editions. |
| Certificate title | SKU + edition + tier + rounded measured main BPW. |
| Model-card H1 | Repo leaf + rounded measured main BPW. |
| `target_class` | Requested lane token matching the repo suffix. Metadata, not a benchmark. |
| `measured_main_bpw` | Authoritative claim. Required on every Spec v1.0 certificate. |
| MP string | Derived display field inside certificates / flagship claims only. Not a repository generator. |

### Floor collapse

A new `4bit` edition must reduce complete weight bytes by at least **5%**
versus the same-base, same-MTP-state `6bit` sibling. Otherwise publish or
advance only the `6bit` SKU and render the existing “no distinct 4-bit pack”
reason on the card.

Existing certified repositories stay, even if a later edition would
collapse. Mark the old edition superseded. Do not delete or rename.

### We will never

- Rename or delete a live certified repository.
- Move or rewrite an edition tag after publication.
- Bind a certificate only to mutable `main`.
- Claim that `4bit` means every tensor is four-bit.
- Describe AXQ `4bit`/`6bit` as GGUF Q4/Q6.
- Put measured BPW into a repository name.
- Publish a redundant floor-collapsed sibling.
- Let `-MTP` imply Tier 2 acceleration.
- Allow `product_class`, plan `target_class`, and repository suffix to disagree.
- Silently upgrade legacy certificate JSON to Certification Spec v1.0.

## Consequences

- 1.8.0 must stop requiring MP names as Hub repositories in `claims.py` /
  `render_certified_model_card`. The MP helper remains for derived claim
  labels.
- Cards and certificates lead with measured BPW so Camp 1's honesty concern
  is met without breaking URLs.
- Flagship campaigns keep `target_class ∈ {4bit, 6bit}` as the public lane
  set. Experimental 2/3-bit stay labeled experimental.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Camp 1: one name per base | Breaks every bound `hub_repo_id`, collection, and citation. Hides the memory-vs-quality choice. Still needs `-MTP` and edition. |
| MP string as the repo | Unstable across editions (5.42 vs 5.07). Unsearchable. Conflicts with `model_card.py` name parser. |
| Unversioned alias repo (`AX-<Base>-MLX-AXQ`) | Second identity to sync; must pick one class and one MTP state; Hub collections already cover discovery. |
| Derive `target_class` from measured BPW | A 5.42-BPW 4-bit lane would flip to a 6-bit label and lie about the requested budget. |
