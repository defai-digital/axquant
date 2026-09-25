# MTP certification evidence hardening (1.9)

AXQuant 1.9 (program MH, AXQ-045) hardens the evidence chain for MTP release
claims. Two behavior changes affect release paths; no frozen artifact schema
changed (`axquant.artifact.v2` is untouched).

## Publish and release-audit reject unbound artifacts

`axquant publish` and `axquant release-audit` now reject an artifact directory
that does not carry a valid evidence-binding sidecar,
`axquant_evidence_binding.json`, next to `axquant_manifest.json`
(`axquant.artifact-evidence-binding.v1`). The sidecar binds:

- the semantic digest (`stable_sha256`) of the loaded artifact manifest — a
  manifest that no longer matches the sidecar digest is rejected; and
- at least one certificate file binding (byte SHA-256 of the Tier 1 and/or
  Tier 2 certificate file). A sidecar with no certificate binding is rejected,
  because release claims require bound certification evidence.

The sidecar is written by release tooling only — never during `convert`, whose
output remains development evidence by design. `publish-prepare` refreshes the
manifest digest automatically after it updates the live manifest; it never
invents certificate bindings.

## Binding an in-flight release candidate

For a release candidate prepared before this version, write the sidecar once
the artifact is final (after `publish-prepare`, so the digest matches the
prepared manifest):

```bash
axquant bind-artifact-evidence \
  --model /path/to/artifact \
  --tier1-certificate /path/to/tier1-cert.json \
  --tier2-certificate /path/to/tier2-cert.json
```

At least one of `--tier1-certificate` / `--tier2-certificate` is required.
`--evidence-kind` defaults to `measured`. The equivalent library helper is
`axquant.artifact_evidence_binding.write_artifact_evidence_binding(...)`.

Campaign scripts that drive `publish-prepare` / `publish` directly should keep
binding the sidecar as the last artifact-mutating step; re-running
`publish-prepare` is safe because it refreshes only the manifest digest.

## MTP manifest fields require a bound Tier 2 certificate

`publish-prepare` writes `mtp_acceptance_retention` and `mtp_measured_speedup`
into `axquant_manifest.json` only when the sidecar carries a Tier 2 certificate
binding. Without a bound Tier 2 certificate both fields stay `None`, even when
validation comparisons measured MTP acceptance and speed. Bind the Tier 2
certificate file (see above) and re-run `publish-prepare` to record the fields.

## MH1: `--allow-mtp-unmeasured`

`axquant plan` (and the measured manual / experimental-mix / joint reuse
paths) now fail closed when the measured sensitivity report carries MTP-scoped
modules whose `mtp_acceptance_loss` is entirely the probe's unmeasured `0.0`
marker while the active profile weights that metric. Campaigns that
intentionally plan from such reports must pass:

```bash
axquant plan ... --allow-mtp-unmeasured
```

The flag appends a warning to the plan instead of raising. Architecture-prior
plans and plans without MTP modules are unaffected and need no flag.
