# Schema-contract migration: restored property schemas (F075)

Two frozen JSON Schema snapshots were rendered by a generator that dropped
annotation keys (`title`, `description`, `examples`, `deprecated`, `readOnly`,
`writeOnly`) at every depth, including inside `properties` maps. A model field
with one of those names therefore lost its property schema while staying in
`required`:

- `axquant.scoreboard.v1` — `ScoreboardReport.title`
- `axquant.reproduction.v3` — `ReproductionCommand.description`

A `required` entry with no matching `properties` entry is valid JSON Schema, so
nothing was rejected, but the published contract did not constrain that field
(any type passed validation).

## What changed

- The generator is name-scoped: noise keys are dropped as annotations, never as
  property or definition names. A render-time guard now fails closed if a
  snapshot would declare a `required` field it does not describe.
- **`axquant.scoreboard.v1` → `axquant.scoreboard.v2`.** The live
  `ScoreboardReport` now describes `title` again. The v1 model and snapshot stay
  loadable and byte-identical, and the v1 snapshot keeps its original rendering.
- **`axquant.reproduction.v3` → `axquant.reproduction.v4`.** Same for
  `ReproductionRecipe` and the nested `ReproductionCommand`.

No existing snapshot digest changed: `schemas/axquant.scoreboard.v1.schema.json`
and `schemas/axquant.reproduction.v3.schema.json` are untouched, and their
entries in `schemas/manifest.json` keep their SHA-256 (only the owning model
name moved to `axquant.schema.frozen_v1`).

## Operator action

- Newly generated scoreboards emit `axquant.scoreboard.v2` and newly generated
  reproduction recipes emit `axquant.reproduction.v4`. Note that
  `axquant scoreboard` writes `scoreboard.json` with the new literal.
- Older artifacts keep loading unchanged. Recipe consumers must go through
  `axquant.schema.loading.load_reproduction_recipe`, which dispatches on
  `schema_version`; loading a v3 recipe with the current-version model alone
  still fails closed, as before.
- Consumers that validate the complete envelope must accept the two new version
  literals before reading newly generated artifacts. Consumers that only read
  fields shared by both versions need no change.

## Not affected

The published v1/v3 snapshots remain under-constrained, exactly as published.
They are pinned by the `_legacy_noise_key_drop` marker on the frozen models in
`axquant/schema/frozen_v1.py`; do not remove it, or the frozen digests change.
