# Benchmark schema v2 migration

AXQuant benchmark artifacts use `axquant.benchmark-config.v2` and
`axquant.benchmark-result.v2` beginning with the Gemma 4 assistant MTP exact-profile update.
The v2 config adds `prompt_format`, whose values are `raw` and `chat-template`; its default is
`raw`.

The v1 config and result models remain available as `BenchmarkConfigV1` and
`BenchmarkResultV1`. Existing v1 artifacts remain valid and their frozen JSON Schema snapshots
are unchanged. New benchmark runs emit v2 artifacts. Consumers that only inspect fields shared by
both versions can continue doing so without transformation. Consumers that validate the complete
envelope must accept the v2 schema version before reading newly generated results.

Gemma 4 assistant MTP exact-profile evidence uses `chat-template`. This field is part of the
matched direct/MTP invariant so the two benchmark arms cannot silently tokenize prompts using
different formats.
