# Independent Implementation Policy

AXQuant is developed as an independent product and method.

Permitted inputs:

- public research papers and their published equations;
- public MLX and MLX-LM APIs and documentation;
- public model formats and Safetensors metadata;
- calibration data supplied or licensed by the AXQuant operator;
- measurements produced by AXQuant or explicitly imported with provenance.

Prohibited inputs:

- copied, translated, decompiled, or mechanically transformed mlx-optiq implementation code;
- mlx-optiq tests, calibration samples, internal manifests, or generated sensitivity tables;
- claims that AXQuant is an official successor to mlx-optiq;
- benchmark or MTP acceptance claims without recorded evidence.

Every release-quality manifest must record:

- immutable source model identity and revision;
- calibration dataset identity and digest;
- metric definitions and objective weights;
- candidate precisions, methods, and hardware constraints;
- per-tensor assignments and their reasons;
- nominal and storage-adjusted bits per weight;
- MTP policy and acceptance measurements;
- validation results and artifact checksums.

AXQuant may compare against public checkpoints as external baselines. Imported results must remain
attributed and cannot be presented as measurements produced by AXQuant.

Baseline feasibility audits are limited to the public load contract: config, weight index,
index-referenced Safetensors, root MTP bundle/provenance, tokenizer presence, and runtime
readiness. Unindexed auxiliary files are excluded and no external sensitivity tables or planner
decisions are imported.
