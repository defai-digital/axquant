# Holo3-35B-A3B AXQ 4-bit-MTP — MTP acceleration Tier 2

**Verdict: not certified.**

This record covers
[`AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-4bit-MTP`](https://huggingface.co/AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-4bit-MTP)
commit
[`c048f577843225ac0545be5674b4d68b9a51dcf0`](https://huggingface.co/AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-4bit-MTP/tree/c048f577843225ac0545be5674b4d68b9a51dcf0).

## Why not certified

Shares the **same byte-identical grafted MTP sidecar** as the 6-bit-MTP pack
(`mtp.safetensors` SHA-256
`a4e12f8ea03b42a10359ff52ba7bf591cbfd8b98084886f63f8810ede62dee60`).

The 6-bit sibling under the Qwen **MoE exact** profile measured:

- exactness **pass**
- draft **accept rate 0%**
- decode speedup **~0.50×** (fail)

That is a **graft limit**, not a 4-bit-vs-6-bit layout issue. Fail closed.

See [holo3-35b-axq6-mtp-tier2.md](holo3-35b-axq6-mtp-tier2.md) for the full
decision table (runtime tuning vs retrain).

## Related

- Checkpoint Tier 1: [holo3-35b-axq4-mtp-tier1.md](holo3-35b-axq4-mtp-tier1.md)
- Sibling probe: [holo3-35b-axq6-mtp-tier2.md](holo3-35b-axq6-mtp-tier2.md)
- Evidence: [evidence/holo3-35b-axq6-mtp-tier2/](evidence/holo3-35b-axq6-mtp-tier2/)
- Machine-readable: [holo3-35b-axq4-mtp-tier2.json](holo3-35b-axq4-mtp-tier2.json)
