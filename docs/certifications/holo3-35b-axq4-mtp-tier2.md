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
The 6-bit sibling soft probe on `df-macstudio-m2` saw exactness pass but
decode speedup ~0.43× (fail). No separate formal authorizing scoreboard was
closed for 4-bit-MTP; fail closed.

Product default remains **direct decode**.

## Related

- Checkpoint Tier 1: [holo3-35b-axq4-mtp-tier1.md](holo3-35b-axq4-mtp-tier1.md)
- Sibling soft probe: [holo3-35b-axq6-mtp-tier2.md](holo3-35b-axq6-mtp-tier2.md)
- Machine-readable: [holo3-35b-axq4-mtp-tier2.json](holo3-35b-axq4-mtp-tier2.json)
