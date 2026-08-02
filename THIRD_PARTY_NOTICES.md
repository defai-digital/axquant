# Third-Party Notices and Research References

AXQuant is an independent implementation. It does not vendor third-party quantizer source code,
tests, calibration data, model weights, or generated sensitivity metadata.

AXQuant itself is distributed under the MIT License. Direct runtime and optional MLX dependencies
retain their own licenses:

- MLX: <https://github.com/ml-explore/mlx>
- MLX-LM: <https://github.com/ml-explore/mlx-lm>
- Safetensors: <https://github.com/huggingface/safetensors>
- Hugging Face Hub client: <https://github.com/huggingface/huggingface_hub>
- Pydantic: <https://github.com/pydantic/pydantic>
- PyYAML: <https://github.com/yaml/pyyaml>
- structlog: <https://github.com/hynek/structlog>
- NumPy (used by measured probes and learned-quantization helpers):
  <https://github.com/numpy/numpy>

Research and public-method references:

- AWQ, activation-aware weight quantization: <https://arxiv.org/abs/2306.00978>
- GPTQ, post-training weight quantization: <https://arxiv.org/abs/2210.17323>
- MLX-LM learned quantization interfaces, including AWQ, DWQ, dynamic quantization, and GPTQ:
  <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LEARNED_QUANTS.md>
- CLP and multi-token prediction accuracy limits: <https://arxiv.org/abs/2606.10935>
- MXSens and mixed 4/6/8-bit sensitivity allocation: <https://arxiv.org/abs/2607.17733>

These references provide research context. Their inclusion does not claim implementation
equivalence, reproduced results, or endorsement.

mlx-optiq is an externally attributed comparison baseline only. AXQuant does not copy, translate,
decompile, or repackage its implementation, tests, prose, calibration data, or generated
metadata, and does not claim to be its official successor.
