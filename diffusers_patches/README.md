Patches to diffusers 0.31.0 (site-packages) required for GE-Sim-V2 to run.
Reapply after any `pip install diffusers==0.31.0`.

1. normalization.py (~line 269, AdaLayerNormSingle.forward):
   **added_cond_kwargs  ->  **(added_cond_kwargs or {})

2. embeddings.py (~line 1563, PixArtAlphaCombinedTimestepSizeEmbeddings.forward):
   def forward(self, timestep, resolution, aspect_ratio, batch_size, hidden_dtype):
   ->
   def forward(self, timestep, resolution=None, aspect_ratio=None, batch_size=None, hidden_dtype=None):
   (safe: both args only read when use_additional_conditions=True; GE-Sim sets it False)

Also: Liger disabled in configs/gesim_v2.yaml (liger_norm/liger_layernorm/triton_rope=false)
Env: torch 2.4.1+cu124, diffusers 0.31.0, transformers 4.46.3, accelerate 0.34.2, liger-kernel 0.8.0
See working_findings.lock for full pip freeze.
