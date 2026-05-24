$Out = "outputs\multivariate_batch_clean"
New-Item -ItemType Directory -Force $Out | Out-Null

python src\run_multivariate.py `
  --output-dir $Out `
  --datasets DC_GECCO DC_SWAN MSL SMAP PSM SMD `
  --models uniform amplitude raw_radius_b1_w0.2 raw_radius_b2_w0.2 residual_tail_b1_w0.2 residual_tail_b2_w0.2 residual_tail_b4_w0.2 `
  --seeds 42 123

