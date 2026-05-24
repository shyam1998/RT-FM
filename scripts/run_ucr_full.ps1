$OutDir = "outputs\ucr_full"
New-Item -ItemType Directory -Force $OutDir | Out-Null
$Out = Join-Path $OutDir "ucr_results.csv"

python src\run_univariate_ucr.py `
  --data-dir "data\UCR_Anomaly_FullData" `
  --all-ucr `
  --seeds 42 123 `
  --out $Out `
  --raw-boosts 4 8 `
  --residual-boosts 1 2 4 8 `
  --fm-iterations 30000 `
  --vae-iterations 30000 `
  --eval-batch 10000 `
  --ode-method rk4 `
  --ode-steps 4
