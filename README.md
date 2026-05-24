# Residual Tail-Guided Flow Matching (RT-FM)

Anonymous code release for reproducing the final experiments.

RT-FM is a tail-aware training strategy for time-series anomaly detection. The
code trains a temporal VAE on nominal windows, removes amplitude-explainable
latent tail variation, and uses the residual tail score to softly reweight the
Flow Matching objective.

## Layout

```text
anonymous_submission_code/
  README.md
  requirements.txt
  src/
    run_univariate_ucr.py
    run_multivariate.py
    summarize_multivariate.py
    make_false_positive_visual.py
  scripts/
    run_ucr_full.ps1
    run_multivariate_clean.ps1
```

## Installation

Use Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

CUDA is used automatically when available.

## Data

Datasets are not redistributed with this anonymous code release. Place them
under `data/` before running the scripts.

Expected layout:

```text
data/
  UCR_TimeSeriesAnomalyDatasets2021/
    FilesAreInHere/
      UCR_Anomaly_FullData/
        *.txt
  multivariate/
    GECCO/
    SWAN/
    PSM/
    SMD/
    MSL/
    SMAP/
```

The UCR runner expects standard UCR anomaly filenames containing the train split
and anomaly interval metadata. The multivariate runner expects train/test arrays
and labels in the multivariate benchmark folders used by the scripts.

## Full Runs

Run the univariate UCR benchmark:

```powershell
.\scripts\run_ucr_full.ps1
```

Run the clean multivariate benchmark:

```powershell
.\scripts\run_multivariate_clean.ps1
```


## Direct Commands

The scripts are thin wrappers around:

```powershell
python src\run_univariate_ucr.py --out outputs\ucr_full\ucr_results.csv
```

```powershell
python src\run_multivariate.py `
  --output-dir outputs\multivariate_clean `
  --datasets DC_GECCO DC_SWAN MSL SMAP PSM SMD `
  --models uniform amplitude raw_radius_b1_w0.2 raw_radius_b2_w0.2 residual_tail_b1_w0.2 residual_tail_b2_w0.2 residual_tail_b4_w0.2 `
  --seeds 42 123
```

## Outputs

The univariate runner writes one CSV row per dataset, seed, and model:

```text
outputs/ucr_full/ucr_results.csv
```

The multivariate runner writes:

```text
outputs/<run_name>/multivariate_batch_results.csv
outputs/<run_name>/multivariate_batch_summary.csv
outputs/<run_name>/<dataset>_<model>_seed<seed>/artifacts.npz
```

The saved artifacts include timestamp/window scores, labels, rare-normal masks,
and reconstruction diagnostics used for the mechanism figure.

## Metrics

The main reported metrics are timestamp-level AP/AUPRC, VUS-PR, best F1,
FP@95R, rare-normal FP@95R, and event-level F1. Window scores are converted to
timestamp scores using mean-overlap aggregation.
