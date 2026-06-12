# Automated Retraining & Drift Monitoring

`scripts/retrain.py` retrains all models for a dataset, appends a metrics
snapshot to a JSONL history file, and compares the new metrics against the
trailing median of recent runs. Degradations surface as `model_drift` alerts
on the dashboard Alerts page and the `/hotels/{id}/alerts` API endpoint, and
as a nonzero exit code for schedulers.

## Usage

Run from the repository root:

```powershell
python scripts/retrain.py                     # demo dataset, force retrain
python scripts/retrain.py --hotel-id abc123   # one registered hotel (data/hotels/abc123/)
python scripts/retrain.py --data my.csv       # ad-hoc CSV (models under data/adhoc/)
python scripts/retrain.py --no-force          # don't clear the cache; snapshot current metrics
python scripts/retrain.py --history-path p.jsonl  # override the history location
```

A full forced retrain on the demo dataset takes several minutes (cancellation
hyperparameter tuning dominates). `--no-force` is near-instant when the cache
is current.

## Exit codes

| Code | Meaning | Scheduler action |
|------|---------|------------------|
| 0 | Retrained, no drift | none |
| 1 | Drift detected | notify / investigate |
| 2 | Execution error | notify / investigate |

## Metrics history

Snapshots are appended to a JSONL file that lives **beside** the models
directory (never inside it — force-retrain clears the models dir):

- demo dataset: `data/metrics_history.jsonl`
- registered hotels: `data/hotels/{hotel_id}/metrics_history.jsonl`
- Streamlit uploads: `data/streamlit_uploads/{id}/metrics_history.jsonl`

Each line records: timestamp, dataset id, model versions, demand/revenue
MAPE + MAE, cancellation accuracy / Brier / ECE / ROC-AUC, conformal interval
coverage (overall and per horizon bin), and any drift findings.

The dashboard's Validation page charts this history; the Alerts page raises a
`model_drift` card when the latest snapshot (younger than 45 days) contains
findings.

## Drift rules

Baseline = median of the 5 most recent snapshots. With fewer than 3 previous
snapshots, only absolute floors apply.

| Metric | Warning | Critical | Rationale |
|--------|---------|----------|-----------|
| demand/revenue MAPE | >1.20× baseline **and** +2.0pp | >1.40× **and** +4.0pp | 30-day holdout noise is ~1–2pp; the absolute floor stops flapping |
| MAPE absolute ceiling | >25% | — | ~2× the healthy 12–13% level |
| cancel accuracy | −3pp vs baseline | −6pp | holdout n≈24k → SE≈0.27pp; 3pp ≈ 11 SE |
| cancel Brier | >1.15× baseline | >1.30× | — |
| interval coverage | <0.85 | <0.80 | coverage SE on 30 days ≈ 5.5pp → 1 SE / 2 SE below the 90% nominal |
| per-bin coverage | <0.75 (warning only) | — | bins rest on 7–16 points |

Thresholds are overridable via the `drift_config` argument of
`src.monitoring.run_retrain` / `detect_drift`.

## Scheduling

### Windows Task Scheduler

```powershell
schtasks /create /sc weekly /d SUN /st 03:00 /tn HotelRetrain `
  /tr "cmd /c cd /d C:\dev\BusinessInsights && python scripts\retrain.py"
```

### GitHub Actions (cron)

```yaml
name: weekly-retrain
on:
  schedule:
    - cron: "0 3 * * 0"   # Sundays 03:00 UTC
jobs:
  retrain:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -r requirements.txt
      - run: python scripts/retrain.py   # job fails on exit 1 (drift) or 2 (error)
```

### Render cron job

In `render.yaml`, add a cron service:

```yaml
  - type: cron
    name: weekly-retrain
    env: python
    schedule: "0 3 * * 0"
    buildCommand: pip install -r requirements.txt
    startCommand: python scripts/retrain.py
```
