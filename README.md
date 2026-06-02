# RPV Predictor Web App - Domain Output Version

Small upload-based web app for reactor pressure vessel steel embrittlement predictions.

## Current Models

- `E900`: ASTM E900 formula ported from `RPV_model_benchmarking/models.py`
- `EONY`: EONY formula ported from `RPV_model_benchmarking/models.py`
- `GBR`: trained gradient boosting regressor from the upstream `model_files/GBR/fullfit` artifacts
- `GKRR`: trained Gaussian kernel ridge regressor from the upstream `model_files/GKRR/fullfit` artifacts

The app reports benchmark uncertainty bands from the source repository residual summaries. The one-sigma value is the standard deviation of prediction minus measured `Measured DT41J  [C]`; 95% bands use `prediction +/- 1.96 * sigma`.

This version also reports a domain/applicability check for every prediction. `Domain` is `In domain` when the uploaded row falls within the benchmark feature ranges used by the source repository, and `Out of domain` when one or more numeric fields are outside those ranges or categorical fields are not represented. The output includes the number of out-of-domain fields and a text description of which fields are outside range.

## Input Columns

CSV, XLSX, and JSON uploads are supported. Required columns:

- `Product Form`
- `temperature_C`
- `wt_percent_Cu`
- `wt_percent_Ni`
- `wt_percent_Mn`
- `wt_percent_P`
- `flux_n_cm2_sec`
- `fluence_n_cm2`

For `GBR`, include `Reactor Type` with `PWR` or `BWR`. If it is omitted, the app treats it as `PWR`.

For `GKRR`, include `wt_percent_Si` and `wt_percent_C`. The app computes `at_percent_*` values from weight percent using Fe as the balance element. If `effective_fluence` is omitted, the app treats it as equal to `fluence_n_cm2`.

Optional columns:

- `alloy`
- `model` with values such as `E900`, `EONY`, `GBR`, `GKRR`, or `E900,EONY,GBR,GKRR`
- `wt_percent_Si`
- `wt_percent_C`
- `Reactor Type`
- `effective_fluence`
- `at_percent_Cu`, `at_percent_Ni`, `at_percent_Mn`, `at_percent_P`, `at_percent_Si`, `at_percent_C`

See `examples/sample_input.csv`.

## Run Locally

```bash
python app.py
```

Then open `http://127.0.0.1:8000`.

If your Python environment does not already have the packages:

```bash
python -m pip install -r requirements.txt
```

## Deploy

The included `Dockerfile` works on Docker-based hosts such as Render, Fly.io, Railway, and many institutional servers:

```bash
docker build -t rpv-predictor .
docker run -p 8000:8000 rpv-predictor
```

For Heroku-style hosts, the `Procfile` starts the same server.
