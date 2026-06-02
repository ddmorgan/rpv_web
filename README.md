# RPV Predictor Web App

Small upload-based web app for reactor pressure vessel steel embrittlement predictions.

## Current Models

- `E900`: ASTM E900 formula ported from `RPV_model_benchmarking/models.py`
- `EONY`: EONY formula ported from `RPV_model_benchmarking/models.py`

The app reports benchmark uncertainty bands using the five-fold residual CSVs from the source repository. The one-sigma value is the standard deviation of prediction minus measured `Measured DT41J  [C]`; 95% bands use `prediction +/- 1.96 * sigma`.

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

Optional columns:

- `alloy`
- `model` with values such as `E900`, `EONY`, or `E900,EONY`
- `wt_percent_Si`
- `wt_percent_C`
- `Reactor Type`

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
