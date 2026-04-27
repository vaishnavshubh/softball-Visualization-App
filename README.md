# softball-Visualization-App

Purdue Softball analytics tools built with **Shiny for Python**:

- A **main dashboard app** (Trackman + Rapsodo + HitTrax, Comparison tab, Prediction tab, PDF downloads)
- A **standalone PDF report app** (upload a Trackman-format CSV → download a one-page PDF)

## Start here (full documentation)

- **Full project handoff / stakeholder doc**: [HANDOFF.md](HANDOFF.md)

## Apps in this repo

- **Main Shiny dashboard**: [app/app.py](app/app.py)
- **Standalone PDF Shiny app**: [pdf app/pdf_report.py](pdf%20app/pdf_report.py)
- **Shared PDF generation library (used by both apps)**: [app/pdf_report.py](app/pdf_report.py)

## Run locally

### Main dashboard

```bash
cd app
pip install -r requirements.txt
shiny run app.py
```

### Standalone PDF app

```bash
cd "pdf app"
pip install -r requirements.txt
shiny run pdf_report.py
```

## Data location (Trackman v3 tree)

- The main app loads CSVs from `app/data/v3/` by default (recursive).
- The repo also contains a top-level `v3/` folder used as a dev mirror in some setups.

## Deployment

- See [app/DEPLOY.md](app/DEPLOY.md) (Shinyapps.io via `rsconnect-python`, or VPS).
