# Softball Pitch Dashboard (Shiny for Python)

Pitch usage, location, and movement dashboard. **Pitch data is loaded from Google BigQuery** (`softball_curated.fact_pitch`); teams from `dim_team`.

**Deploy on Google Cloud Run (recommended if you cannot use service account key files):** see the step-by-step guide at [`docs/CLOUD_RUN_DEPLOY.md`](../docs/CLOUD_RUN_DEPLOY.md) and the `app/Dockerfile`.

## Setup

```bash
cd app
pip install -r requirements.txt
```

## Run locally

```bash
shiny run app.py
```

Open the URL shown (e.g. http://127.0.0.1:8000).

**BigQuery auth (local):** `gcloud auth application-default login` (same Google account that can access the project), or set `GCP_SERVICE_ACCOUNT_JSON` to the full service-account JSON string.

## Deploy to shinyapps.io

shinyapps.io supports **Python 3.7–3.12**. This repo includes `app/.python-version` with `3.12` so the bundle requests a supported runtime.

1. Create an account at [shinyapps.io](https://www.shinyapps.io) and open **Account → Tokens** → **Show** to copy the `rsconnect add` command, or run:

   ```bash
   pip install rsconnect-python
   rsconnect add --account <ACCOUNT> --name shinyapps --token <TOKEN> --secret <SECRET>
   ```

2. **If you see `CERTIFICATE_VERIFY_FAILED` or `SSL` errors** (common on macOS), install **certifi** for the **same Python** that runs `rsconnect`, then point SSL at it:

   ```bash
   # Install certifi (pick ONE python — the one that runs `which rsconnect` / your default `python3`)
   python3 -m pip install certifi

   export SSL_CERT_FILE="$(python3 -c 'import certifi; print(certifi.where())')"
   export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
   ```

   If `python3` still has no certifi, use the **Python.org** 3.13 binary explicitly:

   ```bash
   /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pip install certifi
   export SSL_CERT_FILE="$(/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -c 'import certifi; print(certifi.where())')"
   export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
   ```

   You can add the two `export` lines to `~/.zshrc` so every new terminal picks them up.

3. From the **`app`** directory (where `app.py` and `requirements.txt` live):

   ```bash
   cd app
   rsconnect deploy shiny . --name shinyapps --title "Softball Dashboard" \
     --exclude "data/v3" \
     --exclude "rsconnect-python" \
     --exclude ".venv"
   ```

   (`rsconnect-python` already skips common venvs; `--exclude` keeps local `data/v3` out of the bundle if present.)

4. **BigQuery on shinyapps.io:** the shinyapps.io **dashboard does not offer environment variables** for Python apps (unlike Posit Connect). Use **one** of these:

   - **Recommended for shinyapps.io:** put a service account JSON file at **`app/gcp_sa.json`** (same folder as `app.py`). The file is listed in **`.gitignore`** so it is not committed. When you deploy, **include the file in the bundle** from your machine:

     ```bash
     cd app
     rsconnect deploy shiny . --name shubhvaishnav --title "Softball Dashboard" \
       gcp_sa.json \
       --exclude "data/v3" --exclude "rsconnect-python" --exclude ".venv"
     ```

     Create the key in GCP Console (IAM → Service accounts → Keys) only if your org allows it; grant **BigQuery** access to project `softball-492603`.

   - **Other hosts (e.g. Cloud Run, Posit Connect):** set **`GCP_SERVICE_ACCOUNT_JSON`** to the full JSON string in that platform’s secret/env UI, or mount a credentials file per their docs.

   - **Local dev:** `gcloud auth application-default login` (no `gcp_sa.json` needed).

5. Optional: deploy with a specific Python (if needed):

   ```bash
   rsconnect deploy shiny . --name shinyapps --title "Softball Dashboard" --python /path/to/python3.12
   ```

## Usage

1. Choose **Team**, **date range**, and **player** in the sidebar (filters apply to BigQuery results).
2. Home tab: usage, location, movement, and summary for the selected pitcher or batter.
