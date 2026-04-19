# Deploy the Softball Dashboard to Google Cloud Run

This runs the Shiny app in a container. **BigQuery uses the Cloud Run service account** (metadata server) — **no downloaded JSON key** needs to be inside the image if you attach a runtime service account with BigQuery permissions.

**Prerequisites**

- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) (`gcloud`) installed and logged in: `gcloud auth login`
- A GCP project (e.g. `softball-492603`) with billing enabled (if required by your org)
- Permission to enable APIs, create service accounts, and deploy Cloud Run (roles such as **Owner**, **Editor**, or a custom role with Cloud Run Admin + IAM + Artifact Registry)

Replace placeholders:

| Placeholder | Example |
|-------------|---------|
| `PROJECT_ID` | `softball-492603` |
| `REGION` | `us-central1` |
| `SERVICE_NAME` | `softball-dashboard` |
| `RUNTIME_SA` | `softball-cloud-run@PROJECT_ID.iam.gserviceaccount.com` |

---

## Step 1 — Set the project

```bash
gcloud config set project PROJECT_ID
```

---

## Step 2 — Enable required APIs

```bash
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  bigquery.googleapis.com
```

---

## Step 3 — Create a runtime service account (no key download)

This identity is **attached to Cloud Run**. The app calls BigQuery as this account.

```bash
gcloud iam service-accounts create softball-cloud-run \
  --display-name="Softball dashboard (Cloud Run)"
```

Note the email:

`softball-cloud-run@PROJECT_ID.iam.gserviceaccount.com`

Grant **BigQuery** access (adjust if your org uses custom roles):

```bash
export PROJECT_ID=PROJECT_ID
export RUNTIME_SA="softball-cloud-run@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${RUNTIME_SA}" \
  --role="roles/bigquery.dataViewer"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${RUNTIME_SA}" \
  --role="roles/bigquery.jobUser"
```

If data lives in another project, grant the same roles **on that project** or on the datasets via BigQuery IAM.

---

## Step 4 — Artifact Registry (Docker repository)

```bash
export REGION=us-central1
export AR_REPO=cloud-run

gcloud artifacts repositories create "$AR_REPO" \
  --repository-format=docker \
  --location="$REGION" \
  --description="Cloud Run images" \
  2>/dev/null || true
```

Configure Docker auth:

```bash
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
```

---

## Step 5 — Build the image

From the **repository root** (parent of `app/`):

```bash
export PROJECT_ID=PROJECT_ID
export REGION=us-central1
export AR_REPO=cloud-run
export IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/softball-dashboard:latest"

gcloud builds submit --tag "$IMAGE" app/
```

This uses `app/Dockerfile` and `app/.dockerignore`.

Test locally (optional):

```bash
docker run --rm -p 8080:8080 \
  -e PORT=8080 \
  -v "$HOME/.config/gcloud/application_default_credentials.json":/tmp/adc.json:ro \
  -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/adc.json \
  "$IMAGE"
```

Open http://localhost:8080 . (Local test uses your user ADC; on Cloud Run, credentials come from the **attached service account**.)

---

## Step 6 — Deploy to Cloud Run

```bash
export SERVICE_NAME=softball-dashboard
export RUNTIME_SA="softball-cloud-run@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud run deploy "$SERVICE_NAME" \
  --image "$IMAGE" \
  --region "$REGION" \
  --platform managed \
  --service-account "$RUNTIME_SA" \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --timeout 300 \
  --min-instances 0 \
  --max-instances 3
```

- **`--allow-unauthenticated`**: anyone with the URL can open the app. For campus-only use, remove it and use **IAM** (“Require authentication”) or **Identity-Aware Proxy** (more setup).
- **`--memory`**: increase if BigQuery result sets are large.
- **`--min-instances 1`**: reduces cold starts (adds cost).

When the command finishes, note the **Service URL**.

---

## Step 7 — Verify BigQuery from Cloud Run

1. Open the Service URL in a browser.
2. If the team list is empty or errors appear, check **Cloud Run → your service → Logs**.
3. Common fixes:
   - Runtime SA missing **BigQuery** roles or **dataset-level** access
   - Wrong `BQ_PROJECT` in `app.py` if data is in another project

---

## Redeploy after code changes

```bash
gcloud builds submit --tag "$IMAGE" app/
gcloud run deploy "$SERVICE_NAME" --image "$IMAGE" --region "$REGION"
```

---

## Who deploys the container?

Deploying with `gcloud` uses **your user** identity (or CI). That is separate from the **runtime** service account. Your school may allow `gcloud auth login` with a university Google account without issuing **service account keys**.

---

## Dockerfile notes

- **`GOOGLE_APPLICATION_CREDENTIALS=`** is cleared so the client does not look for a key file and uses **metadata** credentials on Cloud Run.
- **`PORT`** is honored as required by Cloud Run.

---

## Troubleshooting

| Issue | What to check |
|-------|----------------|
| `403` / `Access Denied` on BigQuery | IAM roles on `RUNTIME_SA`; dataset permissions |
| Container exits / import errors | Cloud Run logs; `requirements.txt` complete |
| Cold start slow | `--min-instances 1` or smaller initial queries |
| Wrong Python | Image uses Python 3.12 (see `app/Dockerfile`) |
