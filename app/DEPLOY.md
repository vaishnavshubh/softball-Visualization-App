# Deploying the Softball Pitch Dashboard Online

Your app currently reads CSV files from a **path on your computer** (`V3_PATH`). For an online deployment, that folder is not available on the server. You have two approaches:

---

## Option 1: Deploy to Shinyapps.io (recommended, free tier)

[Shinyapps.io](https://www.shinyapps.io/) is Posit’s hosting for Shiny apps (including Shiny for Python).

### 1. Make the app use data that you can deploy

The server won’t see your Desktop. Either:

- **A) Bundle data with the app**  
  - Copy your `v3` folder (or the CSVs you need) **inside** the `app` folder, e.g. `app/data/v3/`.  
  - In `app.py`, set the path from the app directory, e.g.:

  ```python
  import os
  V3_PATH = os.path.join(os.path.dirname(__file__), "data", "v3")
  ```

  Then the same code works locally and on Shinyapps.io as long as you deploy the `app` folder **including** `data/v3/`.

- **B) Load from a URL**  
  - Upload CSVs to a public URL (e.g. Google Drive “share link”, S3, or a simple file server).  
  - Change the app so the “CSV file” dropdown lists those URLs and you use `pd.read_csv(url)` instead of a local path.  
  - This requires code changes to replace the current “glob under V3_PATH” logic with a fixed list of URLs.

For the least friction, use **A** and deploy with a `data/v3` folder.

### 2. Create a Shinyapps.io account

1. Go to [https://www.shinyapps.io/](https://www.shinyapps.io/).
2. Sign up (free account).
3. After login, go to **Account** → **Tokens**.
4. Create a token and copy the **secret** (you won’t see it again).

### 3. Install and configure rsconnect-python

In a terminal:

```bash
pip install rsconnect-python
```

Configure your account (use the account name and token from the shinyapps.io dashboard). Example for account **shubhvaishnav**:

```bash
rsconnect add --account shubhvaishnav --name shubhvaishnav --token YOUR_TOKEN --secret YOUR_SECRET
```

Replace `YOUR_TOKEN` and `YOUR_SECRET` with the values from shinyapps.io (Account → Tokens). The `--name` is a local nickname you’ll use when deploying; using your account name (e.g. `shubhvaishnav`) keeps it simple.

**If you see `SSL: CERTIFICATE_VERIFY_FAILED`:** Python on macOS often can’t find the CA certificates. Run:

```bash
pip install certifi
```

Then, in the **same terminal session**, set the certificate file and run `rsconnect add` again:

```bash
export SSL_CERT_FILE=$(python3 -c "import certifi; print(certifi.where())")
rsconnect add --account shubhvaishnav --name shubhvaishnav --token YOUR_TOKEN --secret YOUR_SECRET
```

(On some setups you may need `python` instead of `python3` in the `SSL_CERT_FILE` command.) You can add `export SSL_CERT_FILE=...` to your `~/.zshrc` so you don’t have to run it every time.

### 4. Deploy from the app directory

From the **project root** (parent of `app`), deploy the `app` folder. Use **`-n`** with the **same nickname** you used in step 3 (e.g. if you used `--name shubhvaishnav` when adding, use `-n shubhvaishnav`):

```bash
cd "/Users/shubhvaishnav/Desktop/Softball Phase 1"
rsconnect deploy shiny app -n shubhvaishnav --title "Softball Pitch Dashboard"
```

After a successful deploy, your app will be at **https://shubhvaishnav.shinyapps.io/softball-pitch-dashboard/** (or a similar URL based on the title). The `--title` is the app name shown on shinyapps.io; the URL slug is derived from it.

If you see **"Python version constraint missing"**, the app folder now includes a `.python-version` file (e.g. `3.11`) so Connect knows which Python to use. Redeploy after that.

If you put data in `app/data/v3`, make sure that folder is present when you run this so it’s included in the deploy.

After it finishes, the command prints a URL like `https://YOUR_ACCOUNT.shinyapps.io/softball-dashboard/`. Share that link; anyone can open it in a browser.

### 5. Redeploy after changes

After you change code or data, run the same deploy command again:

```bash
rsconnect deploy shiny app --name softball-dashboard --title "Softball Pitch Dashboard"
```

---

## Option 2: Run on a VPS (e.g. DigitalOcean, AWS, Linode)

If you want the server to read from a **folder on the server** (e.g. you upload `v3` once via SSH/SFTP):

1. Rent a small Linux server (e.g. DigitalOcean Droplet, AWS EC2).
2. Install Python 3, then: `pip install -r requirements.txt` (and any system deps).
3. Copy your `app` folder and the `v3` data to the server.
4. Set `V3_PATH` in `app.py` to the path where `v3` lives on the server (e.g. `/var/app/data/v3`).
5. Run the app so it listens on `0.0.0.0` so the internet can reach it:

   ```bash
   shiny run app.py --host 0.0.0.0 --port 8000
   ```

6. Optionally put Nginx (or Caddy) in front and use a domain and HTTPS (e.g. Let’s Encrypt).

This gives you full control but requires server admin. Sharing “the link” then means sharing `http://YOUR_SERVER_IP:8000` (or your domain).

---

## Summary

| Goal | Approach |
|------|----------|
| Easiest “share a link” | Shinyapps.io + bundle `data/v3` inside `app` (Option 1A). |
| Data already at a URL | Shinyapps.io + change app to load from URLs (Option 1B). |
| Server has the files on disk | VPS + set `V3_PATH` and run with `--host 0.0.0.0` (Option 2). |

If you tell me which option you want (e.g. “use Option 1A and bundle data”), I can give exact steps for your project layout or suggest the minimal code change for `V3_PATH` so the same app works locally and on Shinyapps.io.
