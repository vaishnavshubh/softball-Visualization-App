# Softball Pitch Dashboard (Shiny for Python)

Dashboard to view pitch usage, location, and movement by pitcher from Trackman CSV files.

## Setup

```bash
cd app
pip install -r requirements.txt
```

## Run

```bash
shiny run app.py
```

Then open the URL shown (e.g. http://127.0.0.1:8000) in your browser.

## Data path

CSV files are loaded from the `v3` folder. Edit `V3_PATH` at the top of `app.py` if your data lives elsewhere.

## Usage

1. **CSV file**: Choose a game/file from the dropdown (all `*.csv` under `v3` recursively).
2. **Pitcher**: Choose a pitcher; the list updates when you change the CSV.
3. The 2x2 dashboard shows: Pitch Usage % (pie), Pitch Locations, Usage Table, Pitch Movement.
