# SmartCloud Insights (Platform)

Cloud-based intelligent web applications using Machine Learning:
- **Student performance** (at-risk prediction + study resources)
- **Product demand** (high-demand probability + actions)
- **Health trends** (higher-risk probability + guidance links)

This repository contains a **proof-of-concept** implementation:
- Offline training scripts create versioned model artefacts (`models/*.joblib`)
- A FastAPI service serves predictions (`/{module}/predict`)
- A lightweight web UI consumes the API (`web/index.html`)

## Datasets (real datasets)

This proof-of-concept uses **real, public UCI datasets** for all three modules:

- **Student performance:** UCI *Student Performance* dataset (semicolon-separated CSV).
- **Product/service demand:** UCI *Bike Sharing* dataset (daily bike rental demand + weather/season features).
- **Health risk indicator:** UCI *Early Stage Diabetes Risk Prediction* dataset (signs/symptoms + class label).

### Download (recommended)
Run the helper script to download and prepare the required datasets into `data/`:

```powershell
python download_datasets.py
```

If you prefer manual download, the UCI dataset pages are referenced in the report and in the script.

## Quick start (Windows PowerShell)

```powershell
cd SmartCloudInsights_PoC
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Download real datasets (Bike Sharing + Early Stage Diabetes)
python download_datasets.py

# Train all modules
python train.py --module all
python make_defaults.py --module all

# Run API
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Open API docs:
- http://127.0.0.1:8000/docs

Run the web UI:
- Open `web/index.html` in the browser (or serve it with a simple local web server).

## API
- `GET /health`
- `GET /modules`
- `GET /{module}/features`
- `POST /{module}/predict` with JSON body: `{ "features": { ... } }`

Modules: `student`, `demand`, `health`

## Notes
- The student dataset still needs to be placed as `data/student.csv` (UCI format with semicolon separator).
- If the helper download fails (e.g., no internet), demand/health training falls back to small synthetic data and prints a warning.

