# Azure Deployment (PoC)

This PoC can be deployed to Microsoft Azure in two parts:
1) FastAPI prediction API
2) Static web UI

## Option A: Deploy API to Azure App Service (Python)

1. Create App Service (Linux, Python 3.11+)
2. Configure startup command:
   `uvicorn app:app --host 0.0.0.0 --port 8000`
3. Set environment variables:
   - `ALLOWED_ORIGINS` = your web app URL (or `*` for PoC)
4. Deploy code (ZIP deploy or GitHub Actions)
5. Confirm `/docs` works

## Option B: Deploy web UI to Azure Static Web Apps

- Upload the `web/` folder as a static site.
- Edit `web/config.js` and set:
  `window.API_BASE = "https://YOUR-API.azurewebsites.net"`

## Monitoring
- Enable Application Insights on the App Service.
- Capture screenshots of request logs and latency for evidence.

