# DataMind AI
**Autonomous Data Investigation Platform** built with Python, Pandas, SciPy, Scikit-learn, FastAPI, Streamlit, Plotly, and optional local Ollama explanations.

## Features
- CSV and Excel upload
- Automated dataset profile, missing values, duplicates, column statistics and data-quality warnings
- Correlation ranking and interactive heatmap
- Isolation Forest anomaly detection
- Low-vs-high target association investigation using Mann–Whitney U tests
- Welch's t-test for two groups and one-way ANOVA for more than two groups
- Optional local Ollama explanation of computed findings
- FastAPI endpoints with Swagger docs
- Downloadable Markdown report, JSON findings and anomaly CSV
- Synthetic sample dataset and pytest tests

## Windows installation
Prerequisites: Python 3.11 or 3.12 (64-bit recommended). Ollama is optional: https://ollama.com/download/windows

Open PowerShell in the extracted project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If activation is blocked for this PowerShell session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Start dashboard:

```powershell
python -m streamlit run app.py
```

Open the local URL Streamlit prints, usually `http://localhost:8501`.

Start API in a **second PowerShell window** from the project directory (activate `.venv` there too):

```powershell
uvicorn api:app --reload
```

Open `http://127.0.0.1:8000/docs`.

## Optional Ollama local AI
Install Ollama, then run:

```powershell
ollama pull llama3.2
ollama run llama3.2
```

Keep the Ollama service running. In the dashboard, choose **Generate AI explanation**. Model download requires internet access. The rest of the app works without Ollama.

## Sample dataset
`data/sample_restaurant_metrics.csv` contains synthetic restaurant metrics, intentionally including a few missing values and unusual observations for demonstration. It is not real business data.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Project identity |
| GET | `/health` | Health check |
| GET | `/ollama/status` | Ollama availability |
| POST | `/analyze` | Profiling, correlations, anomaly detection and optional target investigation |
| POST | `/hypothesis-test` | Group comparison test |
| POST | `/explain` | Optional Ollama explanation |
| POST | `/report` | Markdown report |

PowerShell API example:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/analyze" `
  -F "file=@data/sample_restaurant_metrics.csv" `
  -F "target=monthly_revenue"
```

## Tests

```powershell
pytest -q
```

## Statistical limitations
- Isolation Forest flags unusual records; it does not prove fraud or errors.
- Correlation and group differences do not establish causality.
- Root-cause investigation ranks associations between features and low/high target groups. It is not a causal inference engine.
- Welch's t-test and ANOVA rely on assumptions; inspect sampling and distributions.
- Statistical significance does not automatically imply practical importance.
- The optional LLM explains computed findings; verify its narrative against the data.
- Demo limits: 25 MB upload, 250,000 rows and 200 columns.
- Do not expose the API publicly without authentication, authorization and deployment hardening.

## Resume-ready project description
**DataMind AI — Autonomous Data Investigation Platform**  
Developed a Python data investigation application using Pandas, SciPy and Scikit-learn for automated dataset profiling, data-quality checks, correlation analysis, Isolation Forest anomaly detection and evidence-based group comparisons. Built an interactive Streamlit dashboard and FastAPI REST endpoints, with optional Ollama local-LLM explanations and downloadable Markdown/JSON reports. Emphasized reproducibility, statistical caveats, and separation of computed evidence from AI-generated narrative.

Only include performance metrics on your resume after measuring them on a documented test dataset.
