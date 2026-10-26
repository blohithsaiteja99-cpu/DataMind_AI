import sys
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

_current_dir = Path(__file__).resolve().parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from dataminds.analysis import build_investigation, run_hypothesis_test
from dataminds.ai import explain_findings, ollama_status
from dataminds.io import read_dataset
from dataminds.report import markdown_report

app=FastAPI(title="DataMind AI API",description="Dataset profiling, anomaly detection, statistical tests, and local Ollama explanations.",version="1.0.0")

@app.get("/")
def root(): return {"project":"DataMind AI","status":"ok","docs":"/docs"}
@app.get("/health")
def health(): return {"status":"healthy"}
@app.get("/ollama/status")
def get_ollama_status(): return ollama_status()

@app.post("/analyze")
async def analyze(file: UploadFile=File(...),target: str|None=Form(default=None)):
    try:
        df=read_dataset(file.filename or "upload.csv",await file.read())
        result=build_investigation(df,target=target if target in df.columns else None)
        result["dataset_name"]=file.filename
        return result
    except ValueError as e: raise HTTPException(status_code=400,detail=str(e))
    except Exception as e: raise HTTPException(status_code=422,detail=f"Could not analyze file: {e}")

@app.post("/hypothesis-test")
async def hypothesis_test(file: UploadFile=File(...),target: str=Form(...),group: str=Form(...)):
    try: return run_hypothesis_test(read_dataset(file.filename or "upload.csv",await file.read()),target,group)
    except ValueError as e: raise HTTPException(status_code=400,detail=str(e))
    except Exception as e: raise HTTPException(status_code=422,detail=f"Could not run test: {e}")

@app.post("/explain")
async def explain(file: UploadFile=File(...),target: str|None=Form(default=None),model: str=Form(default="llama3.2")):
    try:
        df=read_dataset(file.filename or "upload.csv",await file.read())
        return explain_findings(build_investigation(df,target=target if target in df.columns else None),model=model)
    except ValueError as e: raise HTTPException(status_code=400,detail=str(e))
    except Exception as e: raise HTTPException(status_code=422,detail=f"Could not explain file: {e}")

@app.post("/report",response_class=PlainTextResponse)
async def report(file: UploadFile=File(...),target: str|None=Form(default=None)):
    try:
        df=read_dataset(file.filename or "upload.csv",await file.read())
        return markdown_report(build_investigation(df,target=target if target in df.columns else None),file.filename or "uploaded dataset")
    except ValueError as e: raise HTTPException(status_code=400,detail=str(e))
    except Exception as e: raise HTTPException(status_code=422,detail=f"Could not generate report: {e}")
