"""Optional Ollama integration. Statistics are computed outside the LLM."""
import requests
DEFAULT_MODEL="llama3.2"

def ollama_status(base_url="http://localhost:11434"):
    try:
        r=requests.get(f"{base_url.rstrip('/')}/api/tags",timeout=3); r.raise_for_status()
        return {"available":True,"models":[m.get("name","") for m in r.json().get("models",[])]}
    except requests.RequestException as exc:
        return {"available":False,"models":[],"message":str(exc)}

def explain_findings(findings,model=DEFAULT_MODEL,base_url="http://localhost:11434"):
    status=ollama_status(base_url)
    if not status["available"]:
        return {"available":False,"model":model,"text":"Ollama is not reachable. Start Ollama and pull a model to enable AI explanations. Statistical analysis remains available without the LLM.","details":status.get("message","")}
    models = status.get("models", [])
    if models and model not in models and not any(m.startswith(f"{model}:") for m in models):
        return {"available":False,"model":model,"text":f"Model '{model}' is not installed in Ollama. Available: {', '.join(models)}. Run 'ollama pull {model}' in your terminal.","details":"Model not found in local Ollama repository."}
    if not models:
        return {"available":False,"model":model,"text":f"No local models found in Ollama. Run 'ollama pull {model}' in PowerShell to install '{model}'.","details":"Local model list is empty."}
    prompt=("You are DataMind AI, a careful data analyst. Explain supplied computed analysis. Use only values present in JSON; do not invent numbers or claim causation from correlation. Separate observations, possible explanations, recommended checks, uncertainty, and limitations. Return Markdown headings: Executive summary, Evidence, Possible explanations, Recommended next steps, Limitations.\n\nCOMPUTED FINDINGS JSON:\n"+str(findings))
    try:
        r=requests.post(f"{base_url.rstrip('/')}/api/generate",json={"model":model,"prompt":prompt,"stream":False},timeout=120); r.raise_for_status()
        return {"available":True,"model":model,"text":r.json().get("response","").strip()}
    except requests.RequestException as exc:
        details = str(exc)
        if getattr(exc, "response", None) is not None:
            try:
                err = exc.response.json().get("error")
                if err:
                    details = err
            except Exception:
                pass
        return {"available":False,"model":model,"text":"Local model request failed. Check the Ollama service and model name.","details":details}
