"""
FastAPI backend wrapping inference/serve.py HybridModelServer.
Exposes a single POST endpoint that accepts a coding problem and returns
generated code, which model was used (student/teacher), and latency.
"""

import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from inference.serve import HybridModelServer

app = FastAPI(
    title="Agent Distillation Compiler API",
    description="Hybrid student/teacher code generation with complexity routing.",
    version="1.0.0",
)

server: HybridModelServer = None


@app.on_event("startup")
def load_models():
    global server
    server = HybridModelServer()


class GenerateRequest(BaseModel):
    problem: str
    max_new_tokens: int = 512


class GenerateResponse(BaseModel):
    code: str
    route: str
    latency_seconds: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": server is not None}


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    if server is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    if not request.problem.strip():
        raise HTTPException(status_code=400, detail="Problem cannot be empty.")
    t0 = time.time()
    result = server.generate(request.problem, max_new_tokens=request.max_new_tokens)
    latency = time.time() - t0
    return GenerateResponse(
        code=result["code"],
        route=result["route"],
        latency_seconds=round(latency, 3),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)