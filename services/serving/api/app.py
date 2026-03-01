from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from services.serving.inference.engine import InferenceEngine

app = FastAPI(title="llm-stack-serving")
engine = InferenceEngine("local-300m")


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 32


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/version")
async def version() -> dict[str, str]:
    return {"model_version": "0.1.0"}


@app.post("/generate")
async def generate(req: GenerateRequest) -> dict[str, str]:
    text = await engine.generate(req.prompt, req.max_tokens)
    return {"text": text}
