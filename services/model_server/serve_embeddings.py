import argparse
import os

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "intfloat/multilingual-e5-small"

app = FastAPI(title="Embeddings Service", version="0.1.0")

_model: SentenceTransformer | None = None
_model_name: str = DEFAULT_MODEL


class EmbedRequest(BaseModel):
    input: str | list[str]
    model: str = DEFAULT_MODEL


class EmbedObject(BaseModel):
    object: str = "embedding"
    index: int
    embedding: list[float]


class UsageInfo(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbedResponse(BaseModel):
    object: str = "list"
    data: list[EmbedObject]
    model: str
    usage: UsageInfo


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str


def get_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    global _model, _model_name
    if _model is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        _model_name = model_name
        _model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
    return _model


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return HealthResponse(status="ok", model=_model_name, device=device)


@app.post("/v1/embeddings", response_model=EmbedResponse)
@app.post("/embeddings", response_model=EmbedResponse)
def embeddings(req: EmbedRequest) -> EmbedResponse:
    texts = [req.input] if isinstance(req.input, str) else req.input

    try:
        model = get_model(_model_name)
        vecs = model.encode(texts, normalize_embeddings=True)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    total_tokens = sum(len(t.split()) for t in texts)
    return EmbedResponse(
        data=[
            EmbedObject(index=i, embedding=vec.tolist()) for i, vec in enumerate(vecs)
        ],
        model=_model_name,
        usage=UsageInfo(prompt_tokens=total_tokens, total_tokens=total_tokens),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Embeddings")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    get_model(args.model)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
