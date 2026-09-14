import argparse
import os
from pathlib import Path

# Prevent PyTorch MPS from hoarding cached Metal memory
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder
import torch
import uvicorn

DEFAULT_ADAPTER_PATH = str(
    Path(__file__).resolve().parents[2]
    / "scripts/kleine_anfragen/adapters_reranker/final"
)

app = FastAPI(title="BGE Cross-Encoder Reranker Service", version="0.1.0")

_model: CrossEncoder | None = None
_model_path: str = DEFAULT_ADAPTER_PATH


class RerankRequest(BaseModel):
    query: str
    documents: list[str] = Field(default_factory=list)
    top_n: int = 5


class RerankItem(BaseModel):
    index: int
    relevance_score: float


class RerankResponse(BaseModel):
    results: list[RerankItem]


class HealthResponse(BaseModel):
    status: str
    model_path: str
    device: str


def get_model(model_path: str = DEFAULT_ADAPTER_PATH) -> CrossEncoder:
    global _model, _model_path
    if _model is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        _model_path = model_path
        _model = CrossEncoder(model_path, num_labels=1, max_length=512, device=device)
    return _model


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return HealthResponse(status="ok", model_path=_model_path, device=device)


@app.post("/v1/rerank", response_model=RerankResponse)
@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest) -> RerankResponse:
    if not req.documents:
        return RerankResponse(results=[])

    try:
        model = get_model(_model_path)
        pairs = [(req.query, doc) for doc in req.documents]
        scores = model.predict(pairs)
        ranked = sorted(
            enumerate(scores), key=lambda item: float(item[1]), reverse=True
        )[: req.top_n]
        return RerankResponse(
            results=[
                RerankItem(index=idx, relevance_score=float(score))
                for idx, score in ranked
            ]
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Cross-Encoder Reranker")
    parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8001, help="Port (default: 8001)")
    parser.add_argument(
        "--model-path",
        default=DEFAULT_ADAPTER_PATH,
        help=f"Path to adapter or model (default: {DEFAULT_ADAPTER_PATH})",
    )
    args = parser.parse_args()

    # Pre-warm model
    get_model(args.model_path)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
