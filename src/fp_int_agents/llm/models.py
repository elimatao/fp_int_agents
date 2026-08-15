from dataclasses import dataclass

import httpx


@dataclass
class ModelInfo:
    id: str


async def list_models(base_url: str, api_key: str = "ollama") -> list[ModelInfo]:
    """Fetch available models from an OpenAI-compatible /models endpoint."""
    headers = {"Authorization": f"Bearer {api_key}"}
    url = base_url.rstrip("/") + "/models"
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
    data = response.json()
    return [ModelInfo(id=entry["id"]) for entry in data.get("data", [])]
