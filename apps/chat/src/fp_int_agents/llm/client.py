from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def get_chat_model(
    base_url: str,
    model: str,
    api_key: str = "ollama",
    temperature: float = 0.7,
) -> BaseChatModel:
    return ChatOpenAI(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=temperature,
        streaming=True,
    )


def get_embedding_model(
    base_url: str,
    model: str,
    api_key: str = "ollama",
) -> Embeddings:
    return OpenAIEmbeddings(
        base_url=base_url,
        model=model,
        api_key=api_key,
    )
