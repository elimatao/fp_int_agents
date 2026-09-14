from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def get_chat_model(
    base_url: str,
    model: str,
    api_key: str = "none",
    temperature: float = 0.7,
) -> BaseChatModel:
    return ChatOpenAI(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=temperature,
        streaming=True,
    )


_NOMIC_MODELS = {"nomic-embed-text", "nomic-embed-text-v1", "nomic-embed-text-v1.5"}
_E5_MODELS = {"multilingual-e5-small", "multilingual-e5-base", "multilingual-e5-large"}


class _PrefixedEmbeddings(OpenAIEmbeddings):
    query_prefix: str = ""
    document_prefix: str = ""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents([f"{self.document_prefix}{t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(f"{self.query_prefix}{text}")

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await super().aembed_documents(
            [f"{self.document_prefix}{t}" for t in texts]
        )

    async def aembed_query(self, text: str) -> list[float]:
        return await super().aembed_query(f"{self.query_prefix}{text}")


def get_embedding_model(
    base_url: str,
    model: str,
    api_key: str = "none",
) -> Embeddings:
    kwargs = {
        "base_url": base_url,
        "model": model,
        "api_key": api_key,
        "check_embedding_ctx_length": False,
    }
    if model in _NOMIC_MODELS:
        return _PrefixedEmbeddings(
            **kwargs,
            query_prefix="search_query: ",
            document_prefix="search_document: ",
        )
    if model in _E5_MODELS:
        return _PrefixedEmbeddings(
            **kwargs,
            query_prefix="query: ",
            document_prefix="passage: ",
        )
    return OpenAIEmbeddings(**kwargs)
