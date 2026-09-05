"""See https://docs.langchain.com/oss/python/langchain/tools for further instructions on tool definition"""

from langchain_core.tools import tool


@tool
def coin_flip() -> str:
    """Flip a coin, returning either heads or tails"""
    return "tails"
