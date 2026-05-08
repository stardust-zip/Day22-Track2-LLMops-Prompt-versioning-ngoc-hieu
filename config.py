"""
Shared configuration helpers for Day 22 Lab.
Loads environment variables and exposes LLM/embeddings config.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file
_dotenv_path = Path(__file__).parent / ".env"
load_dotenv(_dotenv_path)


# ── LangSmith ────────────────────────────────────────────────────────────────
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "day22-langsmith-lab")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

# Enable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT
os.environ["LANGCHAIN_ENDPOINT"] = LANGSMITH_ENDPOINT


# ── OpenAI-compatible endpoint ───────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")
DEFAULT_EMBEDDING_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-3-small")


def print_config() -> None:
    """Print current configuration (useful for debugging)."""
    print("✅ Config loaded successfully")
    print(f"   LangSmith project  : {LANGSMITH_PROJECT}")
    print(f"   OpenAI endpoint   : {OPENAI_BASE_URL}")
    print(f"   Default LLM model : {DEFAULT_LLM_MODEL}")
    print(f"   Embedding model   : {DEFAULT_EMBEDDING_MODEL}")
