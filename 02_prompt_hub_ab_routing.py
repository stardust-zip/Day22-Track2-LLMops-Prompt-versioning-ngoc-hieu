"""
Step 2 — Prompt Hub & A/B Routing
===================================
TASK:
  1. Write two distinct system prompts (V1: concise, V2: structured)
  2. Push both to LangSmith Prompt Hub via client.push_prompt()
  3. Pull them back via client.pull_prompt()
  4. Implement deterministic A/B routing: hash(request_id) % 2 → V1 or V2
  5. Run all 50 questions through the router → ≥ 50 more LangSmith traces

DELIVERABLE: 2 named prompts visible in https://smith.langchain.com Prompt Hub
"""

import hashlib
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client, traceable
from pydantic import SecretStr

from qa_pairs import QA_PAIRS

# ── 1. Environment setup ────────────────────────────────────────────────────
_dotenv_path = Path(__file__).parent / ".env"
load_dotenv(_dotenv_path)

os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "day22-langsmith-lab")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv(
    "LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"
)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")
DEFAULT_EMBEDDING_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-3-small")

# ── 2. Define two prompt templates ──────────────────────────────────────────
SYSTEM_V1 = (
    "You are a helpful AI assistant. "
    "Answer the user's question using ONLY the provided context. "
    "Keep your answer concise (2-4 sentences). "
    "If the context does not contain the answer, say: 'I don't have enough information.'\n\n"
    "Context:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_V1),
        ("human", "{question}"),
    ]
)

SYSTEM_V2 = (
    "You are an expert AI tutor. Provide a structured, accurate answer.\n\n"
    "Instructions:\n"
    "1. Read the context carefully.\n"
    "2. Identify the key facts relevant to the question.\n"
    "3. Write a clear, well-organized answer (3-5 sentences).\n"
    "4. State explicitly if the context lacks sufficient information.\n\n"
    "Context:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_V2),
        ("human", "{question}"),
    ]
)

# Prompt Hub names
PROMPT_V1_NAME = "ngoc-hieu-rag-prompt-v1"
PROMPT_V2_NAME = "ngoc-hieu-rag-prompt-v2"


# ── 3. Push prompts to LangSmith Prompt Hub ──────────────────────────────────
def push_prompts_to_hub(client):
    """Upload both prompt versions to LangSmith Prompt Hub."""
    try:
        url = client.push_prompt(
            PROMPT_V1_NAME,
            object=PROMPT_V1,
            description="V1 – concise answers, 2-4 sentences",
        )
        print(f"✅ Pushed V1 → {url}")
    except Exception as e:
        print(f"⚠️  V1 push failed: {e}")

    try:
        url = client.push_prompt(
            PROMPT_V2_NAME,
            object=PROMPT_V2,
            description="V2 – structured/expert answers, 3-5 sentences",
        )
        print(f"✅ Pushed V2 → {url}")
    except Exception as e:
        print(f"⚠️  V2 push failed: {e}")


# ── 4. Pull prompts from Prompt Hub ─────────────────────────────────────────
def pull_prompts_from_hub(client):
    """Download both prompt versions from LangSmith Prompt Hub. Falls back to local on error."""
    prompts = {}

    try:
        prompts[PROMPT_V1_NAME] = client.pull_prompt(PROMPT_V1_NAME)
        print(f"↓ Pulled '{PROMPT_V1_NAME}' from Hub")
    except Exception as e:
        prompts[PROMPT_V1_NAME] = PROMPT_V1
        print(f"ℹ️  Using local fallback for '{PROMPT_V1_NAME}' (Error: {e})")

    try:
        prompts[PROMPT_V2_NAME] = client.pull_prompt(PROMPT_V2_NAME)
        print(f"↓ Pulled '{PROMPT_V2_NAME}' from Hub")
    except Exception as e:
        prompts[PROMPT_V2_NAME] = PROMPT_V2
        print(f"ℹ️  Using local fallback for '{PROMPT_V2_NAME}' (Error: {e})")

    return prompts


# ── 5. A/B routing — deterministic hash ─────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    """
    Route a request to prompt V1 or V2 based on the MD5 hash of request_id.
    Even hash → V1, Odd hash → V2. Deterministic.
    """
    hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


# ── 6. Build FAISS vector store ──────────────────────────────────────────────
def build_vectorstore():
    """Load KB, chunk, embed, and index with FAISS."""
    kb_path = Path(__file__).parent / "data" / "knowledge_base.txt"
    text = kb_path.read_text()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    print(f"Split into {len(chunks)} chunks")

    embeddings = OpenAIEmbeddings(
        model=DEFAULT_EMBEDDING_MODEL,
        api_key=SecretStr(OPENAI_API_KEY),
        base_url=OPENAI_BASE_URL,
    )
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore


# ── 7. Traced A/B query function ─────────────────────────────────────────────
@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    """
    Run the RAG chain using the given prompt version.
    Returns a dict: {"question": ..., "answer": ..., "version": ...}
    """
    # print(f"  [DEBUG] Retrieving docs for: {question[:30]}...")
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    # print(f"  [DEBUG] Invoking LLM...")
    answer = (prompt | llm | StrOutputParser()).invoke(
        {"context": context, "question": question}
    )
    return {"question": question, "answer": answer, "version": version}


# ── 8. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 2: Prompt Hub A/B Routing")
    print("=" * 60)

    # Create LangSmith client
    client = Client(api_key=os.environ["LANGCHAIN_API_KEY"])

    # Push both prompts
    push_prompts_to_hub(client)

    # Pull both prompts from Hub
    prompts = pull_prompts_from_hub(client)

    # Build vectorstore, retriever, and LLM
    vectorstore = build_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(
        model=DEFAULT_LLM_MODEL,
        api_key=SecretStr(OPENAI_API_KEY),
        base_url=OPENAI_BASE_URL,
    )

    # Run all 50 questions with A/B routing
    v1_count = 0
    v2_count = 0

    for i, qa in enumerate(QA_PAIRS):
        request_id = f"req-{i:04d}"
        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt = prompts[version_key]

        print(
            f"[{i + 1:02d}/{len(QA_PAIRS)}] [prompt-{version_tag}] Q: {qa['question'][:55]}..."
        )
        result = ask_ab(retriever, llm, prompt, qa["question"], version_tag)
        print(f"         A: {result['answer'][:100]}\n")

        if version_tag == "v1":
            v1_count += 1
        else:
            v2_count += 1

    # Routing summary
    print("=" * 60)
    print("  A/B Routing Summary")
    print("=" * 60)
    print(f"  Total queries    : {len(QA_PAIRS)}")
    print(f"  Prompt V1 (even): {v1_count}")
    print(f"  Prompt V2 (odd) : {v2_count}")
    print(
        f"  ✅ {v1_count + v2_count} traces sent to LangSmith project '{os.environ['LANGCHAIN_PROJECT']}'"
    )
    print("   Open https://smith.langchain.com to view traces.")


if __name__ == "__main__":
    main()
