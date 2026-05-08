"""
Step 1 — LangSmith-instrumented RAG Pipeline
=============================================
TASK:
  1. Load your dataset, split into chunks, index with FAISS
  2. Build a RAG chain: retriever → prompt → LLM → output parser
  3. Decorate the query function with @traceable so every call is traced
  4. Run all 50 questions → generates ≥ 50 LangSmith traces

DELIVERABLE: Open https://smith.langchain.com and confirm traces appear.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable
from pydantic import SecretStr

from qa_pairs import QA_PAIRS

# ── 1. Environment setup (must happen before LangChain imports are used) ─────
_dotenv_path = Path(__file__).parent / ".env"
load_dotenv(_dotenv_path)

os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "day22-langsmith-lab")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv(
    "LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"
)

# ── 3. LLM and Embeddings ───────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")
DEFAULT_EMBEDDING_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-3-small")

llm = ChatOpenAI(
    model=DEFAULT_LLM_MODEL,
    api_key=SecretStr(OPENAI_API_KEY),
    base_url=OPENAI_BASE_URL,
)

embeddings = OpenAIEmbeddings(
    model=DEFAULT_EMBEDDING_MODEL,
    api_key=SecretStr(OPENAI_API_KEY),
    base_url=OPENAI_BASE_URL,
)


# ── 4. Build FAISS vector store ─────────────────────────────────────────────
def build_vectorstore():
    """
    Load the knowledge base, split into chunks, embed and index with FAISS.

    Steps:
      a) Read your dataset
      b) Split text with RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
      c) Call FAISS.from_texts(chunks, embeddings) to build the index
      d) Return the vectorstore
    """
    # Read the knowledge base
    kb_path = Path(__file__).parent / "data" / "knowledge_base.txt"
    text = kb_path.read_text()

    # Create a text splitter and split the text
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    print(f"Split into {len(chunks)} chunks")

    # Build and return the FAISS vectorstore
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore


# ── 5. RAG prompt template ──────────────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Use ONLY the context below to answer the question. "
            "If the context does not contain the answer, say you don't know.\n\nContext:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


# ── 6. Build the RAG chain ──────────────────────────────────────────────────
def build_rag_chain(vectorstore):
    """
    Build a LangChain RAG chain using LCEL (pipe operator).

    Chain structure:
        {"context": retriever | format_docs, "question": passthrough}
        | prompt
        | llm
        | StrOutputParser()

    Returns: (chain, retriever)
    """
    # Create a retriever from the vectorstore (k=3)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Define a helper to join retrieved docs into a single string
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Build and return the LCEL chain
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever


# ── 7. Traced query function ────────────────────────────────────────────────
@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain, question: str) -> str:
    """
    Run the RAG chain on a single question.
    The @traceable decorator sends input/output/latency to LangSmith.
    """
    return chain.invoke(question)


# ── 9. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 1: LangSmith RAG Pipeline")
    print("=" * 60)

    # Build the vectorstore
    vectorstore = build_vectorstore()

    # Build the RAG chain
    chain, retriever = build_rag_chain(vectorstore)

    # Loop through all questions from qa_pairs.py, call ask(), print results
    for i, qa in enumerate(QA_PAIRS, 1):
        answer = ask(chain, qa["question"])
        print(f"[{i:02d}/{len(QA_PAIRS)}] Q: {qa['question'][:60]}")
        print(f"       A: {answer[:100]}\n")

    # Print confirmation that traces were sent
    print(
        f"✅ {len(QA_PAIRS)} traces sent to LangSmith project "
        f"'{os.environ['LANGCHAIN_PROJECT']}'"
    )
    print("   Open https://smith.langchain.com to view traces.")


if __name__ == "__main__":
    main()
