"""
Step 3 — RAGAS Evaluation
===========================
TASK:
  1. Run all 50 QA pairs through BOTH prompt versions, capturing answers + contexts
  2. Build EvaluationDataset with SingleTurnSample objects
  3. Evaluate with 4 RAGAS metrics: faithfulness, answer_relevancy,
     context_recall, context_precision
  4. Print a V1 vs V2 comparison table
  5. Save results to data/ragas_report.json

DELIVERABLE: faithfulness ≥ 0.8 for at least one prompt version
             + data/ragas_report.json file saved

⏰ NOTE: This step takes ~20-30 minutes. Start it early!
"""

import json
import os
import warnings
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from qa_pairs import QA_PAIRS

warnings.filterwarnings("ignore")  # suppress RAGAS deprecation warnings

# ── 1. Environment setup ────────────────────────────────────────────────────
_dotenv_path = Path(__file__).parent / ".env"
load_dotenv(_dotenv_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")
DEFAULT_EMBEDDING_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-3-small")

# ── 2. Prompt templates (same as step 2) ────────────────────────────────────
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

PROMPTS = {
    "v1": PROMPT_V1,
    "v2": PROMPT_V2,
}


# ── 3. Build vectorstore (reuse logic from step 1) ───────────────────────────
def build_vectorstore():
    kb_path = Path(__file__).parent / "data" / "knowledge_base.txt"
    if not kb_path.exists():
        raise FileNotFoundError(f"Knowledge base not found at {kb_path}")

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


# ── 4. Run RAG and capture outputs + contexts ────────────────────────────────
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """
    Run the RAG chain for one question.
    Returns: {"answer": str, "contexts": list[str]}
    """
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    ctx_str = "\n\n".join(contexts)

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": ctx_str, "question": question})

    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    """
    Run all 50 QA pairs through the given prompt version.
    Returns a list of dicts with keys: question, reference, answer, contexts.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(
        model=DEFAULT_LLM_MODEL,
        api_key=SecretStr(OPENAI_API_KEY),
        base_url=OPENAI_BASE_URL,
    )
    prompt = PROMPTS[prompt_version]

    results = []
    print(f"\nRunning 50 questions with prompt {prompt_version} ...")

    for i, qa in enumerate(QA_PAIRS, 1):
        out = run_rag(retriever, llm, prompt, qa["question"])
        results.append(
            {
                "question": qa["question"],
                "reference": qa["reference"],
                "answer": out["answer"],
                "contexts": out["contexts"],
            }
        )
        if i % 10 == 0:
            print(f"  [{i:02d}/50] {qa['question'][:60]}...")

    return results


# ── 5. Build RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list):
    """
    Convert a list of RAG result dicts into a RAGAS EvaluationDataset.
    """
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]
    return EvaluationDataset(samples=samples)


# ── 6. Run RAGAS evaluation ──────────────────────────────────────────────────
def run_ragas_eval(rag_results: list, version: str) -> dict:
    """
    Evaluate RAG outputs with 4 RAGAS metrics.
    Returns a dict: {metric_name: mean_score}
    """
    print(f"\n📐 Running RAGAS evaluation for prompt {version} ...")

    dataset = build_ragas_dataset(rag_results)

    llm_eval = ChatOpenAI(
        model=DEFAULT_LLM_MODEL,
        api_key=SecretStr(OPENAI_API_KEY),
        base_url=OPENAI_BASE_URL,
    )
    emb_eval = OpenAIEmbeddings(
        model=DEFAULT_EMBEDDING_MODEL,
        api_key=SecretStr(OPENAI_API_KEY),
        base_url=OPENAI_BASE_URL,
    )

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm_eval,
        embeddings=emb_eval,
    )

    scores = {}
    for key in [
        "faithfulness",
        "answer_relevancy",
        "context_recall",
        "context_precision",
    ]:
        raw = result[key]  # list of floats
        scores[key] = float(np.mean([v for v in raw if v is not None]))

    for k, v in scores.items():
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}")
    return scores


# ── 7. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 3: RAGAS Evaluation")
    print("=" * 60)

    # Ensure data directory exists
    Path("data").mkdir(exist_ok=True)

    # Build vectorstore
    vectorstore = build_vectorstore()

    # Collect outputs for V1 and V2
    v1_results = collect_rag_outputs(vectorstore, "v1")
    v2_results = collect_rag_outputs(vectorstore, "v2")

    # Run RAGAS evaluation on both
    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")

    # Print comparison table
    print("\n" + "=" * 60)
    print(f"  {'Metric':30s} | {'V1':8s} | {'V2':8s} | {'Winner'}")
    print("-" * 60)
    for metric in [
        "faithfulness",
        "answer_relevancy",
        "context_recall",
        "context_precision",
    ]:
        s1, s2 = v1_scores[metric], v2_scores[metric]
        winner = "V1" if s1 > s2 else "V2" if s2 > s1 else "Tie"
        print(f"  {metric:30s} | {s1:.4f} | {s2:.4f} | {winner}")
    print("=" * 60)

    # Check faithfulness target
    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"✅ Target met: faithfulness = {best_faith:.4f}")
    else:
        print(f"⚠️  Below target ({best_faith:.4f}). Try adjusting chunking or prompts.")

    # Save JSON report to data/ragas_report.json
    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
    }
    report_path = Path("data/ragas_report.json")
    report_path.write_text(json.dumps(report, indent=2))
    print(f"💾 Saved {report_path}")


if __name__ == "__main__":
    main()
