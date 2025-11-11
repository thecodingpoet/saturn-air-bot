import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from evaluator import evaluate_answer

MODEL = "gpt-4.1-nano"
CHROMA_DB_PATH = Path("chroma_db")

SYSTEM_PROMPT = """
You are a helpful and professional customer service assistant for Saturn Airlines.

Your role is to provide accurate, friendly, and concise information about Saturn Airlines services, policies, flights, bookings, and general inquiries.

Guidelines:
- Only answer questions related to Saturn Airlines and the knowledge base
- Use the provided context to answer questions accurately
- Be friendly and professional in tone
- Prioritize safety and policy information
- Keep responses concise and clear

Handling Out-of-Scope Questions:
1. **Questions completely unrelated to Saturn Airlines** (e.g., general knowledge, other companies, unrelated topics):
   - Politely decline: "I'm here to assist with Saturn Airlines services and policies. I'm not able to help with [topic]. If you have any questions about Saturn Airlines, I'd be happy to help!"

2. **Questions related to Saturn Airlines but not in the knowledge base** (e.g., specific booking issues, account problems, complex policy questions):
   - Politely decline and provide contact information: "I don't have that specific information in my knowledge base. For assistance with [topic], please contact Saturn Airlines customer support at 1-800-SATURN-1 (1-800-728-8761) or email customerservice@saturnairlines.com. Our team is available to help you!"

3. **If context is empty or chunks are irrelevant**:
   - Assess if the question is related to Saturn Airlines
   - If related: Decline with contact information
   - If unrelated: Simply decline without contact information

Context:
{context}
"""

EMBEDDING_MODEL = "text-embedding-3-large"
RETRIEVAL_K = 10

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
vector_store = Chroma(
    persist_directory=str(CHROMA_DB_PATH), embedding_function=embeddings
)
retriever = vector_store.as_retriever()
llm = ChatOpenAI(model_name=MODEL, temperature=0)


def fetch_context(query: str) -> list[Document]:
    """Retrieve relevant context documents for a question."""
    try:
        logging.info(f"🔍 Searching knowledge base...")
        docs = retriever.invoke(query, k=RETRIEVAL_K)
        return docs
    except Exception as e:
        logging.error(f"❌ Failed to retrieve documents: {e}")
        return []


def answer_query(query: str, evaluate: bool = False) -> dict:
    """
    Answer a user query using retrieved context and the language model.
    Optionally evaluate the answer quality.
    """
    context_docs = fetch_context(query)
    context_text = "\n\n".join(doc.page_content for doc in context_docs)

    prompt = SYSTEM_PROMPT.format(context=context_text)
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=query),
    ]

    logging.info("🤖 Generating answer...")
    response = llm.invoke(messages)

    answer_text = response.content.strip()

    result = {
        "user_question": query,
        "system_answer": answer_text,
        "chunks_related": [
            {"content": doc.page_content, "metadata": doc.metadata or {}}
            for doc in context_docs
        ],
    }

    if evaluate:
        evaluation = evaluate_answer(query, answer_text, context_docs)
        result["evaluation"] = evaluation
        result["quality_score"] = evaluation.get("overall_score", 0)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Saturn Airlines Q&A Assistant")
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        required=True,
        help="Your question about Saturn Airlines",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run quality evaluation on the answer",
    )

    args = parser.parse_args()

    result = answer_query(args.query, evaluate=args.evaluate)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
