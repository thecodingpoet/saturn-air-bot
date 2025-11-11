import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

MODEL = "gpt-4.1-nano"
CHROMA_DB_PATH = Path("chroma_db")

SYSTEM_PROMPT = """
You are a helpful and professional customer service assistant for Saturn Airlines.

Your role is to provide accurate, friendly, and concise information about Saturn Airlines services, policies, flights, bookings, and general inquiries.

Guidelines:
- Only answer questions related to Saturn Airlines and the knowledge base
- Use the provided context to answer questions accurately
- Be friendly and professional in tone
- For questions outside the knowledge base, politely decline and suggest contacting Saturn Airlines customer support
- For booking or account-specific questions, guide users to contact Saturn Airlines directly
- If information is not in the context, be honest and suggest contacting customer support
- Prioritize safety and policy information
- Keep responses concise and clear

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


def answer_query(query: str) -> dict:
    """
    Answer a user query using retrieved context and the language model.
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

    return {
        "user_question": query,
        "system_answer": response.content.strip(),
        "chunks_related": [
            {"content": doc.page_content, "metadata": doc.metadata or {}}
            for doc in context_docs
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Saturn Airlines Q&A Assistant")
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        required=True,
        help="Your question about Saturn Airlines",
    )

    args = parser.parse_args()

    result = answer_query(args.query)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
