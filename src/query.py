from pathlib import Path

from dotenv import load_dotenv
from langchain.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

MODEL = "gpt-4.1-nano"
CHROMA_DB_PATH = Path("chroma_db")

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Saturn Airlines.
You are chatting with a user about Saturn Airlines.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""

EMBEDDING_MODEL = "text-embedding-3-large"
RETRIEVAL_K = 10

load_dotenv()

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
vector_store = Chroma(
    persist_directory=str(CHROMA_DB_PATH), embedding_function=embeddings
)
retriever = vector_store.as_retriever()
llm = ChatOpenAI(model_name=MODEL, temperature=0)


def fetch_context(query: str) -> list[Document]:
    """
    Retrieve relevant context documents for a question.
    """
    return retriever.invoke(query, k=RETRIEVAL_K)


def answer_query(query: str) -> str:
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

    response = llm.invoke(messages)
    return response.content


def main():
    while True:
        user_query = input("Enter your question (or 'exit' to quit): ")

        if user_query.lower() in ["exit", "quit"]:
            break
        answer = answer_query(user_query)
        print(f"Answer: {answer}\n")


if __name__ == "__main__":
    main()
