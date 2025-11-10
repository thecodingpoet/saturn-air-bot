from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

FAQ_DOCUMENT_PATH = Path("data/faq_document.txt")
CHROMA_DB_PATH = Path("chroma_db")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


def load_document() -> list[Document]:
    """Load the FAQ document from the data directory."""
    loader = TextLoader(FAQ_DOCUMENT_PATH)
    return loader.load()


def create_chunks(documents) -> list[Document]:
    """Create chunks from the documents."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    return text_splitter.split_documents(documents)


def create_embeddings(chunks) -> Chroma:
    """Create embeddings from the chunks and save them to the ChromaDB."""
    if CHROMA_DB_PATH.exists():
        Chroma(
            persist_directory=CHROMA_DB_PATH, embedding_function=embeddings
        ).delete_collection()

    return Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=str(CHROMA_DB_PATH)
    )


if __name__ == "__main__":
    documents = load_document()
    chunks = create_chunks(documents)
    index = create_embeddings(chunks)

    collection = index._collection
    count = collection.count()

    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")

