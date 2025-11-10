from pathlib import Path
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

load_dotenv()

DATA_DIR = Path("data")
FAQ_DOCUMENT_PATH = DATA_DIR / "faq_document.txt"
CHROMA_DB_PATH = Path("chroma_db")

EMBEDDING_MODEL = "text-embedding-3-large"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 200

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)


def load_document(path: Path) -> list[Document]:
    """Load a single FAQ document from the given path."""
    logging.info(f"📄 Loading document from {path}")
    loader = TextLoader(path)
    return loader.load()


def split_into_chunks(documents: list[Document]) -> list[Document]:
    """Split documents into smaller overlapping chunks."""
    logging.info(f"✂️  Splitting {len(documents)} document(s) into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    logging.info(f"✅ Created {len(chunks)} chunks")
    return chunks


def build_vector_store(chunks: list[Document]) -> Chroma:
    """Embed the chunks and persist them in a Chroma vector store."""
    if CHROMA_DB_PATH.exists():
        logging.info("🗑️  Existing Chroma DB found — deleting old collection.")
        Chroma(
            persist_directory=str(CHROMA_DB_PATH), embedding_function=embeddings
        ).delete_collection()

    logging.info("📦  Creating new Chroma vector store...")
    store = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=str(CHROMA_DB_PATH)
    )
    logging.info("✅ Chroma DB created successfully.")
    return store


def display_vector_store_stats(store: Chroma) -> None:
    """Display statistics about the Chroma vector store."""
    collection = store._collection
    count = collection.count()
    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    logging.info(
        f"📊 Vector store contains {count:,} vectors of {dimensions:,} dimensions."
    )


def main() -> None:
    """Main pipeline for loading, splitting, and embedding the FAQ document."""
    documents = load_document(FAQ_DOCUMENT_PATH)
    chunks = split_into_chunks(documents)
    store = build_vector_store(chunks)
    display_vector_store_stats(store)


if __name__ == "__main__":
    main()
