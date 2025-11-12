import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def repo_root(monkeypatch):
    root = Path(__file__).resolve().parent.parent
    monkeypatch.syspath_prepend(str(root))
    return root


@pytest.fixture
def fake_build_modules(monkeypatch):
    mod_docs = ModuleType("langchain_core.documents")

    class Document:
        def __init__(self, page_content, metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

    mod_docs.Document = Document
    monkeypatch.setitem(sys.modules, "langchain_core.documents", mod_docs)

    sub = ModuleType("langchain_community.document_loaders")

    class TextLoader:
        def __init__(self, path):
            self.path = path

        def load(self):
            return [Document("dummy content from loader")]

    sub.TextLoader = TextLoader
    monkeypatch.setitem(
        sys.modules, "langchain_community", ModuleType("langchain_community")
    )
    monkeypatch.setitem(sys.modules, "langchain_community.document_loaders", sub)

    mod_split = ModuleType("langchain_text_splitters")

    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=100, chunk_overlap=0):
            self.chunk_size = chunk_size

        def split_documents(self, documents):
            out = []
            for doc in documents:
                text = doc.page_content
                for i in range(0, len(text), self.chunk_size):
                    out.append(Document(text[i : i + self.chunk_size]))
            return out

    mod_split.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter
    monkeypatch.setitem(sys.modules, "langchain_text_splitters", mod_split)

    mod_emb = ModuleType("langchain_openai")

    class OpenAIEmbeddings:
        def __init__(self, model=None):
            self.model = model

    mod_emb.OpenAIEmbeddings = OpenAIEmbeddings
    monkeypatch.setitem(sys.modules, "langchain_openai", mod_emb)

    mod_chroma = ModuleType("langchain_chroma")

    class FakeCollection:
        def __init__(self):
            self._vectors = []

        def count(self):
            return len(self._vectors)

        def get(self, limit=1, include=None):
            return {"embeddings": [[0.1, 0.2, 0.3]]}

    class Chroma:
        def __init__(self, persist_directory=None, embedding_function=None):
            self._collection = FakeCollection()

        @classmethod
        def from_documents(cls, documents, embedding=None, persist_directory=None):
            inst = cls()
            inst._collection._vectors.extend([doc.page_content for doc in documents])
            return inst

        def delete_collection(self):
            self._collection._vectors = []

    mod_chroma.Chroma = Chroma
    monkeypatch.setitem(sys.modules, "langchain_chroma", mod_chroma)


def test_split_into_chunks_returns_expected(repo_root, fake_build_modules):
    import src.build_index as build_index

    doc = build_index.Document("x" * 1200)
    chunks = build_index.split_into_chunks([doc])
    assert isinstance(chunks, list)
    assert len(chunks) >= 2
    assert all(hasattr(c, "page_content") for c in chunks)


def test_build_vector_store_handles_existing_db(
    tmp_path, repo_root, fake_build_modules
):
    import src.build_index as build_index

    db_dir = tmp_path / "chroma_db"
    db_dir.mkdir()
    build_index.CHROMA_DB_PATH = db_dir

    chunks = [build_index.Document("a" * 100), build_index.Document("b" * 100)]
    store = build_index.build_vector_store(chunks)
    assert hasattr(store, "_collection")
    assert store._collection.count() == 2
