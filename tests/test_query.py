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
def fake_query_modules(monkeypatch):
    # documents
    mod_docs = ModuleType("langchain_core.documents")

    class Document:
        def __init__(self, page_content, metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

    mod_docs.Document = Document
    monkeypatch.setitem(sys.modules, "langchain_core.documents", mod_docs)

    # openai
    mod_openai = ModuleType("langchain_openai")

    class ChatOpenAI:
        def __init__(self, model_name=None, temperature=0):
            self.model_name = model_name

        def invoke(self, messages):
            class R:
                def __init__(self):
                    self.content = "fake answer"

            return R()

    class OpenAIEmbeddings:
        def __init__(self, model=None):
            self.model = model

    mod_openai.ChatOpenAI = ChatOpenAI
    mod_openai.OpenAIEmbeddings = OpenAIEmbeddings
    monkeypatch.setitem(sys.modules, "langchain_openai", mod_openai)

    # chroma
    mod_chroma = ModuleType("langchain_chroma")

    class Retriever:
        def __init__(self, docs=None):
            self.docs = docs or []

        def invoke(self, query, k=10):
            if isinstance(self.docs, Exception):
                raise self.docs
            return self.docs

    class Chroma:
        def __init__(self, persist_directory=None, embedding_function=None):
            self.persist_directory = persist_directory

        def as_retriever(self):
            return Retriever()

    mod_chroma.Chroma = Chroma
    monkeypatch.setitem(sys.modules, "langchain_chroma", mod_chroma)

    # evaluator
    mod_eval = ModuleType("evaluator")

    def evaluate_answer(query, answer, chunks):
        return {"overall_score": 7}

    mod_eval.evaluate_answer = evaluate_answer
    monkeypatch.setitem(sys.modules, "evaluator", mod_eval)


def test_fetch_context_returns_empty_on_error(
    repo_root, fake_query_modules, monkeypatch
):
    import src.query as query

    class BadRetriever:
        def invoke(self, q, k=10):
            raise RuntimeError("boom")

    monkeypatch.setattr(query, "retriever", BadRetriever())
    docs = query.fetch_context("hello")
    assert docs == []


def test_answer_query_returns_result_and_evaluation(
    repo_root, fake_query_modules, monkeypatch
):
    import src.query as query

    doc1 = query.Document("context one", metadata={"id": 1})
    doc2 = query.Document("context two", metadata={"id": 2})

    class GoodRetriever:
        def invoke(self, q, k=10):
            return [doc1, doc2]

    monkeypatch.setattr(query, "retriever", GoodRetriever())

    class FakeLLM:
        def invoke(self, messages):
            class R:
                def __init__(self):
                    self.content = "This is a test answer"

            return R()

    monkeypatch.setattr(query, "llm", FakeLLM())

    result = query.answer_query("What is Saturn?", evaluate=True)
    assert result["user_question"] == "What is Saturn?"
    assert result["system_answer"] == "This is a test answer"
    assert isinstance(result["chunks_related"], list)
    assert "evaluation" in result
    assert result.get("quality_score") == 7
