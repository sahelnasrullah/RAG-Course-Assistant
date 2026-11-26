"""Retrieval over the real index (downloads the embedding model on first run, no LLM).

Builds a temporary index from data/ so the test does not depend on a prior `ingest`.
"""
import pytest

from rag import config
from rag.ingest import build_index
from rag.retrieve import Retriever


@pytest.fixture(scope="module")
def retriever(tmp_path_factory):
    index_dir = tmp_path_factory.mktemp("index")
    build_index(config.DATA_DIR, index_dir)
    return Retriever(index_dir)


@pytest.mark.parametrize(
    "question, doc, section",
    [
        ("How many assignments does the DV2658 course have?", "DV2658_syllabus", "Assessment and grading"),
        ("What are the entry requirements for DV2658?", "DV2658_syllabus", "Entry requirements"),
        ("How long can a deferment of studies last?", "BTH_Study_administrative_rules", "Deferment of studies"),
        ("What is a take-home examination?", "BTH_Modes_of_Assessment", "Take-Home Examination"),
    ],
)
def test_hybrid_finds_gold_chunk_in_top_k(retriever, question, doc, section):
    hits = retriever.search(question, k=config.TOP_K, mode="hybrid")
    assert any(h["id"].startswith(doc + ":") and h["section"] == section for h in hits)


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_search_returns_k_unique_chunks(retriever, mode):
    hits = retriever.search("grading scale", k=5, mode=mode)
    assert len(hits) == 5
    assert len({h["id"] for h in hits}) == 5
    assert all({"text", "source", "section", "page"} <= h.keys() for h in hits)


def test_unknown_mode_raises(retriever):
    with pytest.raises(ValueError):
        retriever.search("x", mode="nope")
