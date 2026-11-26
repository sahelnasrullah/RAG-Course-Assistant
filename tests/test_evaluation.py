"""Scoring helpers, prompt building and BM25 tokenisation (no models needed)."""
import json
from pathlib import Path

from rag.evaluation import auto_correct, contains_phrase, gold_rank, is_refusal, keyfact_recall, normalize
from rag.generate import REFUSAL, build_messages, format_context
from rag.retrieve import _tokenize

QUESTIONS = Path(__file__).resolve().parents[1] / "eval" / "questions.json"


def test_normalize_dashes_ordinals_and_of():
    assert normalize("1st – 15th of May") == "1-15 may"
    assert normalize("**A–F**") == "a-f"


def test_keyfact_recall_partial():
    facts = [["artificial intelligence"], ["machine learning"]]
    assert keyfact_recall("Requires Artificial Intelligence and Machine Learning.", facts) == 1.0
    assert keyfact_recall("Requires Artificial Intelligence.", facts) == 0.5


def test_contains_phrase_is_whole_word():
    assert not contains_phrase("The course typically has 10 assignments.", "1")
    assert contains_phrase("There is 1 assignment.", "1")
    assert contains_phrase("Graded A–F.", "a-f")
    assert not contains_phrase("unbiased", "bias")


def test_auto_correct_answerable_and_unanswerable():
    item = {"must_include": [["3 credits"]]}
    assert auto_correct(item, "It is worth 3 credits [1].")
    assert not auto_correct(item, "It is worth 7.5 credits.")
    assert not auto_correct({"answerable": False}, "The examiner is Prof. X.")
    assert auto_correct({"answerable": False}, "I could not find this in the provided documents.")


def test_refusal_is_not_a_correct_answer():
    # a refusal must not count as correct even if a key-fact string happens to appear in it
    item = {"must_include": [["lab sessions"]]}
    assert not auto_correct(item, "DV2658 does not specify whether it includes lab sessions or not.")
    assert auto_correct(item, "Yes, DV2658 includes lab sessions [1].")


def test_is_refusal():
    assert is_refusal(f"I {REFUSAL}.")
    assert is_refusal("The syllabus does not specify the examiner.")
    assert is_refusal("The examiner of DV2658 is not specified directly in the provided documents.")
    assert is_refusal("I'm sorry, but I couldn't find any specific information.")
    assert not is_refusal("The course is worth 6 credits.")


def test_gold_rank():
    item = {"gold": {"doc": "DV2658_syllabus", "section": "Assessment"}}
    chunks = [
        {"id": "BTH_Study_administrative_rules:9:0", "section": "Assessment and grading"},
        {"id": "DV2658_syllabus:1:0", "section": "Decision"},
        {"id": "DV2658_syllabus:10:0", "section": "Assessment and grading"},
    ]
    assert gold_rank(item, chunks) == 3
    assert gold_rank(item, chunks[:2]) is None
    assert gold_rank({"answerable": False}, chunks) is None


def test_tokenize_stems_plurals_and_drops_stopwords():
    assert _tokenize("How many assignments does this course have?") == ["assignment"]
    assert _tokenize("Credits and grades") == ["credit", "grade"]
    assert _tokenize("class") == ["class"]  # "ss" is not a plural


def test_build_messages_with_and_without_context():
    chunk = {"source": "a.pdf", "page": 2, "section": "S", "text": "[a | S]\nhello"}
    rag = build_messages("Q?", [chunk])
    assert rag[0]["role"] == "system" and "ONLY" in rag[0]["content"]
    assert "[1] (source: a.pdf, page 2, section: S)" in rag[1]["content"]
    assert rag[1]["content"].endswith("Question: Q?")
    plain = build_messages("Q?", None)
    assert plain[1]["content"] == "Q?"
    assert "Context" not in plain[0]["content"]
    assert format_context([chunk, chunk]).count("[2]") == 1


def test_question_file_is_well_formed():
    qs = json.loads(QUESTIONS.read_text())
    assert 30 <= len(qs) <= 60
    assert len({q["id"] for q in qs}) == len(qs)
    for q in qs:
        assert q["question"] and q["answer"]
        if q.get("answerable", True):
            assert q["must_include"] and all(q["must_include"])
            assert q["gold"]["doc"] and q["gold"]["section"]


def test_reference_answers_pass_their_own_key_facts():
    """Sanity check of the test set: each reference answer contains its key facts."""
    for q in json.loads(QUESTIONS.read_text()):
        if q.get("answerable", True):
            assert keyfact_recall(q["answer"], q["must_include"]) == 1.0, q["id"]
