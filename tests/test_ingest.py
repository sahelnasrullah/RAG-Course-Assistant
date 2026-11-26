"""Parsing and chunking of the real course documents (no models needed)."""
from pathlib import Path

import pytest

from rag.ingest import chunk_pdf, parse_sections, split_text

DATA = Path(__file__).resolve().parents[1] / "data"
SYLLABUS = DATA / "DV2658_syllabus.pdf"
RULES = DATA / "BTH_Study_administrative_rules.pdf"
MODES = DATA / "BTH_Modes_of_Assessment.pdf"


def sections(pdf):
    return {s["section"]: s for s in parse_sections(pdf)}


def test_syllabus_headings_become_sections():
    names = list(sections(SYLLABUS))
    for expected in [
        "Entry requirements",
        "Learning activities",
        "Assessment and grading",
        "Learning outcomes > 4.3. Judgement and approach",
    ]:
        assert expected in names


def test_exam_table_rows_stay_together():
    text = sections(SYLLABUS)["Assessment and grading"]["text"]
    assert "2510 | Assignment | 3 credits | A-F" in text
    assert "2520 | Course Project | 3 credits | G-U" in text


def test_section_page_numbers():
    s = sections(SYLLABUS)
    assert s["Entry requirements"]["page"] == 1
    assert s["Assessment and grading"]["page"] == 2


def test_wrapped_heading_is_merged():
    names = list(sections(RULES))
    assert "Choice of specialisation within a programme" in names
    assert "a programme" not in names


def test_table_of_contents_and_footers_removed():
    for pdf in (RULES, MODES):
        for s in parse_sections(pdf):
            assert "contents" not in s["section"].lower()
            assert "......" not in s["text"]
            assert "371 79 Karlskrona" not in s["text"]


def test_chunks_carry_document_and_section():
    chunks = chunk_pdf(SYLLABUS)
    assert chunks, "no chunks"
    assessment = [c for c in chunks if c.section == "Assessment and grading"]
    assert len(assessment) == 1
    c = assessment[0]
    assert c.text.startswith("[DV2658 syllabus | Assessment and grading]")
    assert c.source == "DV2658_syllabus.pdf"
    assert len({ch.id for ch in chunks}) == len(chunks)  # ids are unique


def test_split_text_short_text_unchanged():
    assert split_text("one\ntwo", size=100, overlap=10) == ["one\ntwo"]


def test_split_text_respects_size_and_overlap():
    lines = [f"line number {i:02d}" for i in range(30)]  # 14 chars each
    pieces = split_text("\n".join(lines), size=100, overlap=20)
    assert len(pieces) > 1
    assert all(len(p) <= 100 + 20 for p in pieces)
    # every original line survives somewhere
    joined = "\n".join(pieces)
    assert all(line in joined for line in lines)
    # consecutive pieces overlap
    assert pieces[0][-10:] in pieces[1]


@pytest.mark.parametrize("pdf", [SYLLABUS, RULES, MODES])
def test_every_document_produces_chunks(pdf):
    assert len(chunk_pdf(pdf)) >= 10
