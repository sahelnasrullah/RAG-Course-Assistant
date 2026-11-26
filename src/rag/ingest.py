"""PDF -> sections -> chunks -> embeddings -> Chroma.

Chunking strategy: split the PDF at headings (bold lines larger than the body
font), so each chunk belongs to one section such as "Assessment and grading".
Long sections are split further on paragraph boundaries with overlap.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import pymupdf

from . import config


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    section: str
    page: int  # 1-based


# ---------------------------------------------------------------- parsing
def _is_bold(span: dict) -> bool:
    return bool(span["flags"] & 16) or "bold" in span["font"].lower()


class Line(NamedTuple):
    page: int    # 1-based
    text: str
    size: float  # largest font size on the line
    bold: bool   # every span is bold
    block: tuple  # (page, block_no): lines of one wrapped heading share a block


def _lines(doc: pymupdf.Document):
    """Yield a Line for every text line in the doc.

    Lines inside a detected table are merged into one "a | b | c" line per
    row, so a table row stays together in a single chunk line.
    """
    for page in doc:
        tables = [pymupdf.Rect(t.bbox) for t in page.find_tables().tables]
        rows: dict[tuple[int, int], list] = {}  # (table_idx, y) -> [(x, text, size, bold)]
        emitted: set[tuple[int, int]] = set()

        page_lines = []
        for b_no, block in enumerate(page.get_text("dict")["blocks"]):
            for line in block.get("lines", []):
                spans = [s for s in line["spans"] if s["text"].strip()]
                if not spans:
                    continue
                text = "".join(s["text"] for s in line["spans"]).strip()
                size = max(s["size"] for s in spans)
                bold = all(_is_bold(s) for s in spans)
                bbox = pymupdf.Rect(line["bbox"])
                t_idx = next((i for i, r in enumerate(tables) if r.contains(bbox)), None)
                if t_idx is None:
                    page_lines.append(("line", text, size, bold, b_no))
                    continue
                # Group by vertical centre (4pt tolerance) to form table rows.
                yc = (bbox.y0 + bbox.y1) / 2
                key = next(
                    (k for k in rows if k[0] == t_idx and abs(k[1] - yc) < 4),
                    (t_idx, yc),
                )
                rows.setdefault(key, []).append((bbox.x0, text, size, bold))
                page_lines.append(("row", key, size, False, None))

        pno = page.number + 1
        for kind, payload, size, bold, b_no in page_lines:
            if kind == "line":
                yield Line(pno, payload, size, bold, (pno, b_no))
            elif payload not in emitted:
                emitted.add(payload)
                cells = [t for _, t, _, _ in sorted(rows[payload])]
                yield Line(pno, " | ".join(cells), size, False, (pno, "table", payload))


PAGE_NO = re.compile(r"^(page\s*)?\d+\s*(\(\d+\)|/\s*\d+|of\s+\d+)?$", re.I)
TOC_LINE = re.compile(r"\.{5,}\s*\d*\s*$")  # "Introduction ........ 3"
TOC_TITLE = re.compile(r"^ta\w* of contents$", re.I)  # also matches the "Tabel" typo


def _boilerplate(lines: list[Line], n_pages: int) -> set[str]:
    """Lines repeated on most pages are running headers/footers."""
    if n_pages < 3:
        return set()
    pages_per_line: dict[str, set[int]] = {}
    for ln in lines:
        pages_per_line.setdefault(ln.text, set()).add(ln.page)
    return {t for t, pages in pages_per_line.items() if len(pages) >= n_pages / 2}


def parse_sections(pdf_path: Path) -> list[dict]:
    """Split a PDF into [{section, page, text}] using bold headings."""
    doc = pymupdf.open(pdf_path)
    lines = list(_lines(doc))
    if not lines:
        return []

    repeated = _boilerplate(lines, len(doc))
    lines = [
        ln for ln in lines
        if ln.text not in repeated and not PAGE_NO.match(ln.text) and not TOC_LINE.search(ln.text)
    ]

    # Most common font size = body text.
    body_size = Counter(round(ln.size, 1) for ln in lines).most_common(1)[0][0]

    sections, current = [], {"section": "Header", "page": 1, "lines": []}
    parent = None
    prev_block = None  # block of the previous line if it was a heading
    for page, text, size, bold, block in lines:
        # A heading is a short, fully bold line at least as large as body text.
        # Bold "Label:" lines (e.g. "Course code:") are field labels, not headings.
        is_heading = (
            bold and size >= body_size - 0.1 and len(text) < 100 and not text.endswith(":")
        )
        if is_heading:
            title = re.sub(r"\s+", " ", text)
            if block == prev_block and not current["lines"]:
                # Heading wrapped onto a second line: glue it to the previous one.
                title = f"{current['section'].split(' > ')[-1]} {title}"
                if parent and current["section"] == parent:
                    parent = None
            elif current["lines"]:
                sections.append(current)
            # "3.1. Objective" is a subsection of the last unnumbered heading.
            if re.match(r"^\d+(\.\d+)*\.?\s", title) and parent:
                name = f"{parent} > {title}"
            else:
                parent, name = title, title
            current = {"section": name, "page": page, "lines": []}
        else:
            current["lines"].append(text)
        prev_block = block if is_heading else None
    if current["lines"]:
        sections.append(current)

    return [
        {"section": s["section"], "page": s["page"], "text": "\n".join(s["lines"])}
        for s in sections
        if not TOC_TITLE.match(s["section"])
    ]


# ---------------------------------------------------------------- chunking
def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Greedy split on line boundaries, carrying `overlap` chars to the next piece."""
    if len(text) <= size:
        return [text]
    pieces, buf = [], ""
    for line in text.split("\n"):
        if buf and len(buf) + len(line) + 1 > size:
            pieces.append(buf)
            buf = buf[-overlap:] if overlap else ""
        buf = f"{buf}\n{line}" if buf else line
    if buf.strip():
        pieces.append(buf)
    return pieces


def chunk_pdf(pdf_path: Path) -> list[Chunk]:
    # "DV2658_syllabus.pdf" -> "DV2658 syllabus": name files meaningfully.
    doc_label = pdf_path.stem.replace("_", " ")
    chunks = []
    for s_idx, sec in enumerate(parse_sections(pdf_path)):
        for p_idx, piece in enumerate(
            split_text(sec["text"], config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        ):
            chunks.append(
                Chunk(
                    id=f"{pdf_path.stem}:{s_idx}:{p_idx}",
                    # Document + section title in the text help both embeddings and BM25.
                    text=f"[{doc_label} | {sec['section']}]\n{piece}",
                    source=pdf_path.name,
                    section=sec["section"],
                    page=sec["page"],
                )
            )
    return chunks


# ---------------------------------------------------------------- indexing
def build_index(data_dir: Path = config.DATA_DIR, index_dir: Path = config.INDEX_DIR):
    import chromadb
    from sentence_transformers import SentenceTransformer

    pdfs = sorted(data_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {data_dir}")

    chunks = [c for pdf in pdfs for c in chunk_pdf(pdf)]
    print(f"{len(pdfs)} PDF(s) -> {len(chunks)} chunks")

    model = SentenceTransformer(config.EMBED_MODEL)
    embeddings = model.encode(
        [config.DOC_PREFIX + c.text for c in chunks], normalize_embeddings=True, show_progress_bar=True
    )

    index_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(index_dir))
    try:
        client.delete_collection(config.COLLECTION)  # full rebuild each time
    except Exception:
        pass
    col = client.create_collection(config.COLLECTION, metadata={"hnsw:space": "cosine"})
    col.add(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embeddings.tolist(),
        metadatas=[{"source": c.source, "section": c.section, "page": c.page} for c in chunks],
    )

    # Plain copy of the chunks for BM25 and for inspection.
    (index_dir / "chunks.json").write_text(
        json.dumps([asdict(c) for c in chunks], indent=2, ensure_ascii=False)
    )
    print(f"Index written to {index_dir}")
    return chunks
