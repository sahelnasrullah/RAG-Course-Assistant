# RAG Course Assistant

A Retrieval-Augmented Generation (RAG) assistant that answers students' questions about
university course documents (course syllabi and study regulations at Blekinge Institute of
Technology), using only open-source models that run locally.

Real output from the evaluation run (`Qwen2.5-3B-Instruct` on CPU):

```
Q: How many assignments does the DV2658 course have?
RAG:      Based on the information provided in the syllabus, there is 1 assignment mode for
          the DV2658 course, which is worth 3 credits. [1]
          (source [1]: DV2658_syllabus.pdf p.2 | Assessment and grading)
LLM only: The DV2658 course typically has 4 assignments.

Q: What is the maximum length of a deferment of studies at BTH?
RAG:      The maximum length of a deferment of studies at BTH is 18 months unless there are
          exceptional reasons for an extension. [...] [1][2]
LLM only: The maximum length of deferment of studies at BTH (Blekinge Institute of Technology)
          is typically 2 years.
```

## How it works

```
PDFs ──► parse (PyMuPDF) ──► section chunks ──► embeddings (e5-base-v2) ──► Chroma
                                    └──────────► BM25 keyword index
question ──► hybrid retrieval (dense + BM25, Reciprocal Rank Fusion) ──► top-4 chunks
         ──► prompt "answer ONLY from the context, cite passages, otherwise refuse"
         ──► open-source LLM (Qwen2.5-Instruct via transformers, or Ollama) ──► answer + sources
```

| Step | Choice | Why |
|---|---|---|
| PDF parsing | PyMuPDF with font information | Headings are detected as bold lines, so each chunk is one document section (e.g. *Assessment and grading*). Tables are rebuilt row by row (`2510 \| Assignment \| 3 credits \| A-F`), and running headers/footers, page numbers and the table of contents are removed. |
| Chunking | One section per chunk, long sections split at ~1200 chars with 200 overlap | Keeps a fact and its heading together. Every chunk is prefixed with `[document \| section]`. |
| Embeddings | `intfloat/e5-base-v2` | Best of four models we compared on our questions (bge-small, bge-base, all-mpnet). |
| Vector store | ChromaDB (persistent, cosine) | Simple and local. |
| Keyword search | BM25 with light plural stemming and stop words | Exact terms such as "assignment", "Ladok", "deferment" are matched reliably. |
| Fusion | Reciprocal Rank Fusion | Neither method alone was enough (see results). |
| LLM | `Qwen/Qwen2.5-3B-Instruct` on CPU (bf16), `Qwen2.5-7B-Instruct` on a Colab GPU | Open-source, and small enough to run on a laptop. |

## Knowledge source (`data/`)

| File | Document |
|---|---|
| `DV2658_syllabus.pdf` | Course syllabus DV2658 Generative Artificial Intelligence, Dnr BTH-4.1.14-0213-2025 |
| `BTH_Study_administrative_rules.pdf` | [Study administrative rules for preparatory education and first- and second-cycle education](https://www.bth.se/download/18.7a47adda195b297cd34f318/1743070838602/Studieadministrativa-regler_per_220124_en.pdf) |
| `BTH_Modes_of_Assessment.pdf` | [Modes of Assessment and Assessed Components at BTH](https://www.bth.se/download/18.7a47adda195b297cd34f23c/1743070580999/Modes_of_Assessment_and_Assessed_Components_at_BTH.pdf) |

To add documents, put more PDFs in `data/` and run `rag ingest` again. Give each file a
descriptive name, because the file name is used as the document label in every chunk.

## Setup

Requires Python 3.10+ (tested with 3.12).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # CPU-only torch; see the comment in the file for GPU
pip install -e . --no-deps          # installs the `rag` command
rag ingest                          # build the index from data/*.pdf
```

The first run downloads the embedding model (~0.4 GB) and the LLM (~6 GB for the 3B model).
The 3B model needs about 8 GB of free RAM.

## Usage

```bash
rag search "What are the entry requirements for DV2658?"   # retrieved chunks only (fast, no LLM)
rag ask "How long can a deferment of studies last?"         # RAG answer with sources
rag ask --no-rag "How many credits is DV2658?"              # same LLM without retrieval (baseline)
rag chat                                                    # interactive
rag ask --mode dense -k 6 "..."                             # change retrieval mode / top-k
```

Settings are environment variables (see [`src/rag/config.py`](src/rag/config.py)), for example:

```bash
RAG_LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct rag ask "..."      # smaller, faster model
RAG_LLM_BACKEND=ollama RAG_LLM_MODEL=llama3.1:8b rag ask "..."   # use a local Ollama server
```

**Google Colab (GPU):** open [`notebooks/colab_demo.ipynb`](notebooks/colab_demo.ipynb) and
choose a T4 GPU runtime. It runs the same code with `Qwen2.5-7B-Instruct`.

## Evaluation

[`eval/questions.json`](eval/questions.json) holds 51 questions with reference answers that we
checked by hand against the documents:

- 22 about the DV2658 syllabus, 16 about the study rules and 7 about the modes of assessment.
- 6 questions that are **not answerable** from the documents (e.g. "Who is the examiner?"), which
  test whether the system admits it does not know.

Every answerable question lists its *key facts* (with accepted alternative phrasings) and the
*gold* document section that contains the answer.

```bash
python eval/run_eval.py retrieval                  # Hit@k / MRR of dense, BM25, hybrid
python eval/run_eval.py generate --tag qwen2.5-3b  # RAG and LLM-only answers (resumable)
python eval/run_eval.py judge --tag qwen2.5-3b     # optional LLM judge, needs GEMINI_API_KEY in .env
python eval/run_eval.py report                     # eval/results/summary.md
```

The evaluation uses these metrics:

- **Retrieval:** Hit@k (the gold section is among the top k chunks) and MRR.
- **Correct (auto):** a deterministic check that the answer contains *all* key facts. For
  unanswerable questions, the answer counts as correct only if the system refuses.
- **Correct / Faithful (judge):** `gemini-3.5-flash-lite` grades each answer against the
  reference answer (correct). For RAG answers, it also checks whether every claim is supported
  by the retrieved context (faithful). The judge lists the unsupported claims before it scores.
  It never generates answers; it only grades them.

## Results

All runs were on a laptop CPU (no GPU), with hybrid retrieval and k = 4. Full per-question
answers are in [`eval/results/summary.md`](eval/results/summary.md), and the raw rows, including
the retrieved context and the judge's reasons, are in `eval/results/*.jsonl`.

### Retrieval (45 answerable questions, 79 chunks)

| Mode | Hit@1 | Hit@3 | Hit@5 | MRR@10 |
|---|---|---|---|---|
| dense (e5-base-v2) | 0.80 | 0.98 | 1.00 | 0.89 |
| BM25 | 0.84 | 0.98 | 1.00 | 0.91 |
| **hybrid (RRF)** | **1.00** | **1.00** | **1.00** | **1.00** |

Dense and keyword retrieval each fail on different questions, and fusing them puts the gold
section first for every question. Two early fixes mattered:

- **BM25:** without plural stemming, "assignments" never matched "Assignment", and the
  examination table ranked 11th out of 14 chunks.
- **Embeddings:** the first embedding model (bge-small) ranked the examination table 7th.

### Answers: RAG vs. the same LLM without retrieval

| Model | System | Correct (auto, 45 answerable) | Refused (6 unanswerable) | Correct (judge, all 51) | Faithful to context (judge) | Avg latency (CPU) |
|---|---|---|---|---|---|---|
| Qwen2.5-3B-Instruct | **RAG** | **93%** (42/45) | **100%** (6/6) | **92%** (47/51) | 94% (48/51) | 22.0 s |
| Qwen2.5-3B-Instruct | LLM only | 13% (6/45) | 17% (1/6) | 14% (7/51) | n/a | 8.0 s |
| Qwen2.5-1.5B-Instruct | RAG | 89% (40/45) | 67% (4/6) | 88% (45/51) | 94% (48/51) | 9.6 s |
| Qwen2.5-1.5B-Instruct | LLM only | 13% (6/45) | 67% (4/6) | 25% (13/51) | n/a | 7.5 s |

- **RAG raises correctness from about 14% to about 92%.** Without retrieval, the model
  hallucinates specific numbers with confidence. Examples: "The DV2658 course typically has
  4 assignments", a deferment "is typically 2 years" (the rules say 18 months), and "The
  Assignment module in DV2658 is typically worth 10 credits".
- **Refusing when the answer is not there works with the 3B model:** it refused all 6
  unanswerable questions. The 1.5B model, given context, invented answers for 2 of them
  ("DV2658 syllabus contains 1 lecture"; a presentation date of "February 28, 2025", which is
  actually the date the syllabus starts to apply). The bigger model follows the grounding
  instruction better. The LLM-only "refusals" are generic replies such as "I couldn't find
  any specific information on a course called DV2658", which come from not knowing the course
  at all, not from checking a source.
- **The automatic metric is reliable:** it agrees with the Gemini judge on 191 of 204 answers
  (94%). The disagreements are hedged guesses ("it is often English") and paraphrased
  definitions.

### Error analysis (Qwen2.5-3B + RAG)

Every wrong RAG answer had the correct chunk at rank 1, so all errors happened in generation,
not retrieval:

1. **Mixing general rules with course-specific facts (dv04, dv18).** When asked about the
   grading scale of the DV2658 Assignment, the model answered with BTH's general three-point
   scale G/Ux/U from the regulations chunk, instead of A-F from the syllabus table. Both chunks
   were in the context. The same weakness appeared in manual testing, this time inside the
   table itself. Asked "How many credits is the Course Project worth, and how is it graded?",
   the model answered "3 credits … A-F", reading the grade from the Assignment row (the
   correct grade is G-U). A likely fix is to label every cell with its column name
   (`Module: Course Project | Credit: 3 credits | Grade: G-U`), so a small model doesn't have
   to line up columns.
2. **A list split across chunks (ma06).** The sentence "the following components … are not
   recommended as credit-bearing" sits at the end of the *Workshop* section. The list itself
   (Defence, Method, Theory) consists of the next three sections, so the model answered
   "Workshops".
3. **Mixing in a neighbouring passage (ma05).** A correct description of on-campus exams also
   included "conducted simultaneously at all affiliated institutions", which belongs to the
   National Examination section. Only the judge caught this; the key-fact check did not.

Possible improvements:

- Add the neighbouring chunks of a retrieved chunk to the context ("small-to-big" retrieval).
- Rank course-syllabus chunks above general regulations for course-specific questions.
- Use a larger LLM (7B on Colab).

### Limitations

- The question set was written by us after reading the documents, so it is easier than real
  student questions, and the corpus has only three documents. The perfect hybrid retrieval
  score should be read with that in mind.
- The parser is tuned to BTH's PDF layout (bold headings, simple tables).
- Each configuration was run once (greedy decoding, so it is deterministic). The judge
  (`gemini-3.5-flash-lite`) is a much smaller model than a human grader. We spot-checked its
  disagreements with the automatic metric by hand.

## Tests

```bash
python -m pytest -q
```

The tests cover:

- PDF parsing on the real documents (section detection, table rows, wrapped headings,
  header/footer and table-of-contents removal).
- Chunk splitting, BM25 tokenisation and prompt construction.
- The evaluation scoring functions and the consistency of the question set.
- Retrieval on a freshly built index, which downloads the embedding model but not the LLM.

## Project layout

```
data/                 source PDFs
src/rag/
  config.py           all settings (env-overridable)
  ingest.py           PDF -> sections -> chunks -> embeddings -> Chroma (+ chunks.json for BM25)
  retrieve.py         dense, BM25 and hybrid (RRF) retrieval
  generate.py         prompt + LLM backends (transformers / Ollama)
  evaluation.py       scoring helpers and the Gemini judge
  cli.py              `rag` command
eval/
  questions.json      test set
  run_eval.py         evaluation runner
  results/            retrieval.md, summary.md, per-run answers (*.jsonl)
notebooks/            Colab notebook
tests/                pytest suite
```
