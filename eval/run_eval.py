"""Evaluation: retrieval quality, RAG vs. no-RAG answers, optional LLM judge.

    python eval/run_eval.py retrieval                 # Hit@k / MRR for dense, bm25, hybrid
    python eval/run_eval.py generate --tag qwen3b     # answers with and without RAG
    python eval/run_eval.py judge --tag qwen3b        # Gemini grades correctness/faithfulness
    python eval/run_eval.py report                    # writes eval/results/summary.md

`generate` and `judge` are resumable: finished rows in results/<tag>.jsonl are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag import config  # noqa: E402
from rag.evaluation import auto_correct, gold_rank, is_refusal, keyfact_recall  # noqa: E402

QUESTIONS = ROOT / "eval" / "questions.json"
RESULTS = ROOT / "eval" / "results"


def load_questions(limit: int | None = None) -> list[dict]:
    qs = json.loads(QUESTIONS.read_text())
    return qs[:limit] if limit else qs


def read_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


# ---------------------------------------------------------------- retrieval
def cmd_retrieval(args) -> None:
    from rag.retrieve import Retriever

    r = Retriever()
    qs = [q for q in load_questions(args.limit) if q.get("gold")]
    lines = [
        f"# Retrieval results\n",
        f"{len(qs)} answerable questions, {len(r.ids)} chunks, embedding model `{config.EMBED_MODEL}`.\n",
        "| Mode | Hit@1 | Hit@3 | Hit@5 | MRR@10 |",
        "|---|---|---|---|---|",
    ]
    misses = {}
    for mode in ("dense", "bm25", "hybrid"):
        ranks = [gold_rank(q, r.search(q["question"], 10, mode)) for q in qs]
        hit = lambda k: sum(1 for x in ranks if x and x <= k) / len(ranks)  # noqa: E731
        mrr = sum(1 / x for x in ranks if x) / len(ranks)
        lines.append(f"| {mode} | {hit(1):.2f} | {hit(3):.2f} | {hit(5):.2f} | {mrr:.2f} |")
        misses[mode] = [(q["id"], x) for q, x in zip(qs, ranks) if not x or x > config.TOP_K]
    lines.append(f"\nQuestions whose gold chunk is not in the top {config.TOP_K}:\n")
    for mode, m in misses.items():
        lines.append(f"- {mode}: " + (", ".join(f"{i} (rank {x or '>10'})" for i, x in m) or "none"))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "retrieval.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---------------------------------------------------------------- generation
def cmd_generate(args) -> None:
    from rag.generate import build_messages, format_context, get_generator
    from rag.retrieve import Retriever

    out = RESULTS / f"{args.tag}.jsonl"
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = read_rows(out)
    done = {(r["id"], r["system"]) for r in rows}
    systems = args.systems.split(",")

    retriever = Retriever()
    generator = get_generator()
    for q in load_questions(args.limit):
        for system in systems:
            if (q["id"], system) in done:
                continue
            chunks = retriever.search(q["question"], args.k, args.mode) if system == "rag" else None
            t = time.time()
            answer = generator(build_messages(q["question"], chunks))
            row = {
                "id": q["id"],
                "system": system,
                "model": config.LLM_MODEL,
                "question": q["question"],
                "answer": answer,
                "latency_s": round(time.time() - t, 1),
                "retrieved": [c["id"] for c in chunks] if chunks else None,
                "context": format_context(chunks) if chunks else None,
                "gold_rank": gold_rank(q, chunks) if chunks else None,
            }
            rows.append(row)
            write_rows(out, rows)  # save after every answer so a crash loses nothing
            print(f"[{q['id']}/{system} {row['latency_s']}s] {answer[:110]!r}", flush=True)


# ---------------------------------------------------------------- judge
def cmd_judge(args) -> None:
    from rag.evaluation import GeminiJudge

    config.load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise SystemExit("Set GEMINI_API_KEY (environment or .env) to use the judge.")
    judge = GeminiJudge(key, config.JUDGE_MODEL)

    out = RESULTS / f"{args.tag}.jsonl"
    rows = read_rows(out)
    qs = {q["id"]: q for q in load_questions()}
    for row in rows:
        if "judge" in row and not args.force:
            continue
        q = qs[row["id"]]
        try:
            row["judge"] = judge.grade(q["question"], q["answer"], row["answer"], row["context"])
            row["judge"]["model"] = judge.model
        except Exception as e:  # keep going; the row is retried on the next run
            print(f"[{row['id']}/{row['system']}] judge failed: {e}", flush=True)
            continue
        write_rows(out, rows)
        print(f"[{row['id']}/{row['system']}] {row['judge']}", flush=True)


# ---------------------------------------------------------------- report
def _pct(xs: list) -> str:
    xs = [x for x in xs if x is not None]
    return f"{100 * sum(xs) / len(xs):.0f}% ({sum(xs):.0f}/{len(xs)})" if xs else "-"


def cmd_report(args) -> None:
    qs = {q["id"]: q for q in load_questions()}
    lines = ["# Generation results\n"]
    table = [
        "| Run | System | Correct (auto, answerable) | Key-fact recall | Refused unanswerable "
        "| Correct (judge, all) | Faithful (judge) | Avg latency |",
        "|---|---|---|---|---|---|---|---|",
    ]
    details = []
    for path in sorted(RESULTS.glob("*.jsonl")):
        rows = read_rows(path)
        for system in ("rag", "norag"):
            rs = [r for r in rows if r["system"] == system]
            if not rs:
                continue
            ans = [r for r in rs if qs[r["id"]].get("answerable", True)]
            una = [r for r in rs if not qs[r["id"]].get("answerable", True)]
            judged = [r for r in rs if "judge" in r]
            recall = sum(keyfact_recall(r["answer"], qs[r["id"]]["must_include"]) for r in ans)
            table.append(
                f"| {path.stem} (`{rs[0]['model']}`) | {'RAG' if system == 'rag' else 'LLM only'} "
                f"| {_pct([auto_correct(qs[r['id']], r['answer']) for r in ans])} "
                f"| {recall / len(ans):.2f} "
                f"| {_pct([is_refusal(r['answer']) for r in una])} "
                f"| {_pct([r['judge'].get('correct') for r in judged])} "
                f"| {_pct([r['judge'].get('faithful') for r in judged]) if system == 'rag' else 'n/a'} "
                f"| {sum(r['latency_s'] for r in rs) / len(rs):.1f}s |"
            )
        details.append(f"\n## {path.stem}: per-question answers\n")
        details.append("| ID | Question | RAG answer | LLM-only answer |")
        details.append("|---|---|---|---|")
        by = {(r["id"], r["system"]): r for r in rows}
        for qid, q in qs.items():
            cells = []
            for system in ("rag", "norag"):
                r = by.get((qid, system))
                if not r:
                    cells.append("-")
                    continue
                mark = "✅" if auto_correct(q, r["answer"]) else "❌"
                cells.append(f"{mark} " + r["answer"].replace("\n", " ").replace("|", "\\|")[:220])
            details.append(f"| {qid} | {q['question']} | {cells[0]} | {cells[1]} |")

    lines += table
    lines.append(
        "\nCorrect (auto): every key fact from `questions.json` appears in the answer. "
        "Refused unanswerable: share of the unanswerable questions where the system said it "
        "could not find / does not know the answer. Judge columns: graded by Gemini against the "
        "reference answer (correct) and the retrieved context (faithful)."
    )
    (RESULTS / "summary.md").write_text("\n".join(lines + details) + "\n")
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("retrieval", "generate", "judge", "report"):
        sp = sub.add_parser(name)
        sp.add_argument("--limit", type=int, default=None, help="only the first N questions")
        if name in ("generate", "judge"):
            sp.add_argument("--tag", required=True, help="results file name, e.g. qwen3b")
        if name == "generate":
            sp.add_argument("--systems", default="rag,norag")
            sp.add_argument("-k", type=int, default=config.TOP_K)
            sp.add_argument("--mode", default=config.RETRIEVAL_MODE)
        if name == "judge":
            sp.add_argument("--force", action="store_true", help="re-grade already judged rows")
    args = p.parse_args()
    {"retrieval": cmd_retrieval, "generate": cmd_generate, "judge": cmd_judge, "report": cmd_report}[
        args.cmd
    ](args)


if __name__ == "__main__":
    main()
