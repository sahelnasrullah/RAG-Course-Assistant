"""Command-line interface.

    rag ingest                      # build the index from data/*.pdf
    rag search "How many assignments?"   # retrieval only, no LLM
    rag ask "How many assignments?"      # full RAG answer
    rag ask --no-rag "..."               # LLM alone (baseline)
    rag chat                        # interactive loop
"""
import argparse

from . import config


def show_chunks(chunks):
    for i, c in enumerate(chunks, 1):
        preview = c["text"].replace("\n", " ")[:160]
        print(f"  [{i}] {c['source']} p.{c['page']} | {c['section']}\n      {preview}...")


def answer(question, retriever, generator, args):
    from .generate import build_messages

    chunks = None if args.no_rag else retriever.search(question, args.k, args.mode)
    print(generator(build_messages(question, chunks)))
    if chunks:
        print("\nSources:")
        show_chunks(chunks)


def main():
    p = argparse.ArgumentParser(prog="rag")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="build the vector index from data/*.pdf")
    helps = {
        "search": "show the retrieved chunks for a question (no LLM)",
        "ask": "answer one question",
        "chat": "interactive question loop",
    }
    for name in ("search", "ask", "chat"):
        sp = sub.add_parser(name, help=helps[name])
        if name != "chat":
            sp.add_argument("question")
        sp.add_argument("-k", type=int, default=config.TOP_K)
        sp.add_argument("--mode", default=config.RETRIEVAL_MODE, choices=["dense", "bm25", "hybrid"])
        if name != "search":
            sp.add_argument("--no-rag", action="store_true", help="skip retrieval (baseline)")
    args = p.parse_args()

    if args.cmd == "ingest":
        from .ingest import build_index
        build_index()
        return

    from .retrieve import Retriever
    retriever = None if getattr(args, "no_rag", False) else Retriever()

    if args.cmd == "search":
        show_chunks(retriever.search(args.question, args.k, args.mode))
        return

    from .generate import get_generator
    generator = get_generator()

    if args.cmd == "ask":
        answer(args.question, retriever, generator, args)
    else:
        print("Ask a question (empty line to quit).")
        while q := input("\n> ").strip():
            answer(q, retriever, generator, args)


if __name__ == "__main__":
    main()
