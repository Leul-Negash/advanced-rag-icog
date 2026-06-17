"""Ask a question against the Basic and/or Advanced system (demo CLI).

Usage:
    python scripts/ask.py "What does MeTTa stand for?"
    python scripts/ask.py --system advanced "How are PeTTa and MORK related?"
    python scripts/ask.py --system both "What does MORK stand for?"
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.advanced_rag import AdvancedRAG  # noqa: E402
from src.basic_rag import BasicRAG  # noqa: E402


def show(name, result):
    print(f"\n===== {name} =====")
    if result.trace.get("techniques"):
        print(f"techniques: {', '.join(result.trace['techniques'])}")
    print(f"refused: {result.refused}")
    print(f"\nANSWER:\n{result.answer}")
    print("\nSOURCES:")
    for i, s in enumerate(result.contexts, 1):
        print(f"  [{i}] {s.chunk.title}  (score={s.score:.3f}, via {s.source})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--system", choices=["basic", "advanced", "both"], default="both")
    args = ap.parse_args()

    if args.system in ("basic", "both"):
        show("BASIC RAG", BasicRAG().query(args.question))
    if args.system in ("advanced", "both"):
        show("ADVANCED RAG", AdvancedRAG().query(args.question))


if __name__ == "__main__":
    main()
