"""Build the vector indexes for both the Basic and Advanced systems.

Usage:
    python scripts/build_index.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.index import build_all  # noqa: E402


def main():
    print("Building indexes (Basic: fixed-size; Advanced: context-aware + "
          "hierarchical parent/child) ...")
    stats = build_all()
    print("\n=== Index build complete ===")
    print(f"Documents indexed      : {stats['documents']}")
    print(f"Basic fixed-size chunks: {stats['basic_chunks']}")
    print(f"Advanced child chunks   : {stats['advanced']['children']}")
    print(f"Advanced parent chunks  : {stats['advanced']['parents']}")


if __name__ == "__main__":
    main()
