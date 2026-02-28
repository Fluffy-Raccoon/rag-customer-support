"""CLI tool for re-indexing documents.

Usage:
    python scripts/reindex.py --documents /path/to/docs [--full-reindex]
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.pipeline import ingest_document, ingest_directory

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Re-index documents into the RAG system")
    parser.add_argument(
        "--documents",
        required=True,
        help="Path to a file or directory to index",
    )
    parser.add_argument(
        "--full-reindex",
        action="store_true",
        help="Clear existing vectors before re-indexing",
    )
    args = parser.parse_args()

    path = Path(args.documents)
    if not path.exists():
        print(f"Error: Path not found: {args.documents}")
        sys.exit(1)

    if path.is_dir():
        count = ingest_directory(str(path), full_reindex=args.full_reindex)
    else:
        count = ingest_document(str(path), full_reindex=args.full_reindex)

    print(f"\nReindexing complete. {count} chunks processed.")


if __name__ == "__main__":
    main()
