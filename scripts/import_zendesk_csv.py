"""CLI tool for importing Zendesk CSV exports.

Usage:
    python scripts/import_zendesk_csv.py --csv /path/to/export.csv
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.pipeline import ingest_document

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Import Zendesk CSV export into the RAG system")
    parser.add_argument("--csv", required=True, help="Path to Zendesk CSV export file")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: File not found: {args.csv}")
        sys.exit(1)

    if csv_path.suffix.lower() != ".csv":
        print("Error: File must be a .csv file")
        sys.exit(1)

    count = ingest_document(str(csv_path))
    print(f"\nImport complete. {count} chunks indexed from Zendesk CSV.")


if __name__ == "__main__":
    main()
