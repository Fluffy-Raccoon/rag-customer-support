"""CLI tool for testing queries against the RAG system.

Usage:
    python scripts/test_query.py --query "How do I reset my password?"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation.response_generator import process_customer_query


def main():
    parser = argparse.ArgumentParser(description="Test a query against the RAG system")
    parser.add_argument("--query", required=True, help="Customer query to process")
    args = parser.parse_args()

    print(f"Processing query: {args.query}\n")
    result = process_customer_query(args.query)

    print("DRAFT RESPONSE")
    print("=" * 60)
    print(result["draft"])
    print()

    print("REFERENCES")
    print("=" * 60)
    for cite in result["citations"]:
        print(cite)
    print()

    print("ESCALATION RECOMMENDATION")
    print("=" * 60)
    esc = result["escalation"]
    if esc.get("needs_escalation"):
        print(f"!! ESCALATE: {esc['reason']}")
    else:
        print("No escalation needed")
    print()

    print(f"Language detected: {result['detected_language'].upper()}")


if __name__ == "__main__":
    main()
