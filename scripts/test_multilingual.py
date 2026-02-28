"""Multilingual end-to-end test script.

Tests queries in EN, DE, FR against the live RAG system and verifies
language detection and response language.

Usage:
    python scripts/test_multilingual.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation.response_generator import process_customer_query

TEST_QUERIES = [
    ("en", "How do I install the software on-premise?"),
    ("de", "Wie installiere ich die Software on-premise?"),
    ("fr", "Comment installer le logiciel sur site?"),
]


def main():
    print("=" * 70)
    print("MULTILINGUAL RAG TEST")
    print("=" * 70)

    results = []
    for expected_lang, query in TEST_QUERIES:
        print(f"\n{'─' * 70}")
        print(f"QUERY ({expected_lang.upper()}): {query}")
        print(f"{'─' * 70}")

        result = process_customer_query(query)

        detected = result["detected_language"]
        complexity = result["complexity"]
        match = "OK" if detected == expected_lang else "MISMATCH"

        print(f"Detected language: {detected.upper()} [{match}]")
        print(f"Complexity: {complexity}")
        print(f"\nDRAFT RESPONSE (first 300 chars):")
        print(result["draft"][:300])
        if result["citations"]:
            print(f"\nCitations: {', '.join(result['citations'][:3])}")

        esc = result["escalation"]
        if esc.get("needs_escalation"):
            print(f"Escalation: YES — {esc['reason']}")
        else:
            print("Escalation: No")

        results.append((expected_lang, detected, match))

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    for expected, detected, match in results:
        print(f"  Expected: {expected.upper()} | Detected: {detected.upper()} | {match}")

    passed = sum(1 for _, _, m in results if m == "OK")
    print(f"\n{passed}/{len(results)} language detections correct")


if __name__ == "__main__":
    main()
