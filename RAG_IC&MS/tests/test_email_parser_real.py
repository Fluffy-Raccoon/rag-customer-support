import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.parsers.email_parser import parse_email

TEST_DATA = Path(__file__).resolve().parent.parent.parent / "test_data_tyrex_support"
EML_FILES = list(TEST_DATA.glob("*.eml"))


@pytest.mark.parametrize("eml_file", EML_FILES, ids=[f.name for f in EML_FILES])
def test_parse_real_email(eml_file):
    result = parse_email(str(eml_file))
    print(f"\n{'='*60}")
    print(f"FILE: {eml_file.name}")
    print(f"LENGTH: {len(result)} chars")
    print(f"{'='*60}")
    print(result[:2000])
    if len(result) > 2000:
        print(f"\n... truncated ({len(result) - 2000} more chars)")
    print(f"{'='*60}")
    assert "Subject:" in result
    assert len(result) > 20
