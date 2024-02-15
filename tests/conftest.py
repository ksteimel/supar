from pathlib import Path
import pytest

@pytest.fixture
def test_fixtures_path() -> Path:
    """Path for test fixtures directory."""
    return Path(__file__).parent / "test_fixtures"


@pytest.fixture
def sample_conllu_file_path(test_fixtures_path: Path) -> Path:
    """Path to sample conllu file."""
    return test_fixtures_path / "sample_conllu.conllu"

