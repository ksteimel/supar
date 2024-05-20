from pathlib import Path
from typing import List, Dict
import pytest
from supar.models.dep.biaffine.transform import CoNLL
from supar.utils.field import Field, RawField
from supar.utils.data import Dataset
from supar.utils.common import BOS, PAD, UNK


@pytest.fixture
def test_fixtures_path() -> Path:
    """Path for test fixtures directory."""
    return Path(__file__).parent / "test_fixtures"


@pytest.fixture
def sample_conllu_file_path(test_fixtures_path: Path) -> Path:
    """Path to sample conllu file."""
    return test_fixtures_path / "sample_conllu.conllu"

@pytest.fixture
def augmented_sample_conllu_file_path(test_fixtures_path: Path) -> Path:
    """Path to sample conllu file."""
    return test_fixtures_path / "aug_sample_conllu.conllu"

@pytest.fixture
def vocabs(sample_conllu_file_path, augmented_sample_conllu_file_path) -> Dict[str, Dict[str, int]]:
    """
    Get vocabs from the sample and augmented file specified.
    """
    vocabs = {"TAG": [], "WORD": []}
    for file in [sample_conllu_file_path, augmented_sample_conllu_file_path]:
        with file.open() as in_file:
            for line in in_file:
                if line.strip():
                    pieces = line.split()
                    assert len(pieces) == 10
                    vocabs['TAG'].append(pieces[3])
                    vocabs['WORD'].append(pieces[1].lower())
    for key, entry_list in vocabs.items():
        entry_set = set(entry_list)
        entry_index_map = {entry: i for i, entry in enumerate(entry_set)}
        vocabs[key] = entry_index_map
    return vocabs

@pytest.fixture
def basic_biaffine_transform(sample_conllu_file_path, augmented_sample_conllu_file_path):
    files = [str(sample_conllu_file_path), str(augmented_sample_conllu_file_path)]
    TAG, CHAR, ELMO, BERT = None, None, None, None
    BOS = ""
    TAG = Field('tags', bos=BOS)
    ARC = Field('arcs', bos=BOS, use_vocab=False, fn=CoNLL.get_arcs)
    REL = Field('rels', bos=BOS)
    WORD = Field('words', pad=PAD, unk=UNK, bos=BOS, lower=True)
    TEXT = RawField('texts')
    transform = CoNLL(FORM=(WORD, TEXT, CHAR, ELMO, BERT), CPOS=TAG, HEAD=ARC, DEPREL=REL)
    dataset = Dataset(transform, files)
    TAG.build(dataset)
    ARC.build(dataset)
    REL.build(dataset)
    WORD.build(dataset)
    return transform


@pytest.fixture
def example_dataset(augmented_sample_conllu_file_path, basic_biaffine_transform):
    """
    Create a sample dataset from augmented data.
    """
    dataset = Dataset(transform=basic_biaffine_transform, data=str(augmented_sample_conllu_file_path))
    res = dataset.build(batch_size=2)
    return res


