from pathlib import Path
from supar.utils.transform import Batch
from supar.models.dep.biaffine.transform import CoNLL
from supar.utils.data import Dataset, Sampler


def test_basic_dataset_construction(sample_conllu_file_path: Path):
    """Test that the dataset object can be instantiated using the biaffine dependency parsing as an example."""
    dataset = Dataset(transform=CoNLL(), data=str(sample_conllu_file_path))
    assert isinstance(dataset, Dataset)
    assert len(dataset) == 1


