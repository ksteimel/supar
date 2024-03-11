from pathlib import Path
from supar.utils.transform import Batch
from supar.models.dep.biaffine.transform import CoNLL
from supar.utils.data import Dataset, Sampler
from supar.models.dep.biaffine.transform import CoNLL, CoNLLSentence
from supar.utils.transform import Transform


def test_basic_dataset_construction(
    sample_conllu_file_path: Path, basic_biaffine_transform: Transform
):
    """Test that the dataset object can be instantiated using biaffine dependency parsing as an example."""

    dataset = Dataset(
        transform=basic_biaffine_transform, data=str(sample_conllu_file_path)
    )
    assert isinstance(dataset, Dataset)
    assert len(dataset) == 2
    dataset.build(batch_size=2)


def test_dataset_with_augmentation(
    sample_conllu_file_path: Path,
    augmented_sample_conllu_file_path: Path,
    basic_biaffine_transform: Transform,
):
    """Test loading data with augmentation using low quality data."""
    dataset = Dataset(
        transform=basic_biaffine_transform,
        data=str(sample_conllu_file_path),
        data_for_augmentation=str(augmented_sample_conllu_file_path),
    )
    assert isinstance(dataset, Dataset)
    assert len(dataset) == 6
    res = dataset.build(batch_size=2)
    assert len(res.sentences) == 2
    assert len(res.aug_sentences) == 4
    for sentence in res.sentences:
        assert isinstance(sentence, CoNLLSentence)
    for sentence in res.aug_sentences:
        assert isinstance(sentence, CoNLLSentence)

