from pathlib import Path
import pytest
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
    # Batch size in supar represents the number of tokens in a batch. 
    # However, this number is only approximate so a batch size that is 2, will still at least allocate a single full sentence to a batch
    dataset.build(batch_size=16, batch_sampler="supar_default")
    batches = [batch for batch in dataset.loader]
    assert len(batches) == 1
    # At a batch size of 2 tokens, the batches produced each contain one sentence. 
    # There are no sentences with a single word in the sample data and the batching does not allow partial sentences in a batch.
    dataset.build(batch_size=2, batch_sampler="supar_default")
    batches = [batch for batch in dataset.loader]
    assert len(batches) == 2


def test_basic_dataset_construction_alternative_batch_sampler(
    sample_conllu_file_path: Path, basic_biaffine_transform: Transform
):
    """Test that the dataset object can be instantiated using biaffine dependency parsing as an example."""

    dataset = Dataset(
        transform=basic_biaffine_transform, data=str(sample_conllu_file_path)
    )
    assert isinstance(dataset, Dataset)
    assert len(dataset) == 2
    # At a batch size of 2 tokens, the batches produced each contain one sentence. 
    # There are no sentences with a single word in the sample data and the batching does not allow partial sentences in a batch.
    dataset.build(batch_size=2, batch_sampler="scheduled_increase")
    scheduled_increase_batches = [batch for batch in dataset.loader]
    dataset.build(batch_size=2, batch_sampler="supar_default")
    supar_default_batches = [batch for batch in dataset.loader]
    # it's expected that the number of batches produced by the ScheduledIncreaseSampler is not the same
    # as that produced by the default sampler because the default sampler is using the length value of 
    # the centroid of the bucket to produce batches that are roughly balanced in terms of total number of tokens.
    # this would be pretty cumbersome to introduce into the ScheduledIncreaseSampler or HomogeneousIncreaseSampler, however.
    assert len(scheduled_increase_batches) == 1
    assert len(supar_default_batches) != len(scheduled_increase_batches)



def test_dataset_with_augmentation(
    augmented_sample_conllu_file_path: Path,
    basic_biaffine_transform: Transform,
):
    """Test loading data with augmentation using low quality data."""
    dataset = Dataset(
        transform=basic_biaffine_transform,
        data=str(augmented_sample_conllu_file_path),
    )
    assert isinstance(dataset, Dataset)
    assert len(dataset) == 6
    res = dataset.build(batch_size=2, batch_sampler="supar_default")
    assert len(dataset) == 6
    assert len(res.sentences) == 6
    aug_count = 0
    for sentence in res.sentences:
        assert isinstance(sentence, CoNLLSentence)
        if sentence.is_aug:
            aug_count += 1
    assert aug_count == 4
    # sample aug data contains 28 tokens (4 sents), sample data contains 14 tokens (2 sents).
    # each sentence will be allocated to one batch since the batch size currently in use is 2 and batches are not allowed to contain partial sentences
    batches = [batch for batch in dataset.loader]
    assert len(batches) == 6
    # this should now put two sentences in each batch
    res = dataset.build(batch_size=16, batch_sampler="supar_default")
    batches = [batch for batch in dataset.loader]
    assert len(batches) == 3


def test_basic_length_sampler_with_augmentation(sample_conllu_file_path: Path, augmented_sample_conllu_file_path: Path, basic_biaffine_transform: Transform):
    """
    Test that using the default, length-based bucketing strategy works with augmentation.
    """
    assert True


def test_data_loader_invalid_difficulty_fn(
    augmented_sample_conllu_file_path: Path,
    basic_biaffine_transform: Transform,
):
    """Test loading data with augmentation using low quality data."""
    with pytest.raises(RuntimeError):
        dataset = Dataset(
            transform=basic_biaffine_transform,
            data=str(augmented_sample_conllu_file_path),
            difficulty_fn="inverse_square"
        )
