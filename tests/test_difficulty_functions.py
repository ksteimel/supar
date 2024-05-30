from supar.utils.difficulty_functions import length, length_with_aug, aug


def test_length_difficulty_function(example_dataset):
    """
    Test basic length difficulty function.
    """
    for sentence in example_dataset:
        assert length(sentence) == len(sentence)


def test_length_with_aug_data_difficulty_function(example_dataset):
    """
    Test that length with additional offset for augmentation works as expected.
    """
    # test default offset
    for sentence in example_dataset:
        difficulty_offset = 0
        if sentence.is_aug:
            difficulty_offset = 100
        assert length_with_aug(sentence) == len(sentence) + difficulty_offset
    # test user specified offset
    target_offset = 20
    for sentence in example_dataset:
        difficulty_offset = 0
        if sentence.is_aug:
            difficulty_offset = target_offset
        assert length_with_aug(sentence, aug_difficulty_offset = difficulty_offset) == len(sentence) + difficulty_offset


def test_aug_difficulty_function(example_dataset):
    """
    Test that length with additional offset for augmentation works as expected.
    """
    # test default offset
    for sentence in example_dataset:
        difficulty_offset = 1
        if sentence.is_aug:
            difficulty_offset = 100
        assert aug(sentence) == difficulty_offset
    # test user specified offset
    target_offset = 20
    for sentence in example_dataset:
        difficulty_offset = 1
        if sentence.is_aug:
            difficulty_offset = target_offset
        assert aug(sentence, aug_difficulty_offset = difficulty_offset) == difficulty_offset


