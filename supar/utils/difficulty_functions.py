from typing import List, Union


def length(sentence, **kwargs) -> float:
    """
    A simple length based difficulty function.
    """
    return len(sentence)


def length_with_aug(sentence, aug_difficulty_offset: Union[int, float] = 100, **kwargs):
    """
    A simple length based difficulty function where augmented sentences get an additional difficulty penalty.
    """
    if sentence.is_aug:
        return len(sentence) + aug_difficulty_offset
    else:
        return len(sentence)

def aug(sentence, aug_difficulty_offset: Union[int, float] = 100, **kwargs):
    """
    Difficulty function that only considers whether the sentence provided is augmented or not.
    """
    if sentence.is_aug:
        return aug_difficulty_offset
    else:
        return 1

