from typing import List, Union


def length(sentence, **kwargs) -> List[float]:
    """
    A simple length based difficulty function.
    """
    return len(sentence)


def length_with_aug(sentence, aug_difficulty_offset: Union[int, float] = 100):
    NotImplemented
