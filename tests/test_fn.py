# -*- coding: utf-8 -*-
import pytest

from supar.structs.fn import tarjan
from supar.utils.fn import kmeans


def test_tarjan():
    sequences = [[4, 1, 2, 0, 4, 4, 8, 6, 8],
                 [2, 5, 0, 3, 1, 5, 8, 6, 8],
                 [2, 5, 0, 4, 1, 5, 8, 6, 8],
                 [2, 5, 0, 4, 1, 9, 6, 5, 7]]
    answers = [None, [[2, 5, 1]], [[2, 5, 1]], [[2, 5, 1], [9, 7, 6]]]
    for sequence, answer in zip(sequences, answers):
        if answer is None:
            assert next(tarjan(sequence), None) == answer
        else:
            assert list(tarjan(sequence)) == answer


def test_kmeans():
    input_lengths = [9, 23, 10, 1, 3, 2, 0, 11, 43, 44, 92, 1]
    output = kmeans(x = input_lengths, k=4)
    assert len(output) == 2
    assert len(output[0]) == 4
   
