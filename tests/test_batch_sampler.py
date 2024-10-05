from supar.utils.batch_samplers import (
    Sampler,
    HomogeneousIncreaseSampler,
    ScheduledIncreaseSampler,
)


def test_original_batch_sampler():
    """Test original sampler implementation."""
    buckets = {10: [9, 3, 2, 10], 22.3: [1, 4, 7, 12], 35.1: [5, 6, 8, 9, 11]}
    # batch size is not used except to calculate how many batches per bucket.
    sampler = Sampler(buckets=buckets, batch_size=2, shuffle=False)
    total = 0
    indices = []
    for batch in sampler:
        assert len(batch) == 1
        assert isinstance(batch, list)
        total += 1
        indices += batch

    unnested_vals = [val for bucket in buckets.values() for val in bucket]
    assert total == len(unnested_vals)
    assert indices == unnested_vals
    # test with shuffling
    sampler = Sampler(buckets=buckets, batch_size=2, shuffle=True)
    total = 0
    indices = []
    for batch in sampler:
        assert len(batch) == 1
        assert isinstance(batch, list)
        total += 1
        indices += batch

    unnested_vals = [val for bucket in buckets.values() for val in bucket]
    assert total == len(unnested_vals)
    assert indices != unnested_vals
    assert sorted(indices) == sorted(unnested_vals)


def test_homogeneous_increase_sampler():
    """Test sampler that produces homogeneous batches where each batch gets slightly harder in difficulty."""
    buckets = {22.3: [1, 4, 7, 12], 10: [9, 3, 2, 10], 35.1: [5, 6, 8, 9]}
    # batch size is not used except to calculate how many batches per bucket.
    batch_size = 2
    sampler = HomogeneousIncreaseSampler(buckets=buckets, batch_size=2, shuffle=False)
    total = 0
    indices = []
    for batch in sampler:
        assert len(batch) == batch_size
        assert isinstance(batch, list)
        total += len(batch)
        indices += batch

    sorted_keys = sorted(buckets.keys())
    unnested_vals = [val for key in sorted_keys for val in buckets[key]]
    assert total == len(unnested_vals)
    assert indices == unnested_vals


def test_homogeneous_increase_sampler_uneven_batches():
    """Test sampler that produces homogeneous batches where each batch gets slightly harder in difficulty."""
    buckets = {22.3: [1, 4, 7, 12, 11], 10: [9, 3, 2, 10], 35.1: [5, 6, 8, 9]}
    # batch size is not used except to calculate how many batches per bucket.
    batch_size = 2
    sampler = HomogeneousIncreaseSampler(
        buckets=buckets, batch_size=batch_size, shuffle=False
    )
    total = 0
    indices = []
    for batch in sampler:
        assert len(batch) <= batch_size
        assert isinstance(batch, list)
        total += len(batch)
        indices += batch

    sorted_keys = sorted(buckets.keys())
    unnested_vals = [val for key in sorted_keys for val in buckets[key]]
    assert total == len(unnested_vals)
    assert indices == unnested_vals


def test_homogeneous_increase_sampler_with_shuffle():
    """Test sampler that produces homogeneous batches where each batch gets slightly harder in difficulty."""
    buckets = {22.3: [1, 4, 7, 12], 10: [3, 2, 10], 35.1: [5, 6, 8, 9, 11]}
    # batch size is not used except to calculate how many batches per bucket.
    batch_size = 2
    sampler = HomogeneousIncreaseSampler(
        buckets=buckets, batch_size=batch_size, shuffle=True
    )
    total = 0
    indices = []
    index_to_difficulty = {
        index: difficulty
        for difficulty, indices in buckets.items()
        for index in indices
    }
    last_bucket_difficulty = 0
    n_batches = 0
    for batch in sampler:
        print(f"{batch=}")
        assert len(batch) <= batch_size
        assert isinstance(batch, list)
        total += len(batch)
        n_batches += 1
        # check that batch is homogeneous
        difficulties = [index_to_difficulty[ind] for ind in batch]
        print(f"{difficulties=}")
        assert len(set(difficulties)) == 1
        # check that this batch contains indices with difficulties equal or greater to the last difficulties.
        assert difficulties[0] >= last_bucket_difficulty
        last_bucket_difficulty = difficulties[0]

    assert n_batches == 7


def test_scheduled_increase_sampler_linear():
    """Test sampler that produces heterogeneous batches with a scheduled increase in the difficulty of sentences to sample from."""
    per_sentence_difficulties = [
        (4.266948461782848, 0),
        (7.041091230697622, 1),
        (1.8793920680163847, 2),
        (3.2187372366811084, 3),
        (2.6653065991348512, 4),
        (7.14093678974219, 5),
        (1.5015760119626393, 6),
        (9.650716591200075, 7),
        (2.0448376789561604, 8),
        (1.981865691142347, 9),
        (7.9540808011587805, 10),
        (3.656149351594115, 11),
    ]
    batch_size = 2
    sampler = ScheduledIncreaseSampler(
        buckets={}, batch_size=batch_size, shuffle=True, difficulties_per_sent=per_sentence_difficulties
    )
    sampler = [batch for batch in sampler]
    assert len(sampler) != 0
    assert len(sampler) == len(per_sentence_difficulties) / batch_size
    per_sentence_difficulties.sort()
    # this has to be initialized to the batch size because the initial difficult offset is batch_size * 2.
    # the line where the sentence offset is incremented happs before the checks at the end of the loop so 
    # the sentence_offset is indeed batch_size * 2 by the time we get to comparisons with the sentence_offset value
    sentence_offset = batch_size
    instances = []
    for batch in sampler:
        assert len(batch) == batch_size
        assert isinstance(batch, list)
        sentence_offset += len(batch)
        for instance in batch:
            assert isinstance(instance, int)
            # the index of the current sentence must be less than the available portion of the data
            assert instance < sentence_offset
            instances.append(instance)

    assert instances != sorted(instances)

def test_scheduled_increase_sampler_linear_no_shuffle():
    """
    Test sampler that produces heterogeneous batches with a scheduled increase in the difficulty of sentences to sample from.

    In this case, the samples are not shuffled. This is boring but necessary.
    """
    per_sentence_difficulties = [
        (4.266948461782848, 0),
        (7.041091230697622, 1),
        (1.8793920680163847, 2),
        (3.2187372366811084, 3),
        (2.6653065991348512, 4),
        (7.14093678974219, 5),
        (1.5015760119626393, 6),
        (9.650716591200075, 7),
        (2.0448376789561604, 8),
        (1.981865691142347, 9),
        (7.9540808011587805, 10),
        (3.656149351594115, 11),
    ]
    batch_size = 2
    sampler = ScheduledIncreaseSampler(
        buckets={}, batch_size=batch_size, shuffle=False, difficulties_per_sent=per_sentence_difficulties
    )
    sampler = [batch for batch in sampler]
    assert len(sampler) != 0
    assert len(sampler) == len(per_sentence_difficulties) / batch_size
    per_sentence_difficulties.sort()
    # this has to be initialized to the batch size because the initial difficult offset is batch_size * 2.
    # the line where the sentence offset is incremented happs before the checks at the end of the loop so 
    # the sentence_offset is indeed batch_size * 2 by the time we get to comparisons with the sentence_offset value
    sentence_offset = batch_size
    instances = []
    for batch in sampler:
        assert len(batch) == batch_size
        assert isinstance(batch, list)
        sentence_offset += len(batch)
        for instance in batch:
            assert isinstance(instance, int)
            # the index of the current sentence must be less than the available portion of the data
            assert instance < sentence_offset
            instances.append(instance)

    assert instances == sorted(instances)
