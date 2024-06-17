from supar.utils.batch_samplers import Sampler, HomogeneousIncreaseSampler


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
    sampler = HomogeneousIncreaseSampler(buckets=buckets, batch_size=batch_size, shuffle=False)
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
    buckets = {22.3: [1, 4, 7, 12], 10: [9, 3, 2, 10], 35.1: [5, 6, 8, 9, 11]}
    # batch size is not used except to calculate how many batches per bucket.
    batch_size = 2
    sampler = HomogeneousIncreaseSampler(buckets=buckets, batch_size=batch_size, shuffle=True)
    total = 0
    indices = []
    index_to_difficulty = {
        index: difficulty
        for difficulty, indices in buckets.items()
        for index in indices
    }
    last_bucket_difficulty = 0
    for batch in sampler:
        assert len(batch) <= batch_size
        assert isinstance(batch, list)
        total += len(batch)
        # check that batch is homogeneous
        assert index_to_difficulty[batch[0]] == index_to_difficulty[batch[1]]
        # check that this batch contains indices with difficulties equal or greater to the last difficulties.
        assert index_to_difficulty[batch[0]] >= last_bucket_difficulty
        last_bucket_difficulty = index_to_difficulty[batch[0]]
