from typing import Dict, List, Tuple
import torch
import torch.distributed as dist


class Sampler(torch.utils.data.Sampler):
    r"""
    Sampler that supports for bucketization and token-level batchification.

    Args:
        buckets (Dict):
            A dict that maps each centroid to indices of clustered sentences.
            The centroid corresponds to the average length of all sentences in the bucket.
        batch_size (int):
            Token-level batch size. The resulting batch contains roughly the same number of tokens as ``batch_size``.
        shuffle (bool):
            If ``True``, the sampler will shuffle both buckets and samples in each bucket. Default: ``False``.
        distributed (bool):
            If ``True``, the sampler will be used in conjunction with :class:`torch.nn.parallel.DistributedDataParallel`
            that restricts data loading to a subset of the dataset.
            Default: ``False``.
        even (bool):
            If ``True``, the sampler will add extra indices to make the data evenly divisible across the replicas.
            Default: ``True``.
        seed (int):
            Random seed used to shuffle the samples. Default: ``1``.
        difficulties_per_sent (List[Tuple[float, int]]):
            List of tuples where the first element in each tuple is the computed difficulty of that sent and the second element in
            the tuple is the index of that sent. For this base sampler, this is unused.
    """

    def __init__(
        self,
        buckets: Dict[float, List],
        batch_size: int,
        shuffle: bool = False,
        distributed: bool = False,
        even: bool = True,
        seed: int = 1,
        difficulties_per_sent: List[Tuple[float, int]] = []
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.distributed = distributed
        self.even = even
        self.seed = seed
        self.sizes, self.buckets = zip(
            *[(size, bucket) for size, bucket in buckets.items()]
        )
        # number of batches in each bucket, clipped by range [1, len(bucket)]
        self.n_batches = [
            min(len(bucket), max(round(size * len(bucket) / batch_size), 1))
            for size, bucket in zip(self.sizes, self.buckets)
        ]
        self.rank, self.n_replicas, self.n_samples = 0, 1, self.n_total_samples
        if distributed:
            self.rank = dist.get_rank()
            self.n_replicas = dist.get_world_size()
            self.n_samples = self.n_total_samples // self.n_replicas
            if self.n_total_samples % self.n_replicas != 0:
                self.n_samples += (
                    1
                    if even
                    else int(self.rank < self.n_total_samples % self.n_replicas)
                )
        self.epoch = 1

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)
        self.epoch += 1

        total, batches = 0, []
        # if `shuffle=True`, shuffle both the buckets and samples in each bucket
        # for distributed training, make sure each process generates the same random sequence at each epoch
        range_fn = (
            torch.arange
            if not self.shuffle
            else lambda x: torch.randperm(x, generator=g)
        )

        def cycle(length):
            while True:
                for i in range_fn(length).tolist():
                    yield i

        for i in cycle(len(self.buckets)):
            bucket = self.buckets[i]
            split_sizes = [
                (len(bucket) - j - 1) // self.n_batches[i] + 1
                for j in range(self.n_batches[i])
            ]
            # DON'T use `torch.chunk` which may return wrong number of batches
            for batch in range_fn(len(bucket)).split(split_sizes):
                if total % self.n_replicas == self.rank:
                    batches.append([bucket[j] for j in batch.tolist()])
                if len(batches) == self.n_samples:
                    return iter(batches[i] for i in range_fn(self.n_samples).tolist())
                total += 1

    def __len__(self):
        return self.n_samples

    @property
    def n_total_samples(self):
        return sum(self.n_batches)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class HomogeneousIncreaseSampler(Sampler):
    r"""
    Sampler that supports for bucketization and token-level batchification.
    The batches produced will only contain instances from a single bucket and the buckets are selected in order of increasing difficulty.

    Args:
        buckets (Dict):
            A dict that maps each centroid to indices of clustered sentences.
            The centroid corresponds to the average difficulty of all sentences in the bucket.
        batch_size (int):
            Token-level batch size. The resulting batch contains roughly the same number of tokens as ``batch_size``.
        shuffle (bool):
            If ``True``, the sampler will shuffle the samples within a bucket. The order of buckets will still be in increasing order of difficulty.
        distributed (bool):
            If ``True``, the sampler will be used in conjunction with :class:`torch.nn.parallel.DistributedDataParallel`
            that restricts data loading to a subset of the dataset.
            Default: ``False``.
        even (bool):
            If ``True``, the sampler will add extra indices to make the data evenly divisible across the replicas.
            Default: ``True``.
        seed (int):
            Random seed used to shuffle the samples. Default: ``1``.
        difficulties_per_sent (List[Tuple[float, int]]):
            List of tuples where the first element in each tuple is the computed difficulty of that sent and the second element in
            the tuple is the index of that sent. For this sampler, this is unused.
    """

    def __init__(
        self,
        buckets: Dict[float, List],
        batch_size: int,
        shuffle: bool = False,
        distributed: bool = False,
        even: bool = True,
        seed: int = 1,
        difficulties_per_sent: List[Tuple[float, int]] = []
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.distributed = distributed
        self.even = even
        self.seed = seed
        # sort buckets in order of increasing order of difficulty
        buckets = {
            difficulty: buckets[difficulty] for difficulty in sorted(buckets.keys())
        }
        self.difficulties, self.buckets = zip(
            *[(size, bucket) for size, bucket in buckets.items()]
        )
        # cannot use self.sizes to gauge the number of tokens in the bucket because the
        # indices are now just the difficulty. Sometimes this lines up with length, sometimes it does not.
        # it just depends upon the difficulty function used.
        # number of batches in each bucket, clipped by range [1, len(bucket)]
        self.n_batches = [
            min(
                len(bucket),
                len(bucket) // batch_size + int(bool(len(bucket) % batch_size)),
            )
            for bucket in self.buckets
        ]
        print(f"{self.n_batches=}")
        self.rank, self.n_replicas, self.n_samples = 0, 1, self.n_total_samples
        if distributed:
            self.rank = dist.get_rank()
            self.n_replicas = dist.get_world_size()
            self.n_samples = self.n_total_samples // self.n_replicas
            if self.n_total_samples % self.n_replicas != 0:
                self.n_samples += (
                    1
                    if even
                    else int(self.rank < self.n_total_samples % self.n_replicas)
                )
        self.epoch = 1

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)
        self.epoch += 1

        total, batches = 0, []
        # if `shuffle=True`, shuffle the samples in a bucket
        range_fn = (
            torch.arange
            if not self.shuffle
            else lambda x: torch.randperm(x, generator=g)
        )

        def cycle(length):
            while True:
                for i in torch.arange(length).tolist():
                    yield i
        for i in cycle(len(self.buckets)):
            bucket = self.buckets[i]
            split_sizes = [
                (len(bucket) - j - 1) // self.n_batches[i] + 1
                for j in range(self.n_batches[i])
            ]
            # DON'T use `torch.chunk` which may return wrong number of batches
            for batch in range_fn(len(bucket)).split(split_sizes):
                if total % self.n_replicas == self.rank:
                    batches.append([bucket[j] for j in batch.tolist()])
                if len(batches) == self.n_samples:
                    return iter(batches[i] for i in torch.arange(self.n_samples).tolist())
                total += 1

    def __len__(self):
        return self.n_samples

    @property
    def n_total_samples(self):
        return sum(self.n_batches)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class ScheduledIncreaseSampler(Sampler):
    r"""
    Sampler that supports bucketization and token-level batchification.
    The batches produced are heterogenous (e.g. a single batch can be composed of instances from multiple buckets).

    Because of limitations in how the

    Args:
        buckets (Dict):
            A dict that maps each centroid to indices of clustered sentences.
            The centroid corresponds to the average difficulty of all sentences in the bucket.
            For this particular sampler, this argument is unused.
        batch_size (int):
            Token-level batch size. The resulting batch contains roughly the same number of tokens as ``batch_size``.
        shuffle (bool):
            If ``True``, the sampler will shuffle the samples within a bucket. The order of buckets will still be in increasing order of difficulty.
        distributed (bool):
            If ``True``, the sampler will be used in conjunction with :class:`torch.nn.parallel.DistributedDataParallel`
            that restricts data loading to a subset of the dataset.
            Default: ``False``.
        even (bool):
            If ``True``, the sampler will add extra indices to make the data evenly divisible across the replicas.
            Default: ``True``.
        seed (int):
            Random seed used to shuffle the samples. Default: ``1``.
        difficulties_per_sent (List[Tuple[float, int]]):
            List of tuples where the first element in each tuple is the computed difficulty of that sent and the second element in
            the tuple is the index of that sent. These per sentence difficulties are what is actually used to construct the batches for this sampler.
        curriculum_duration: int
            This determines the number of epochs to go through with this sampler method. After self.epoch passes this value, then regular random sampling will occur.
    """

    def __init__(
        self,
        buckets: Dict[float, List],
        batch_size: int,
        shuffle: bool = True,
        distributed: bool = False,
        even: bool = True,
        seed: int = 1,
        difficulties_per_sent: List[Tuple[float, int]] = [],
        curriculum_duration: int = 2,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.distributed = distributed
        self.even = even
        self.seed = seed
        # sort sentence difficulties in order of increasing order of difficulty.
        difficulties_per_sent.sort(key=lambda x: x[0])
        self.difficulties_per_sent = difficulties_per_sent
        self.rank, self.n_replicas, self.n_samples = 0, 1, self.n_total_samples
        if distributed:
            self.rank = dist.get_rank()
            self.n_replicas = dist.get_world_size()
            self.n_samples = self.n_total_samples // self.n_replicas
            if self.n_total_samples % self.n_replicas != 0:
                self.n_samples += (
                    1
                    if even
                    else int(self.rank < self.n_total_samples % self.n_replicas)
                )
        self.epoch = 1
        self.training_step = 0
        self.curriculum_duration = curriculum_duration

    @property
    def n_total_samples(self):
        return len(self.difficulties_per_sent)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)

        # this vector will be 1 if the value is available to be picked.
        selectable_mask = torch.ones(self.n_total_samples)
        # if this value gets initialized to just self.batch_size there is no affect between shufflign vs not
        difficulty_offset = self.batch_size * 2
        total, batches = 0, []
        # if `shuffle=True`, shuffle the samples in a bucket
        range_fn = (
            torch.arange
            if not self.shuffle
            else lambda x: torch.randperm(x, generator=g)
        )

        if self.epoch > self.curriculum_duration:
            # just return some random batch
            for batch in range_fn(self.n_total_samples).split(self.batch_size):
                if not self.distributed or self.rank == 0:
                    yield batch
        else:
            candidate_indices = torch.argwhere(selectable_mask[:difficulty_offset])
            while candidate_indices.shape[0] > 0:
                selected_indices = [int(candidate_indices[j]) for j in range_fn(candidate_indices.shape[0])]
                selected_indices = selected_indices[0: self.batch_size]
                # set these values to 0 in the mask so that they cannot be sampled in the future.
                selectable_mask[selected_indices] = 0
                difficulty_offset += self.batch_size
                candidate_indices = torch.argwhere(selectable_mask[:difficulty_offset])
                yield selected_indices
                


    def __len__(self):
        return self.n_samples


    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
