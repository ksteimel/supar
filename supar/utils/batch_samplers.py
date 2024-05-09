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
    """

    def __init__(
        self,
        buckets: Dict[float, List],
        batch_size: int,
        shuffle: bool = False,
        distributed: bool = False,
        even: bool = True,
        seed: int = 1
    ) -> Sampler:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.distributed = distributed
        self.even = even
        self.seed = seed
        self.sizes, self.buckets = zip(*[(size, bucket) for size, bucket in buckets.items()])
        # number of batches in each bucket, clipped by range [1, len(bucket)]
        self.n_batches = [min(len(bucket), max(round(size * len(bucket) / batch_size), 1))
                          for size, bucket in zip(self.sizes, self.buckets)]
        self.rank, self.n_replicas, self.n_samples = 0, 1, self.n_total_samples
        if distributed:
            self.rank = dist.get_rank()
            self.n_replicas = dist.get_world_size()
            self.n_samples = self.n_total_samples // self.n_replicas
            if self.n_total_samples % self.n_replicas != 0:
                self.n_samples += 1 if even else int(self.rank < self.n_total_samples % self.n_replicas)
        self.epoch = 1

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)
        self.epoch += 1

        total, batches = 0, []
        # if `shuffle=True`, shuffle both the buckets and samples in each bucket
        # for distributed training, make sure each process generates the same random sequence at each epoch
        range_fn = torch.arange if not self.shuffle else lambda x: torch.randperm(x, generator=g)

        def cycle(length):
            while True:
                for i in range_fn(length).tolist():
                    yield i

        for i in cycle(len(self.buckets)):
            bucket = self.buckets[i]
            split_sizes = [(len(bucket) - j - 1) // self.n_batches[i] + 1 for j in range(self.n_batches[i])]
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


