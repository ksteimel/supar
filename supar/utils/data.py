# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import queue
import tempfile
import threading
from contextlib import contextmanager
from typing import Dict, Iterable, List, Union

import pathos.multiprocessing as mp
import torch
import torch.distributed as dist
from torch.distributions.utils import lazy_property

from supar.utils.common import INF
from supar.utils.fn import binarize, debinarize, kmeans
from supar.utils.logging import get_logger, progress_bar
from supar.utils.parallel import gather, is_dist, is_master
from supar.utils.transform import Batch, Transform
from supar.utils.difficulty_functions import length, length_with_aug
from supar.utils.batch_samplers import (
    Sampler,
    HomogeneousIncreaseSampler,
    ScheduledIncreaseSampler,
)

logger = get_logger(__name__)


class Dataset(torch.utils.data.Dataset):
    r"""
    Dataset that is compatible with :class:`torch.utils.data.Dataset`, serving as a wrapper for manipulating all data fields
    with the operating behaviours defined in :class:`~supar.utils.transform.Transform`.
    The data fields of all the instantiated sentences can be accessed as an attribute of the dataset.

    Args:
        transform (Transform):
            An instance of :class:`~supar.utils.transform.Transform` or its derivations.
            The instance holds a series of loading and processing behaviours with regard to the specific data format.
        data (Union[str, Iterable]):
            A filename or a list of instances that will be passed into :meth:`transform.load`.
        cache (bool):
            If ``True``, tries to use the previously cached binarized data for fast loading.
            In this way, sentences are loaded on-the-fly according to the meta data.
            If ``False``, all sentences will be directly loaded into the memory.
            Default: ``False``.
        binarize (bool):
            If ``True``, binarizes the dataset once building it. Only works if ``cache=True``. Default: ``False``.
        bin (str):
            Path to binarized files, required if ``cache=True``. Default: ``None``.
        max_len (int):
            Sentences exceeding the length will be discarded. Default: ``None``.
        difficulty_fn(str):
            Specifies how difficulty should be calculated when creating sentence bins.
        kwargs (Dict):
            Together with `data`, kwargs will be passed into :meth:`transform.load` to control the loading behaviour.

    Attributes:
        transform (Transform):
            An instance of :class:`~supar.utils.transform.Transform`.
        sentences (List[Sentence]):
            A list of sentences loaded from the data.
            Each sentence includes fields obeying the data format defined in ``transform``.
            If ``cache=True``, each is a pointer to the sentence stored in the cache file.
    """

    def __init__(
        self,
        transform: Transform,
        data: Union[str, Iterable],
        cache: bool = False,
        binarize: bool = False,
        bin: str = None,
        max_len: int = None,
        difficulty_fn: str = "len",
        **kwargs,
    ) -> Dataset:
        super(Dataset, self).__init__()
        self.transform = transform
        self.data = data
        self.cache = cache
        self.binarize = binarize
        self.bin = bin
        self.max_len = max_len or INF
        self.kwargs = kwargs
        self.difficulty_fn = difficulty_fn
        self.difficulty_function_map = {"len": length, "len_w_aug": length_with_aug}
        if self.difficulty_fn not in self.difficulty_function_map.keys():
            raise RuntimeError(f"Invalid difficulty_fn: {self.difficulty_fn}")
        if cache:
            if not isinstance(data, str) or not os.path.exists(data):
                raise FileNotFoundError("Please specify a valid file path for caching!")
            if self.bin is None:
                self.fbin = data + ".pt"
            else:
                os.makedirs(self.bin, exist_ok=True)
                self.fbin = os.path.join(self.bin, os.path.split(data)[1]) + ".pt"
            if not self.binarize and os.path.exists(self.fbin):
                try:
                    self.sentences = debinarize(self.fbin, meta=True)["sentences"]
                except Exception:
                    raise RuntimeError(
                        f"Error found while debinarizing {self.fbin}, which may have been corrupted. "
                        "Try re-binarizing it first!"
                    )
        else:
            self.sentences = list(transform.load(data, **kwargs))

    def __repr__(self):
        s = f"{self.__class__.__name__}("
        s += f"n_sentences={len(self.sentences)}"
        if hasattr(self, "loader"):
            s += f", n_batches={len(self.loader)}"
        if hasattr(self, "buckets"):
            s += f", n_buckets={len(self.buckets)}"
        if self.cache:
            s += f", cache={self.cache}"
        if self.binarize:
            s += f", binarize={self.binarize}"
        if self.max_len < INF:
            s += f", max_len={self.max_len}"
        s += ")"
        return s

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return (
            debinarize(self.fbin, self.sentences[index])
            if self.cache
            else self.sentences[index]
        )

    def __getattr__(self, name):
        if name not in {f.name for f in self.transform.flattened_fields}:
            raise AttributeError(f"Property {name} unavailable!")
        if self.cache:
            if os.path.exists(self.fbin) and not self.binarize:
                sentences = self
            else:
                sentences = self.transform.load(self.data, **self.kwargs)
            return (getattr(sentence, name) for sentence in sentences)
        return [getattr(sentence, name) for sentence in self.sentences]

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    @lazy_property
    def sizes(self):
        if not self.cache:
            return [s.size for s in self.sentences]
        return debinarize(self.fbin, "sizes")

    def build(
        self,
        batch_size: int,
        n_buckets: int = 1,
        shuffle: bool = False,
        distributed: bool = False,
        even: bool = True,
        n_workers: int = 0,
        seed: int = 1,
        pin_memory: bool = True,
        chunk_size: int = 10000,
        batch_sampler: str = "supar_default",
    ) -> Dataset:
        # if not forced and the binarized file already exists, directly load the meta file
        if self.cache and os.path.exists(self.fbin) and not self.binarize:
            self.sentences = debinarize(self.fbin, meta=True)["sentences"]
        else:
            with tempfile.TemporaryDirectory() as ftemp:
                ftemp = gather(ftemp)[0] if is_dist() else ftemp
                fbin = self.fbin if self.cache else os.path.join(ftemp, "data.pt")

                @contextmanager
                def cache(sentences):
                    fs = os.path.join(ftemp, "sentences")
                    fb = os.path.join(ftemp, os.path.basename(fbin))
                    global global_transform
                    global_transform = self.transform
                    sentences = binarize({"sentences": progress_bar(sentences)}, fs)[1][
                        "sentences"
                    ]
                    try:
                        yield (
                            (
                                sentences[s : s + chunk_size],
                                fs,
                                f"{fb}.{i}",
                                self.max_len,
                            )
                            for i, s in enumerate(range(0, len(sentences), chunk_size))
                        )
                    finally:
                        del global_transform

                def numericalize(sentences, fs, fb, max_len):
                    sentences = global_transform(
                        (debinarize(fs, sentence) for sentence in sentences)
                    )
                    sentences = [i for i in sentences if len(i) < max_len]
                    return binarize(
                        {
                            "sentences": sentences,
                            "sizes": [sentence.size for sentence in sentences],
                        },
                        fb,
                    )[0]

                logger.info(f"Caching the data to {fbin}")
                # numericalize the fields of each sentence
                if is_master():
                    with cache(
                        self.transform.load(self.data, **self.kwargs)
                    ) as chunks, mp.Pool(32) as pool:
                        results = [
                            pool.apply_async(numericalize, chunk) for chunk in chunks
                        ]
                        self.sentences = binarize(
                            (r.get() for r in results), fbin, merge=True
                        )[1]["sentences"]
                if is_dist():
                    dist.barrier()
                self.sentences = debinarize(fbin, meta=True)["sentences"]
                if not self.cache:
                    self.sentences = [
                        debinarize(fbin, i) for i in progress_bar(self.sentences)
                    ]
                if is_dist():
                    dist.barrier()


        aug_offset = 100
        difficulty_func = self.difficulty_function_map[self.difficulty_fn]
        difficulties = [
            difficulty_func(sent, aug_offset=aug_offset) for sent in self.sentences
        ]
        difficulties_per_sent = [
            (difficulty, sent_index)
            for sent_index, difficulty in enumerate(difficulties)
        ]
        # NOTE: the final bucket count is roughly equal to n_buckets
        self.buckets = dict(zip(*kmeans(difficulties, n_buckets)))
        if batch_sampler == "homogeneous_increase":
            sampler_obj = HomogeneousIncreaseSampler(
                buckets=self.buckets,
                batch_size=batch_size,
                shuffle=shuffle,
                distributed=distributed,
                even=even,
                seed=seed,
                difficulties_per_sent=difficulties_per_sent,
            )
        elif batch_sampler == "scheduled_increase":
            sampler_obj = ScheduledIncreaseSampler(
                buckets=self.buckets,
                batch_size=batch_size,
                shuffle=shuffle,
                distributed=distributed,
                even=even,
                seed=seed,
                difficulties_per_sent=difficulties_per_sent,
            )

        elif batch_sampler == "supar_default":
            sampler_obj = Sampler(
                buckets=self.buckets,
                batch_size=batch_size,
                shuffle=shuffle,
                distributed=distributed,
                even=even,
                seed=seed,
                difficulties_per_sent=difficulties_per_sent,
            )
        else:
            raise ValueError(f"batch_sampler is invalid value: {batch_sampler}")

        self.loader = DataLoader(
            transform=self.transform,
            dataset=self,
            batch_sampler=sampler_obj,
            num_workers=n_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        )
        return self


class DataLoader(torch.utils.data.DataLoader):
    r"""
    A wrapper for native :class:`torch.utils.data.DataLoader` enhanced with a data prefetcher.
    See http://stackoverflow.com/questions/7323664/python-generator-pre-fetch and
    https://github.com/NVIDIA/apex/issues/304.
    """

    def __init__(self, transform, **kwargs):
        super().__init__(**kwargs)

        self.transform = transform

    def __iter__(self):
        return PrefetchGenerator(self.transform, super().__iter__())


class PrefetchGenerator(threading.Thread):

    def __init__(self, transform, loader, prefetch=1):
        threading.Thread.__init__(self)

        self.transform = transform

        self.queue = queue.Queue(prefetch)
        self.loader = loader
        self.daemon = True
        if torch.cuda.is_available():
            self.stream = torch.cuda.Stream()

        self.start()

    def __iter__(self):
        return self

    def __next__(self):
        if hasattr(self, "stream"):
            torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.queue.get()
        if batch is None:
            raise StopIteration
        return batch

    def run(self):
        # `torch.cuda.current_device` is thread local
        # see https://github.com/pytorch/pytorch/issues/56588
        if is_dist() and torch.cuda.is_available():
            torch.cuda.set_device(dist.get_rank())
        if hasattr(self, "stream"):
            with torch.cuda.stream(self.stream):
                for batch in self.loader:
                    self.queue.put(batch.compose(self.transform))
        else:
            for batch in self.loader:
                self.queue.put(batch.compose(self.transform))
        self.queue.put(None)


def collate_fn(x):
    return Batch(x)
