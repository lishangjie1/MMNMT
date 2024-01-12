# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""GPT style dataset."""
import copy
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from utils.utils import print_rank_0
from utils.data.indexed_dataset import make_dataset as make_indexed_dataset
from utils.dist_utils import wait_for_everyone


def get_indexed_dataset(data_prefix, data_impl='mmap'):
    """Build indexed dataset."""
    print_rank_0(' > building dataset index ...')

    start_time = time.time()

    indexed_dataset = make_indexed_dataset(data_prefix, data_impl, True)

    print_rank_0(' > finished creating indexed dataset in {:4f} '
                 'seconds'.format(time.time() - start_time))

    print_rank_0('    number of documents: {}'.format(indexed_dataset.sizes.shape[0]))

    return indexed_dataset


class PretrainingDataset(Dataset):

    def __init__(self, data_prefix, exp_name, indexed_dataset, documents, seq_length, seed):
        """
        name: xxxx
        data_prefix: xxxx
        documents: (np.ndarray) List of document ids.
        indexed_dataset: (MMapIndexedDataset)
        num_samples: xxxx
        seq_length: Maximum length the model support.
        """
        # self.name = name
        self._data_prefix = data_prefix

        self._exp_name = exp_name

        self.indexed_dataset = indexed_dataset

        # Checks
        assert np.min(documents) >= 0
        assert np.max(documents) < self.indexed_dataset.sizes.shape[0]

        # Build index mappings.
        self.doc_idx, self.sample_idx, self.shuffle_idx = _build_index_mappings(
            data_prefix=self._data_prefix,
            exp_name=self._exp_name,
            documents=documents,
            sizes=self.indexed_dataset.sizes,
            seq_length=seq_length,
            seed=seed
        )

    def __len__(self):
        # -1 is due to data structure used to retieve the index:
        #    sample i --> [sample_idx[i], sample_idx[i+1])
        return self.sample_idx.shape[0] - 1

    def __getitem__(self, idx):
        """
        idx -> non-shuffled idx -> document span
        """
        # Get the shuffled index.
        idx = self.shuffle_idx[idx]
        # Start and end documents and offsets.
        doc_index_f = self.sample_idx[idx][0]
        doc_index_l = self.sample_idx[idx + 1][0]
        offset_f = self.sample_idx[idx][1]
        offset_l = self.sample_idx[idx + 1][1]
        # If we are within the same document, just extract the chunk.
        doc_ids = []
        if doc_index_f == doc_index_l:
            doc_ids.append(self.doc_idx[doc_index_f])
            sample = self.indexed_dataset.get(self.doc_idx[doc_index_f],
                                              offset=offset_f,
                                              length=offset_l - offset_f + 1)
        else:
            # Otherwise, get the rest of the initial document.
            doc_ids.append(self.doc_idx[doc_index_f])
            sample_list = [self.indexed_dataset.get(self.doc_idx[doc_index_f], offset=offset_f)]

            # Loop over all in between documents and add the entire document.
            for i in range(doc_index_f + 1, doc_index_l):
                doc_ids.append(self.doc_idx[i])
                sample_list.append(self.indexed_dataset.get(self.doc_idx[i]))

            # And finally add the relevant portion of last document.

            doc_ids.append(self.doc_idx[doc_index_l])
            sample_list.append(self.indexed_dataset.get(self.doc_idx[doc_index_l], length=offset_l + 1))
            sample = np.concatenate(sample_list)

        input_ids = torch.tensor(sample.astype(np.int64), dtype=torch.long)
        labels = copy.deepcopy(input_ids)

        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    def collect_fn(self, instances):

        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))

        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)

        # TODO:
        # can attention mask be None?
        return dict(
            input_ids=input_ids,
            labels=labels
        )


def _build_index_mappings(data_prefix, exp_name, documents, sizes, seq_length, seed):
    """Build doc-idx, sample-idx, and shuffle-idx.

    documents: list of document ids.
    sizes: list of document sizes (number of tokens).
    """
    # Number of tokens in each epoch and number of required epochs.
    tokens_per_epoch = _num_tokens(documents, sizes)

    # rng state
    np_rng = np.random.RandomState(seed=seed)

    # Filename of the index mappings.

    _filename = data_prefix + '_' + exp_name
    doc_idx_filename = _filename + '_doc_idx.npy'
    sample_idx_filename = _filename + '_sample_idx.npy'
    shuffle_idx_filename = _filename + '_shuffle_idx.npy'

    # Build the indexed mapping if not exist.
    if torch.distributed.get_rank() == 0:
        if (not os.path.isfile(doc_idx_filename)) or \
                (not os.path.isfile(sample_idx_filename)) or \
                (not os.path.isfile(shuffle_idx_filename)):
            print_rank_0(' > WARNING: could not find index map files, building '
                         'the indices on rank 0 ...')

            # For the last epoch, decide whether include the entire epoch
            # in the global shuffle or not.

            # doc-idx.
            start_time = time.time()
            doc_idx = _build_doc_idx(documents, np_rng)
            np.save(doc_idx_filename, doc_idx, allow_pickle=True)
            print_rank_0(' > elasped time to build and save doc-idx mapping '
                         '(seconds): {:4f}'.format(time.time() - start_time))
            # sample-idx.
            start_time = time.time()

            # Use C++ implementation for speed.
            # First compile and then import.
            from utils.data import helpers
            assert doc_idx.dtype == np.int32
            assert sizes.dtype == np.int32

            # sample_idx是一个2D的矩阵
            # 格式如下
            # [[doc_id_1, doc_offset_1], [doc_id_2, doc_offset_2], ..., [doc_id_N, doc_offset_N]]
            # 对于第k个样本, 其内容为 doc_id_k + doc_offset_k 和 doc_id_k+1 + doc_offset_k+1 之间所有
            # 的doc拼接在一起

            sample_idx = helpers.build_sample_idx(sizes, doc_idx, seq_length, 1, tokens_per_epoch)

            np.save(sample_idx_filename, sample_idx, allow_pickle=True)

            print_rank_0(' > elasped time to build and save sample-idx mapping '
                         '(seconds): {:4f}'.format(time.time() - start_time))
            # shuffle-idx.
            start_time = time.time()

            num_samples = sample_idx.shape[0] - 1
            shuffle_idx = _build_shuffle_idx(num_samples, sample_idx.shape[0] - 1, np_rng)
            np.save(shuffle_idx_filename, shuffle_idx, allow_pickle=True)

            print_rank_0(' > elasped time to build and save shuffle-idx mapping'
                         ' (seconds): {:4f}'.format(time.time() - start_time))

    wait_for_everyone()

    # Load mappings.
    start_time = time.time()
    print_rank_0(' > loading doc-idx mapping from {}'.format(
        doc_idx_filename))
    doc_idx = np.load(doc_idx_filename, allow_pickle=True, mmap_mode='r')
    print_rank_0(' > loading sample-idx mapping from {}'.format(
        sample_idx_filename))
    sample_idx = np.load(sample_idx_filename, allow_pickle=True, mmap_mode='r')
    print_rank_0(' > loading shuffle-idx mapping from {}'.format(
        shuffle_idx_filename))
    shuffle_idx = np.load(shuffle_idx_filename, allow_pickle=True, mmap_mode='r')
    print_rank_0('    loaded indexed file in {:3.3f} seconds'.format(
        time.time() - start_time))
    print_rank_0('    total number of samples: {}'.format(
        sample_idx.shape[0]))

    return doc_idx, sample_idx, shuffle_idx


def _num_tokens(documents, sizes):
    """Total number of tokens in the dataset."""
    return np.sum(sizes[documents])


def _num_epochs(tokens_per_epoch, seq_length, num_samples):
    """Based on number of samples and sequence lenght, calculate how many
    epochs will be needed."""
    num_epochs = 0
    total_tokens = 0
    while True:
        num_epochs += 1
        total_tokens += tokens_per_epoch
        # -1 is because we need to retrieve seq_length + 1 token each time
        # but the last token will overlap with the first token of the next
        # sample except for the last sample.
        if ((total_tokens - 1) // seq_length) >= num_samples:
            return num_epochs


def _build_doc_idx(documents, np_rng):
    """ 这个函数的目的是打乱文档顺序，这样会将随机的文档拼接在一起，而非按照原数据集中文档的顺序
    np_rng: 随机数发生器
    """
    num_epochs = 1
    doc_idx = np.mgrid[0:num_epochs, 0:len(documents)][1]
    doc_idx[:] = documents
    doc_idx = doc_idx.reshape(-1)
    doc_idx = doc_idx.astype(np.int32)
    np_rng.shuffle(doc_idx)

    return doc_idx


def _build_shuffle_idx(num_samples, total_size, np_rng):
    """Build the range [0, size) and shuffle."""
    print(' > building shuffle index with split [0, {}) and [{}, {}) '
          '...'.format(num_samples, num_samples, total_size), flush=True)

    dtype_ = np.uint32
    if total_size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64

    shuffle_idx_first = np.arange(start=0, stop=num_samples,
                                  step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx_first)
    if num_samples == total_size:
        return shuffle_idx_first

    shuffle_idx_last = np.arange(start=num_samples, stop=total_size,
                                 step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx_last)

    return np.concatenate((shuffle_idx_first, shuffle_idx_last))
