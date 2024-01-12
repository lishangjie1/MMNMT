
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "training")))
from utils.data.pretraining_dataset import get_indexed_dataset, PretrainingDataset
import numpy as np
import deepspeed

def get_train_valid_test_split_(splits_string, size):
        splits = []
        if splits_string.find(',') != -1:
            splits = [float(s) for s in splits_string.split(',')]
        elif splits_string.find('/') != -1:
            splits = [float(s) for s in splits_string.split('/')]
        else:
            splits = [float(splits_string)]
        while len(splits) < 3:
            splits.append(0.)
        splits = splits[:3]
        splits_sum = sum(splits)
        assert splits_sum > 0.0
        splits = [split / splits_sum for split in splits]
        splits_index = [0]
        for index, split in enumerate(splits):
            splits_index.append(splits_index[index] +
                                int(round(split * float(size))))
        diff = splits_index[-1] - size
        for index in range(1, len(splits_index)):
            splits_index[index] -= diff
        assert len(splits_index) == 4
        assert splits_index[-1] == size
        return splits_index

deepspeed.init_distributed()


data_path = "/mnt/nas/users/lsj/llama/data/ct_train_zhen_llama_text"
max_seq_len = 100

indexed_dataset = get_indexed_dataset(data_path,
                                           'mmap')

total_num_of_documents = indexed_dataset.sizes.shape[0]
splits = get_train_valid_test_split_('969, 30, 1', total_num_of_documents)
documents = np.arange(start=splits[0], stop=splits[1],
                                      step=1, dtype=np.int32)

train_dataset = PretrainingDataset(
            data_prefix=data_path,
            exp_name='train',
            indexed_dataset=indexed_dataset,
            documents=documents,
            seq_length=max_seq_len,
            seed=0,
        )


print("done")

