"""Data generator and batch loading utilities."""

import os
import threading
from typing import Dict, Any, Generator, Tuple

import h5py
import numpy as np


class ThreadSafeIter:
    """Takes an iterator/generator and makes it thread-safe by serializing calls."""

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)

    def next(self):
        with self.lock:
            return self.it.next()


def thread_safe_generator(f):
    """Decorator that wraps a generator function into a thread-safe iterator."""
    def g(*a, **kw):
        return ThreadSafeIter(f(*a, **kw))
    return g


@thread_safe_generator
def generate(config: Dict[str, Any], data_type: str, mixture_num: int = 0) -> Generator[Any, None, None]:
    """
    Yields batches of data formatted for the specified mixture sub-model.

    Args:
        config: Project configuration dictionary.
        data_type: 'training_data', 'validation_data', or 'test_data'.
        mixture_num: Index of the current mixture sub-model.

    Yields:
        Tuple of (inputs_dict, outputs_dict) for training/validation,
        or inputs_dict for test data.
    """
    num_mon_sites = config['num_mon_sites']
    num_mon_inst_train = config['num_mon_inst_train']
    num_mon_inst_test = config['num_mon_inst_test']
    num_mon_inst = num_mon_inst_train + num_mon_inst_test
    num_unmon_sites_train = config['num_unmon_sites_train']
    num_unmon_sites_test = config['num_unmon_sites_test']

    data_dir = config.get('data_dir', 'data/processed/')
    batch_size = config.get('batch_size', 50)
    mixture = config.get('mixture', [["dir", "metadata"]])

    sub_comb = mixture[mixture_num] if mixture_num < len(mixture) else mixture[0]
    use_dir = 'dir' in sub_comb
    use_time = 'time' in sub_comb
    use_metadata = 'metadata' in sub_comb

    h5_filename = f"{num_mon_sites}_{num_mon_inst}_{num_unmon_sites_train}_{num_unmon_sites_test}.h5"
    h5_path = os.path.join(data_dir, h5_filename)

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"HDF5 dataset not found: {h5_path}. Run preprocessing first.")

    with h5py.File(h5_path, 'r') as f:
        dir_seq = f[f'{data_type}/dir_seq']
        time_seq = f[f'{data_type}/time_seq']
        metadata = f[f'{data_type}/metadata']
        labels = f[f'{data_type}/labels']

        total_samples = len(labels)
        batch_start = 0

        while True:
            if batch_start >= total_samples:
                batch_start = 0

            batch_end = min(batch_start + batch_size, total_samples)

            batch_inputs = {}
            if use_dir:
                batch_inputs['dir_input'] = dir_seq[batch_start:batch_end]
            if use_time:
                batch_inputs['time_input'] = time_seq[batch_start:batch_end]
            if use_metadata:
                batch_inputs['metadata_input'] = metadata[batch_start:batch_end]

            batch_labels = labels[batch_start:batch_end]
            batch_start += batch_size

            if data_type == 'test_data':
                yield batch_inputs
            else:
                yield (batch_inputs, {'model_output': batch_labels})
