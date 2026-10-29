"""Unit tests for data pipeline and preprocessing."""

import os
import tempfile
import unittest
import numpy as np

from src.data.preprocessing import process_trace
from src.data.dataset import ThreadSafeIter


class TestDataPipeline(unittest.TestCase):

    def setUp(self):
        # Create a mock Wang trace file
        self.temp_dir = tempfile.mkdtemp()
        self.trace_file = os.path.join(self.temp_dir, "5-10")  # site 5, instance 10
        with open(self.trace_file, "w") as f:
            f.write("0.001\t1\n")
            f.write("0.005\t-1\n")
            f.write("0.010\t1\n")
            f.write("0.025\t-1\n")

    def tearDown(self):
        if os.path.exists(self.trace_file):
            os.remove(self.trace_file)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    def test_process_trace(self):
        seq_len = 100
        dir_seq, time_seq, metadata, label = process_trace(self.trace_file, seq_length=seq_len)

        self.assertEqual(len(dir_seq), seq_len)
        self.assertEqual(len(time_seq), seq_len)
        self.assertEqual(len(metadata), 7)
        self.assertEqual(label, 6)  # 5 + 1

        # Check packet values
        self.assertEqual(dir_seq[0], 1)
        self.assertEqual(dir_seq[1], -1)
        self.assertEqual(dir_seq[2], 1)
        self.assertEqual(dir_seq[3], -1)
        self.assertEqual(dir_seq[4], 0)  # Padded

        # Check metadata
        total_packets = metadata[0]
        self.assertEqual(total_packets, 4)

    def test_thread_safe_iter(self):
        data = list(range(10))
        it = iter(data)
        safe_it = ThreadSafeIter(it)

        collected = []
        for val in safe_it:
            collected.append(val)

        self.assertEqual(collected, data)


if __name__ == "__main__":
    unittest.main()
