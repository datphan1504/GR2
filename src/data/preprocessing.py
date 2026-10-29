"""Data preprocessing pipelines for network traffic traces.

Transforms raw packet traces (CSV or Wang format) into structured sequences
(direction, inter-arrival time) and summary metadata, saving final sets into
compressed HDF5 files.
"""

import os
import random
import time
from typing import Tuple, Dict, Any, List, Optional

import h5py
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

try:
    from tensorflow.keras.utils import to_categorical
except ImportError:
    try:
        from keras.utils.np_utils import to_categorical
    except ImportError:
        def to_categorical(y, num_classes=None):
            y = np.array(y, dtype='int')
            if num_classes is None:
                num_classes = np.max(y) + 1
            n = y.shape[0]
            categorical = np.zeros((n, num_classes), dtype='float32')
            categorical[np.arange(n), y] = 1
            return categorical

from src.utils.helpers import get_logger, ensure_dir

logger = get_logger("DataPreprocessing")


def process_trace(trace_file_path: str, seq_length: int = 5000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Extracts direction sequence, timestamp sequence, metadata features, and label
    from a Wang-formatted traffic trace.
    """
    with open(trace_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [line.strip() for line in f if line.strip()]

    dir_seq = np.zeros(seq_length, dtype=np.int8)
    time_seq = np.zeros(seq_length, dtype=np.float32)

    base_name = os.path.basename(trace_file_path)
    label = 0 if '-' not in base_name else int(base_name.split('-')[0]) + 1

    if not lines:
        metadata = np.zeros(7, dtype=np.float32)
        return dir_seq, time_seq, metadata, label

    total_time = float(lines[-1].split('\t')[0])
    total_incoming = 0
    total_outgoing = 0

    for packet_num, line in enumerate(lines):
        parts = line.split('\t')
        if len(parts) < 2:
            continue
        curr_time = float(parts[0])
        curr_dir = int(np.sign(float(parts[1])))

        if packet_num < seq_length:
            dir_seq[packet_num] = curr_dir
            time_seq[packet_num] = curr_time

        if curr_dir > 0:
            total_outgoing += 1
        elif curr_dir < 0:
            total_incoming += 1

    total_packets = total_incoming + total_outgoing
    if total_packets == 0:
        metadata = np.zeros(7, dtype=np.float32)
    else:
        metadata = np.array([
            total_packets,
            total_incoming,
            total_outgoing,
            total_incoming / total_packets,
            total_outgoing / total_packets,
            total_time,
            total_time / total_packets
        ], dtype=np.float32)

    return dir_seq, time_seq, metadata, label


def process_csv_trace(csv_file_path: str, seq_length: int = 5000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts direction sequence, cumulative timestamp sequence, and 7 metadata features
    from a semicolon-delimited CSV trace (protocol;length;relative_time;direction;...).
    """
    dir_seq = np.zeros(seq_length, dtype=np.int8)
    time_seq = np.zeros(seq_length, dtype=np.float32)

    total_incoming = 0
    total_outgoing = 0
    cum_time = 0.0
    packet_num = 0

    with open(csv_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line_idx, raw_line in enumerate(f):
            line = raw_line.strip()
            if not line:
                continue
            # Skip header line
            if line_idx == 0 and line.lower().startswith("protocol"):
                continue

            parts = line.split(';')
            if len(parts) < 4:
                continue

            try:
                delta_t = max(0.0, float(parts[2]))
                dir_flag = int(parts[3])
            except ValueError:
                continue

            cum_time += delta_t
            curr_dir = 1 if dir_flag == 1 else -1

            if packet_num < seq_length:
                dir_seq[packet_num] = curr_dir
                time_seq[packet_num] = cum_time

            if curr_dir > 0:
                total_outgoing += 1
            else:
                total_incoming += 1

            packet_num += 1

    total_packets = total_incoming + total_outgoing
    if total_packets == 0:
        metadata = np.zeros(7, dtype=np.float32)
    else:
        metadata = np.array([
            total_packets,
            total_incoming,
            total_outgoing,
            total_incoming / total_packets,
            total_outgoing / total_packets,
            cum_time,
            cum_time / total_packets
        ], dtype=np.float32)

    return dir_seq, time_seq, metadata


def find_csv_root_dir(config: Dict[str, Any]) -> Optional[str]:
    """Searches common locations for the captured CSV trace directory."""
    raw_data_dir = config.get("raw_data_dir", "data/raw/")
    data_dir = config.get("data_dir", "data/processed/")
    candidates = [
        config.get("csv_data_dir"),
        os.path.join(raw_data_dir, "csv"),
        os.path.join(raw_data_dir, "traces", "csv"),
        os.path.join(raw_data_dir, "data_capture", "traces", "csv"),
        os.path.join(data_dir, "csv"),
    ]
    for cand in candidates:
        if cand and os.path.isdir(cand):
            subdirs = [d for d in os.listdir(cand) if os.path.isdir(os.path.join(cand, d))]
            if subdirs:
                return cand
    return None


def convert_wang_to_npz(config: Dict[str, Any], raw_batch_dir: str = None) -> None:
    """
    Processes all Wang-format traces from raw directory and compiles them
    into all_closed_world.npz and all_open_world.npz.
    """
    data_dir = ensure_dir(config.get("data_dir", "data/processed/"))
    raw_data_dir = config.get("raw_data_dir", "data/raw/")
    if raw_batch_dir is None:
        raw_batch_dir = os.path.join(raw_data_dir, "batch_wang")
        if not os.path.exists(raw_batch_dir):
            raw_batch_dir = os.path.join(data_dir, "batch_wang")

    if not os.path.exists(raw_batch_dir):
        logger.warning(f"Raw Wang trace directory not found at {raw_batch_dir}. Skipping Wang-to-NPZ conversion.")
        return

    num_mon_sites = config['num_mon_sites']
    num_mon_inst_test = config['num_mon_inst_test']
    num_mon_inst_train = config['num_mon_inst_train']
    num_mon_inst = num_mon_inst_test + num_mon_inst_train
    num_unmon_sites_test = config['num_unmon_sites_test']
    num_unmon_sites_train = config['num_unmon_sites_train']
    num_unmon_sites = num_unmon_sites_test + num_unmon_sites_train
    seq_length = config.get("seq_length", 5000)

    trace_files = [os.path.join(raw_batch_dir, f) for f in os.listdir(raw_batch_dir)]
    logger.info(f"Total raw traces found: {len(trace_files)}")

    dir_seq_mon, time_seq_mon, metadata_mon, labels_mon = [], [], [], []
    dir_seq_unmon, time_seq_unmon, metadata_unmon, labels_unmon = [], [], [], []

    for trace_path in tqdm(trace_files, desc="Parsing raw traces"):
        dir_seq, time_seq, metadata, label = process_trace(trace_path, seq_length=seq_length)
        if label == 0:
            if len(labels_unmon) < num_unmon_sites:
                dir_seq_unmon.append(dir_seq)
                time_seq_unmon.append(time_seq)
                metadata_unmon.append(metadata)
                labels_unmon.append(label)
        else:
            if len(labels_mon) < (num_mon_sites * num_mon_inst):
                dir_seq_mon.append(dir_seq)
                time_seq_mon.append(time_seq)
                metadata_mon.append(metadata)
                labels_mon.append(label)

    closed_path = os.path.join(data_dir, "all_closed_world.npz")
    np.savez_compressed(
        closed_path,
        dir_seq=np.array(dir_seq_mon, dtype=np.int8),
        time_seq=np.array(time_seq_mon, dtype=np.float32),
        metadata=np.array(metadata_mon, dtype=np.float32),
        labels=np.array(labels_mon)
    )
    logger.info(f"Saved monitored traces to {closed_path}")

    open_path = os.path.join(data_dir, "all_open_world.npz")
    np.savez_compressed(
        open_path,
        dir_seq=np.array(dir_seq_unmon, dtype=np.int8),
        time_seq=np.array(time_seq_unmon, dtype=np.float32),
        metadata=np.array(metadata_unmon, dtype=np.float32),
        labels=np.array(labels_unmon)
    )
    logger.info(f"Saved unmonitored traces to {open_path}")


def _save_splits_to_h5(
    output_h5_path: str,
    train_seq_and_labels: List[List[Any]],
    test_seq_and_labels: List[List[Any]],
    num_classes: int,
    inter_time: bool = True,
    scale_metadata: bool = True,
    val_split: float = 0.05
) -> str:
    """Transforms, scales, splits into train/val/test, and writes HDF5 dataset."""
    random.shuffle(train_seq_and_labels)
    random.shuffle(test_seq_and_labels)

    train_dir = np.array([x[0] for x in train_seq_and_labels], dtype=np.int8)
    train_time = np.array([x[1] for x in train_seq_and_labels], dtype=np.float32)
    train_metadata = np.array([x[2] for x in train_seq_and_labels], dtype=np.float32)
    train_labels = [x[3] for x in train_seq_and_labels]

    test_dir = np.array([x[0] for x in test_seq_and_labels], dtype=np.int8)
    test_time = np.array([x[1] for x in test_seq_and_labels], dtype=np.float32)
    test_metadata = np.array([x[2] for x in test_seq_and_labels], dtype=np.float32)
    test_labels = [x[3] for x in test_seq_and_labels]

    if inter_time:
        if len(train_time) > 0:
            inter_train = np.zeros_like(train_time)
            inter_train[:, 1:] = train_time[:, 1:] - train_time[:, :-1]
            train_time = inter_train

        if len(test_time) > 0:
            inter_test = np.zeros_like(test_time)
            inter_test[:, 1:] = test_time[:, 1:] - test_time[:, :-1]
            test_time = inter_test

    if len(train_dir) > 0:
        train_dir = np.reshape(train_dir, (train_dir.shape[0], train_dir.shape[1], 1))
        train_time = np.reshape(train_time, (train_time.shape[0], train_time.shape[1], 1))
    if len(test_dir) > 0:
        test_dir = np.reshape(test_dir, (test_dir.shape[0], test_dir.shape[1], 1))
        test_time = np.reshape(test_time, (test_time.shape[0], test_time.shape[1], 1))

    if scale_metadata and len(train_metadata) > 0 and len(test_metadata) > 0:
        scaler = StandardScaler()
        train_metadata = scaler.fit_transform(train_metadata)
        test_metadata = scaler.transform(test_metadata)

    train_labels = to_categorical(train_labels, num_classes=num_classes)
    test_labels = to_categorical(test_labels, num_classes=num_classes)

    split_idx = int((1.0 - val_split) * len(train_dir))

    with h5py.File(output_h5_path, 'w') as f:
        f.create_group('training_data')
        f.create_group('validation_data')
        f.create_group('test_data')

        for ds_name, arr in [('dir_seq', train_dir), ('time_seq', train_time),
                             ('metadata', train_metadata), ('labels', train_labels)]:
            f.create_dataset(f'training_data/{ds_name}', data=arr[:split_idx])
            f.create_dataset(f'validation_data/{ds_name}', data=arr[split_idx:])

        for ds_name, arr in [('dir_seq', test_dir), ('time_seq', test_time),
                             ('metadata', test_metadata), ('labels', test_labels)]:
            f.create_dataset(f'test_data/{ds_name}', data=arr)

    return output_h5_path


def preprocess_from_csv_dir(config: Dict[str, Any], csv_root_dir: str, output_h5_path: str) -> str:
    """
    Builds the HDF5 dataset directly from website CSV folders (csv/<website>/<idx>_<website>.csv).
    - First `num_mon_sites` folders are used as monitored sites (labels 0 .. num_mon_sites-1).
    - If Open-World (`num_unmon_sites > 0`), remaining folders are used as unmonitored sites (label = num_mon_sites).
    """
    num_mon_sites = config['num_mon_sites']
    num_mon_inst_train = config['num_mon_inst_train']
    num_mon_inst_test = config['num_mon_inst_test']
    num_mon_inst = num_mon_inst_train + num_mon_inst_test
    num_unmon_sites_train = config.get('num_unmon_sites_train', 0)
    num_unmon_sites_test = config.get('num_unmon_sites_test', 0)
    num_unmon_sites = num_unmon_sites_train + num_unmon_sites_test
    seq_length = config.get('seq_length', 5000)

    site_dirs = sorted([
        d for d in os.listdir(csv_root_dir)
        if os.path.isdir(os.path.join(csv_root_dir, d))
    ])

    if len(site_dirs) < num_mon_sites:
        raise ValueError(
            f"Found {len(site_dirs)} website folders in {csv_root_dir}, "
            f"but config requires num_mon_sites={num_mon_sites}. "
            f"Please reduce num_mon_sites in your config file to <= {len(site_dirs)}."
        )

    mon_dirs = site_dirs[:num_mon_sites]
    unmon_dirs = site_dirs[num_mon_sites:]

    logger.info(
        f"Parsing CSV traces from {csv_root_dir}: "
        f"{len(mon_dirs)} monitored site folders, {len(unmon_dirs)} unmonitored site folders."
    )
    start_time = time.time()

    train_seq_and_labels = []
    test_seq_and_labels = []

    for site_label, site_name in enumerate(tqdm(mon_dirs, desc="Processing monitored sites")):
        site_path = os.path.join(csv_root_dir, site_name)
        csv_files = sorted([
            os.path.join(site_path, f) for f in os.listdir(site_path)
            if f.lower().endswith('.csv')
        ])
        random.shuffle(csv_files)

        selected_files = csv_files[:num_mon_inst]
        if len(selected_files) < num_mon_inst:
            # Split proportionally if a site has slightly fewer instances
            actual_test = min(num_mon_inst_test, max(1, len(selected_files) - num_mon_inst_train))
            actual_train = len(selected_files) - actual_test
        else:
            actual_train = num_mon_inst_train

        for idx, fpath in enumerate(selected_files):
            d_seq, t_seq, meta = process_csv_trace(fpath, seq_length=seq_length)
            item = [d_seq, t_seq, meta, site_label]
            if idx < actual_train:
                train_seq_and_labels.append(item)
            else:
                test_seq_and_labels.append(item)

    if num_unmon_sites > 0:
        if not unmon_dirs:
            raise ValueError(
                f"Open-World requires unmonitored sites (num_unmon_sites={num_unmon_sites}), "
                f"but all {len(site_dirs)} folders in {csv_root_dir} were used as monitored sites. "
                f"Reduce num_mon_sites (e.g. to 40) so remaining folders serve as unmonitored sites."
            )

        all_unmon_files = []
        for site_name in unmon_dirs:
            site_path = os.path.join(csv_root_dir, site_name)
            all_unmon_files.extend([
                os.path.join(site_path, f) for f in os.listdir(site_path)
                if f.lower().endswith('.csv')
            ])

        random.shuffle(all_unmon_files)
        selected_unmon = all_unmon_files[:num_unmon_sites]

        if len(selected_unmon) < num_unmon_sites:
            logger.warning(
                f"Requested {num_unmon_sites} unmonitored traces, found {len(selected_unmon)}. "
                f"Splitting proportionally between train and test."
            )
            ratio = num_unmon_sites_train / float(num_unmon_sites)
            actual_unmon_train = int(round(len(selected_unmon) * ratio))
        else:
            actual_unmon_train = num_unmon_sites_train

        for idx, fpath in enumerate(tqdm(selected_unmon, desc="Processing unmonitored traces")):
            d_seq, t_seq, meta = process_csv_trace(fpath, seq_length=seq_length)
            item = [d_seq, t_seq, meta, num_mon_sites]
            if idx < actual_unmon_train:
                train_seq_and_labels.append(item)
            else:
                test_seq_and_labels.append(item)

    num_classes = num_mon_sites if num_unmon_sites == 0 else num_mon_sites + 1
    _save_splits_to_h5(
        output_h5_path=output_h5_path,
        train_seq_and_labels=train_seq_and_labels,
        test_seq_and_labels=test_seq_and_labels,
        num_classes=num_classes,
        inter_time=config.get('inter_time', True),
        scale_metadata=config.get('scale_metadata', True),
        val_split=config.get('validation_split', 0.05)
    )
    logger.info(f"Finished saving {output_h5_path} from CSV traces in {time.time() - start_time:.2f} seconds.")
    return output_h5_path


def preprocess_pipeline(config: Dict[str, Any]) -> str:
    """
    Main preprocessing pipeline that:
    1. Checks for captured CSV trace folders or compiled .npz datasets.
    2. Performs train/test splitting.
    3. Calculates inter-arrival timings if configured.
    4. Scales metadata with StandardScaler.
    5. Saves training, validation, and test sets into HDF5 format.

    Returns:
        output_h5_path: Path to the generated .h5 file.
    """
    num_mon_sites = config['num_mon_sites']
    num_mon_inst_test = config['num_mon_inst_test']
    num_mon_inst_train = config['num_mon_inst_train']
    num_mon_inst = num_mon_inst_test + num_mon_inst_train
    num_unmon_sites_test = config['num_unmon_sites_test']
    num_unmon_sites_train = config['num_unmon_sites_train']
    num_unmon_sites = num_unmon_sites_test + num_unmon_sites_train

    inter_time = config.get('inter_time', True)
    scale_metadata = config.get('scale_metadata', True)
    data_dir = ensure_dir(config.get('data_dir', 'data/processed/'))

    output_h5_filename = f"{num_mon_sites}_{num_mon_inst}_{num_unmon_sites_train}_{num_unmon_sites_test}.h5"
    output_h5_path = os.path.join(data_dir, output_h5_filename)

    # Priority 1: Use captured CSV dataset if available
    csv_root_dir = find_csv_root_dir(config)
    if csv_root_dir is not None:
        return preprocess_from_csv_dir(config, csv_root_dir, output_h5_path)

    # Priority 2: Use NPZ archives or convert from Wang format
    mon_data_loc = os.path.join(data_dir, 'all_closed_world.npz')
    unmon_data_loc = os.path.join(data_dir, 'all_open_world.npz')

    if not os.path.exists(mon_data_loc) or (num_unmon_sites > 0 and not os.path.exists(unmon_data_loc)):
        logger.info("NPZ archives not found. Attempting conversion from raw Wang files...")
        convert_wang_to_npz(config)

    if not os.path.exists(mon_data_loc) or (num_unmon_sites > 0 and not os.path.exists(unmon_data_loc)):
        raise FileNotFoundError(
            f"Dataset files not found! Please place your CSV traces in 'data/raw/csv/' "
            f"(or 'data/raw/data_capture/traces/csv/') or place 'all_closed_world.npz' "
            f"and 'all_open_world.npz' in '{data_dir}'."
        )

    logger.info(f"Starting data preprocessing for {output_h5_filename}...")
    start_time = time.time()

    mon_dataset = np.load(mon_data_loc)
    mon_dir_seq = mon_dataset['dir_seq']
    mon_time_seq = mon_dataset['time_seq']
    mon_metadata = mon_dataset['metadata']
    mon_labels = mon_dataset['labels']

    train_seq_and_labels = []
    test_seq_and_labels = []

    mon_site_data: Dict[Any, List] = {}
    mon_site_labels: Dict[Any, int] = {}

    for dir_seq, time_seq, metadata, site_name in zip(mon_dir_seq, mon_time_seq, mon_metadata, mon_labels):
        if site_name not in mon_site_data:
            if len(mon_site_data) >= num_mon_sites:
                continue
            mon_site_data[site_name] = []
            mon_site_labels[site_name] = len(mon_site_labels)

        mon_site_data[site_name].append([dir_seq, time_seq, metadata, mon_site_labels[site_name]])

    for instances in mon_site_data.values():
        random.shuffle(instances)
        for inst_num, item in enumerate(instances):
            if inst_num < num_mon_inst_train:
                train_seq_and_labels.append(item)
            elif inst_num < num_mon_inst:
                test_seq_and_labels.append(item)
            else:
                break

    del mon_dataset, mon_dir_seq, mon_time_seq, mon_metadata, mon_labels, mon_site_data, mon_site_labels

    if os.path.exists(unmon_data_loc) and num_unmon_sites > 0:
        unmon_dataset = np.load(unmon_data_loc)
        unmon_dir_seq = unmon_dataset['dir_seq']
        unmon_time_seq = unmon_dataset['time_seq']
        unmon_metadata = unmon_dataset['metadata']

        unmon_site_data = [
            [d, t, m, num_mon_sites]
            for d, t, m in zip(unmon_dir_seq, unmon_time_seq, unmon_metadata)
        ]
        random.shuffle(unmon_site_data)

        for inst_num, item in enumerate(unmon_site_data):
            if inst_num < num_unmon_sites_train:
                train_seq_and_labels.append(item)
            elif inst_num < num_unmon_sites:
                test_seq_and_labels.append(item)
            else:
                break

        del unmon_dataset, unmon_dir_seq, unmon_time_seq, unmon_metadata, unmon_site_data

    num_classes = num_mon_sites if num_unmon_sites == 0 else num_mon_sites + 1
    _save_splits_to_h5(
        output_h5_path=output_h5_path,
        train_seq_and_labels=train_seq_and_labels,
        test_seq_and_labels=test_seq_and_labels,
        num_classes=num_classes,
        inter_time=inter_time,
        scale_metadata=scale_metadata,
        val_split=config.get("validation_split", 0.05)
    )

    logger.info(f"Finished saving {output_h5_path} in {time.time() - start_time:.2f} seconds.")
    return output_h5_path
