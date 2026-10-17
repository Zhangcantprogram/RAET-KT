import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from .raplkt_data_loader import RAPLKTDataset
from .multikc_protocol import build_raw_q2cs_map


def _seed_worker(_worker_id):
    """Seed Python and NumPy from the deterministic DataLoader worker seed."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def init_raplkt_dataset4train(
    dataset_name,
    data_config,
    fold,
    batch_size,
    max_concepts=8,
    sequence_mode="collapse",
    seed=3407,
):
    cfg = data_config[dataset_name]
    all_folds = set(cfg["folds"])
    train_valid_path = os.path.join(cfg["dpath"], cfg["train_valid_file"])

    # q2cs_train_valid = RAPLKTDataset.build_q2cs_map(
    #     train_valid_path,
    #     folds=all_folds,
    # )

    curvalid = RAPLKTDataset(
        train_valid_path,
        cfg["input_type"],
        {fold},
        max_concepts=max_concepts,
        sequence_mode=sequence_mode,
        q2cs_map=None,
    )

    curtrain = RAPLKTDataset(
        train_valid_path,
        cfg["input_type"],
        all_folds - {fold},
        max_concepts=max_concepts,
        sequence_mode=sequence_mode,
        q2cs_map=None,
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(int(seed))
    valid_generator = torch.Generator()
    valid_generator.manual_seed(int(seed) + 1)

    train_loader = DataLoader(
        curtrain,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
        worker_init_fn=_seed_worker,
    )

    valid_loader = DataLoader(
        curvalid,
        batch_size=batch_size,
        shuffle=False,
        generator=valid_generator,
        worker_init_fn=_seed_worker,
    )

    return train_loader, valid_loader, None, None


def init_raplkt_test_datasets(
    data_config,
    batch_size,
    max_concepts=8,
    sequence_mode="collapse",
    seed=3407,
):
    test_path = os.path.join(data_config["dpath"], data_config["test_file"])
    test_window_path = os.path.join(data_config["dpath"], data_config["test_window_file"])
    test_original_path = os.path.join(
        data_config["dpath"],
        data_config["test_original_file"],
    )

    # 普通 test 不需要 q2cs_map。
    # test_sequences.csv 本身已经包含完整 multi-KC block。
    test_dataset = RAPLKTDataset(
        test_path,
        data_config["input_type"],
        {-1},
        max_concepts=max_concepts,
        sequence_mode=sequence_mode,
        q2cs_map=None,
    )

    # Window Test uses the complete original test metadata. This source is not
    # affected by min_seq_len tail dropping and the builder never reads labels.
    q2cs_test = build_raw_q2cs_map(
        test_original_path,
        folds={-1},
        max_concepts=max_concepts,
    )

    test_window_dataset = RAPLKTDataset(
        test_window_path,
        data_config["input_type"],
        {-1},
        max_concepts=max_concepts,
        sequence_mode=sequence_mode,
        q2cs_map=q2cs_test,
    )

    test_generator = torch.Generator()
    test_generator.manual_seed(int(seed))
    window_generator = torch.Generator()
    window_generator.manual_seed(int(seed) + 1)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        generator=test_generator,
        worker_init_fn=_seed_worker,
    )

    test_window_loader = DataLoader(
        test_window_dataset,
        batch_size=batch_size,
        shuffle=False,
        generator=window_generator,
        worker_init_fn=_seed_worker,
    )

    return test_loader, test_window_loader, None, None
