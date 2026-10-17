import os
from torch.utils.data import DataLoader
from .data_loader import KTDataset
from .multikc_protocol import build_raw_q2cs_map

def init_dataset4train(
    dataset_name,
    model_name,
    data_config,
    fold,
    batch_size,
    diff_level=None,
    args=None,
    not_select_dataset=None,
    sequence_mode="original",
    max_concepts=8,
):
    if model_name != "akt":
        raise ValueError(f"Only AKT is supported in this minimal version, got: {model_name}")

    cfg = data_config[dataset_name]
    all_folds = set(cfg["folds"])

    train_valid_path = os.path.join(cfg["dpath"], cfg["train_valid_file"])

    curvalid = KTDataset(
        train_valid_path,
        cfg["input_type"],
        {fold},
        sequence_mode=sequence_mode,
        max_concepts=max_concepts,
    )

    curtrain = KTDataset(
        train_valid_path,
        cfg["input_type"],
        all_folds - {fold},
        sequence_mode=sequence_mode,
        max_concepts=max_concepts,
    )

    train_loader = DataLoader(curtrain, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(curvalid, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, None, None


def init_test_datasets(
    data_config,
    model_name,
    batch_size,
    diff_level=None,
    args=None,
    re_mapping=False,
    sequence_mode="original",
    max_concepts=8,
):
    if model_name != "akt":
        raise ValueError(f"Only AKT is supported in this minimal version, got: {model_name}")

    test_path = os.path.join(data_config["dpath"], data_config["test_file"])
    test_window_path = os.path.join(
        data_config["dpath"], data_config["test_window_file"]
    )
    test_original_path = os.path.join(
        data_config["dpath"], data_config["test_original_file"]
    )

    test_dataset = KTDataset(
        test_path,
        data_config["input_type"],
        {-1},
        sequence_mode=sequence_mode,
        max_concepts=max_concepts,
        q2cs_map=None,
    )

    window_q2cs = build_raw_q2cs_map(
        test_original_path,
        folds={-1},
        max_concepts=max_concepts,
    )

    test_window_dataset = KTDataset(
        test_window_path,
        data_config["input_type"],
        {-1},
        sequence_mode=sequence_mode,
        max_concepts=max_concepts,
        q2cs_map=window_q2cs,
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_window_loader = DataLoader(test_window_dataset, batch_size=batch_size, shuffle=False)

    # 如果你暂时不关心 question-level 指标，直接返回 None
    test_question_loader = None
    test_question_window_loader = None

    return test_loader, test_window_loader, test_question_loader, test_question_window_loader
