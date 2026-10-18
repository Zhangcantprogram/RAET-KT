"""Raw-data adapters for the three supported datasets."""

from pathlib import Path


def process_raw_data(dataset_name, dname2paths):
    if dataset_name == "algebra2005":
        from .algebra2005_preprocess import read_data_from_csv
    elif dataset_name == "assist2017":
        from .assist2017_preprocess import read_data_from_csv
    elif dataset_name in {"ednet", "ednet_kt1_5k"}:
        from .ednet_kt1_preprocess_20260826_010359 import read_data_from_csv
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    readf = Path(dname2paths[dataset_name])
    dname = readf if readf.is_dir() else readf.parent
    writef = dname / "data.txt"
    print(f"Start preprocessing data: {dataset_name}")
    read_data_from_csv(str(readf), str(writef))
    return str(dname), str(writef)
