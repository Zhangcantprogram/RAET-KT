"""Preprocess Algebra2005, ASSIST2017, or the EdNet-KT1 5k subset."""

import argparse
from pathlib import Path

from pykt.preprocess import process_raw_data
from pykt.preprocess.split_datasets import main as split_concept
from pykt.preprocess.split_datasets_que import main as split_question


dname2paths = {
    "algebra2005": "../data/algebra2005/algebra_2005_2006_train.txt",
    "assist2017": "../data/assist2017/anonymized_full_release_competition_dataset.csv",
    "ednet_kt1_5k": "../data/ednet_kt1_5k/",
}
configf = "../configs/data_config.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset_name", choices=[*dname2paths, "ednet"],
        default="algebra2005",
    )
    parser.add_argument(
        "-f", "--file_path", default=None,
        help="Raw data file; for EdNet, a subset manifest JSON or its dataset directory.",
    )
    parser.add_argument("-m", "--min_seq_len", type=int, default=3)
    parser.add_argument("-l", "--maxlen", type=int, default=200)
    parser.add_argument("-k", "--kfold", type=int, default=5)
    args = parser.parse_args()

    if args.dataset_name == "ednet":
        args.dataset_name = "ednet_kt1_5k"
    if args.file_path is not None:
        dname2paths[args.dataset_name] = args.file_path

    dname, writef = process_raw_data(args.dataset_name, dname2paths)
    for cache in Path(dname).glob("*.pkl"):
        cache.unlink()

    split_concept(
        dname, writef, args.dataset_name, configf,
        args.min_seq_len, args.maxlen, args.kfold,
    )
    split_question(
        dname, writef, args.dataset_name, configf,
        args.min_seq_len, args.maxlen, args.kfold,
    )
