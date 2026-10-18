import os

# Same deterministic runtime as V1.5, set before importing torch.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.optim import Adam, SGD

from pykt.datasets.init_raetkt_dataset import init_raetkt_dataset4train
from pykt.models.raetkt import RAETKT_VERSION, RAETKT
from pykt.models.raetkt_train_model import train_raetkt_model, validate_set_loss_weight
from pykt.utils import set_seed

torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parents[1]


def build_parser():
    parser = argparse.ArgumentParser(description="RAET-KT: differentiable parallel evidence memory")
    parser.add_argument("--dataset_name", default="assist2017")
    parser.add_argument("--emb_type", choices=["qid"], default="qid")
    parser.add_argument("--save_dir", default="saved_model_raetkt")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=.1)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--num_attn_heads", type=int, default=4)
    parser.add_argument("--n_blocks", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--early_stop_rounds", type=int, default=10)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-3)
    parser.add_argument("--max_concepts", type=int, default=2)
    parser.add_argument("--sequence_mode", choices=["collapse", "preserve"], default="collapse")
    parser.add_argument("--evidence_prior_alpha", type=float, default=1.)
    parser.add_argument("--evidence_prior_beta", type=float, default=1.)
    parser.add_argument("--responsibility_temperature", type=float, default=1.0)
    parser.add_argument("--set_loss_weight", type=float, default=0.1)
    return parser


def main(params):
    params = dict(params)
    params.setdefault("responsibility_temperature", 1.0)
    params["set_loss_weight"] = validate_set_loss_weight(params["set_loss_weight"])
    params = dict(params, model_name="raetkt", architecture_version=RAETKT_VERSION,
                  responsibility_rule="normalized_failure_evidence",
                  correct_credit_rule="uniform_1_over_k",
                  incorrect_credit_rule="normalized_failure_responsibility",
                  set_level_partial_label=True,
                  question_aggregation="mean")
    set_seed(params["seed"])
    with (ROOT / "configs/kt_config.json").open(encoding="utf8") as stream:
        train_config = json.load(stream)["train_config"]
    train_config["batch_size"] = params["batch_size"]
    if params["num_epochs"] is not None:
        train_config["num_epochs"] = params["num_epochs"]
    with (ROOT / "configs/data_config.json").open(encoding="utf8") as stream:
        data_all = json.load(stream)
    data_config = data_all[params["dataset_name"]]
    # V1.5 paths are relative to examples/, independently of invocation cwd.
    data_config["dpath"] = str((ROOT / "examples" / data_config["dpath"]).resolve())
    train_loader, valid_loader, *_ = init_raetkt_dataset4train(
        params["dataset_name"], data_all, params["fold"], params["batch_size"],
        max_concepts=params["max_concepts"], sequence_mode=params["sequence_mode"],
        seed=params["seed"])
    model_config = {key: params[key] for key in (
        "d_model", "d_ff", "num_attn_heads", "n_blocks", "dropout", "emb_type",
        "evidence_prior_alpha", "evidence_prior_beta", "responsibility_temperature")}
    model_config.update(n_question=data_config["num_c"], n_pid=data_config["num_q"])
    model = RAETKT(**model_config).to("cuda" if torch.cuda.is_available() else "cpu")
    # Short, deterministic directory names avoid filesystem component limits.
    digest = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]
    checkpoint_dir = Path(params["save_dir"]) / (
        f"{params['dataset_name']}_fold{params['fold']}_seed{params['seed']}_{digest}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with (checkpoint_dir / "config.json").open("w", encoding="utf8") as stream:
        json.dump(dict(params=params, model_config=model_config, train_config=train_config,
                       data_config=data_config), stream, indent=2)
    if train_config["optimizer"] == "adam":
        optimizer = Adam(model.parameters(), lr=params["learning_rate"])
    elif train_config["optimizer"] == "sgd":
        optimizer = SGD(model.parameters(), lr=params["learning_rate"], momentum=.9)
    else:
        raise ValueError(f"Unsupported optimizer: {train_config['optimizer']}")
    print(f"sequence_mode: {params['sequence_mode']}")
    print(f"train smasks: {int(train_loader.dataset.dori['smasks'].sum())}")
    print(f"valid smasks: {int(valid_loader.dataset.dori['smasks'].sum())}")
    print(f"model_config: {model_config}")
    train_raetkt_model(
        model, train_loader, valid_loader, train_config["num_epochs"], optimizer,
        str(checkpoint_dir), save_model=True, early_stop_rounds=params["early_stop_rounds"],
        early_stop_min_delta=params["early_stop_min_delta"],
        set_loss_weight=params["set_loss_weight"])
    print(f"model_save_path: {checkpoint_dir / (params['emb_type'] + '_model.ckpt')}")


if __name__ == "__main__":
    main(vars(build_parser().parse_args()))
