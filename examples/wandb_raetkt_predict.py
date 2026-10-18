import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import json
import math
from pathlib import Path

import torch

from pykt.datasets.init_raetkt_dataset import init_raetkt_test_datasets
from pykt.models.raetkt import RAETKT_VERSION, RAETKT
from pykt.models.raetkt_train_model import evaluate_raetkt, validate_set_loss_weight
from pykt.utils import set_seed

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(4)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main(params):
    if "set_loss_weight" in params:
        raise ValueError("Prediction cannot override training set_loss_weight.")
    if "responsibility_temperature" in params:
        raise ValueError("Prediction cannot override training responsibility temperature.")
    save_dir = Path(params["save_dir"])
    with (save_dir / "config.json").open(encoding="utf8") as stream:
        config = json.load(stream)
    trained = config["params"]
    if trained.get("model_name") != "raetkt":
        raise ValueError("This predictor only accepts model_name=raetkt.")
    if trained.get("architecture_version") != RAETKT_VERSION:
        raise ValueError("Checkpoint is not formal RAET-KT; retrain with the current training script.")
    weight = validate_set_loss_weight(trained["set_loss_weight"])
    if (trained.get("responsibility_rule") != "normalized_failure_evidence"
            or trained.get("correct_credit_rule") != "uniform_1_over_k"
            or trained.get("incorrect_credit_rule") != "normalized_failure_responsibility"
            or trained.get("set_level_partial_label") is not True):
        raise ValueError("Checkpoint does not declare the required set-level training protocol.")
    if trained.get("question_aggregation") != "mean":
        raise ValueError("Checkpoint does not use masked-mean KC aggregation.")
    set_seed(trained["seed"])
    with (ROOT / "configs/data_config.json").open(encoding="utf8") as stream:
        data_config = json.load(stream)[trained["dataset_name"]]
    data_config["dpath"] = str((ROOT / "examples" / data_config["dpath"]).resolve())
    data_config["dataset_name"] = trained["dataset_name"]
    model_config = config["model_config"]
    if (model_config["n_question"], model_config["n_pid"]) != (data_config["num_c"], data_config["num_q"]):
        raise ValueError("Checkpoint vocabulary sizes disagree with dataset configuration.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RAETKT(**model_config).to(device)
    checkpoint_path = save_dir / (trained["emb_type"] + "_model.ckpt")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint, strict=True)
    test_loader, window_loader, *_ = init_raetkt_test_datasets(
        data_config, params["bz"], max_concepts=trained["max_concepts"],
        sequence_mode=trained["sequence_mode"], seed=trained["seed"])
    report = {
        "model_name": "raetkt", "dataset_name": trained["dataset_name"],
        "fold": trained["fold"], "seed": trained["seed"],
        "sequence_mode": trained["sequence_mode"], "max_concepts": trained["max_concepts"],
        "architecture_version": RAETKT_VERSION,
        "question_aggregation": "mean",
        "responsibility_temperature": model.responsibility_temperature,
        "responsibility_rule": "normalized_failure_evidence",
        "correct_credit_rule": "uniform_1_over_k",
        "incorrect_credit_rule": "normalized_failure_responsibility",
        "set_level_partial_label": True, "set_loss_weight": weight,
        "success_budget": 1.0, "failure_budget": 1.0,
        "correct_update": "uniform_1_over_k",
        "evidence_prior_alpha": model.evidence_prior_alpha,
        "evidence_prior_beta": model.evidence_prior_beta,
        "checkpoint_path": str(checkpoint_path)}
    for split, loader in (("test", test_loader), ("window_test", window_loader)):
        report[split] = evaluate_raetkt(
            model, loader, save_path=str(save_dir / f"qid_{split}_predictions.txt"),
            split_name=split, return_group_metrics=True,
            debug_alignment=bool(params.get("debug_alignment", 0)),
            debug_responsibility=bool(params.get("debug_responsibility", 0)),
            responsibility_sample_size=params.get("responsibility_sample_size", 12),
            manifest_path=str(save_dir / f"{split}_manifest.tsv"), set_loss_weight=weight)
    with (save_dir / "qid_evaluation_metrics.json").open("w", encoding="utf8") as stream:
        json.dump(_json_safe(report), stream, indent=2, allow_nan=False)
    print("========== Final reported metric ==========")
    for split in ("test", "window_test"):
        for metric in ("auc", "acc"):
            print(f"{split}{metric} = {report[split]['overall'][metric]:.6f}")
    print("===========================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a formal RAET-KT checkpoint")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--bz", type=int, default=256)
    parser.add_argument("--debug_alignment", type=int, choices=[0, 1], default=0)
    parser.add_argument("--debug_responsibility", type=int, choices=[0, 1], default=0)
    parser.add_argument("--responsibility_sample_size", type=int, default=12)
    main(vars(parser.parse_args()))
