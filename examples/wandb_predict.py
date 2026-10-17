import os
import argparse
import json
import copy
import torch

from pykt.models import evaluate, load_model
from pykt.datasets import init_test_datasets

device = "cpu" if not torch.cuda.is_available() else "cuda"

def main(params):
    save_dir = params["save_dir"]
    batch_size = params["bz"]

    with open(os.path.join(save_dir, "config.json")) as fin:
        config = json.load(fin)

    model_config = copy.deepcopy(config["model_config"])

    for remove_item in [
        "use_wandb",
        "learning_rate",
        "add_uuid",
        "l2",
        "sequence_mode",
        "max_concepts"
    ]:
        if remove_item in model_config:
            del model_config[remove_item]

    trained_params = config["params"]
    model_name = trained_params["model_name"]
    dataset_name = trained_params["dataset_name"]
    emb_type = trained_params["emb_type"]
    sequence_mode = trained_params.get("sequence_mode", "original")
    max_concepts = trained_params.get("max_concepts", 8)

    if model_name != "akt":
        raise ValueError(f"Only AKT is supported in this minimal version, got: {model_name}")

    with open("../configs/data_config.json") as fin:
        data_config_all = json.load(fin)

    data_config = data_config_all[dataset_name]
    data_config["dataset_name"] = dataset_name

    test_loader, test_window_loader, _, _ = init_test_datasets(
        data_config,
        model_name,
        batch_size,
        sequence_mode=sequence_mode,
        max_concepts=max_concepts,
    )

    print(f"sequence_mode: {sequence_mode}")
    print(f"max_concepts: {max_concepts}")

    def print_loader_info(name, loader):
        print(f"{name} dataset len:", len(loader.dataset))

        if hasattr(loader.dataset, "dori") and "smasks" in loader.dataset.dori:
            print(f"{name} smasks:", loader.dataset.dori["smasks"].sum().item())
        else:
            print(f"{name} smasks: NOT FOUND")

    print_loader_info("test", test_loader)
    print_loader_info("test_window", test_window_loader)

    model = load_model(model_name, model_config, data_config, emb_type, save_dir)

    save_test_path = os.path.join(save_dir, model.emb_type + "_test_predictions.txt")
    testauc, testacc = evaluate(
        model, test_loader, model_name, save_path=save_test_path, split_name="test"
    )

    save_test_window_path = os.path.join(save_dir, model.emb_type + "_test_window_predictions.txt")
    window_testauc, window_testacc = evaluate(
        model,
        test_window_loader,
        model_name,
        save_path=save_test_window_path,
        split_name="window_test",
        manifest_path=os.path.join(save_dir, "window_test_manifest.tsv"),
    )

    print("========== Final reported metric ==========")
    print(f"testauc = {testauc:.6f}")
    print(f"testacc = {testacc:.6f}")
    print(f"window_testauc = {window_testauc:.6f}")
    print(f"window_testacc = {window_testacc:.6f}")
    print("===========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bz", type=int, default=256)
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()

    main(vars(args))
