import os
import torch
from .akt import AKT

device = "cpu" if not torch.cuda.is_available() else "cuda"

def init_model(model_name, model_config, data_config, emb_type):
    if model_name != "akt":
        raise ValueError(f"Only AKT is supported in this minimal version, got: {model_name}")

    model = AKT(
        data_config["num_c"],
        data_config["num_q"],
        **model_config,
        emb_type=emb_type,
        emb_path=data_config.get("emb_path", "")
    ).to(device)

    return model

def load_model(model_name, model_config, data_config, emb_type, ckpt_path):
    model = init_model(model_name, model_config, data_config, emb_type)
    net = torch.load(
        os.path.join(ckpt_path, emb_type + "_model.ckpt"),
        map_location=device
    )
    model.load_state_dict(net)
    return model