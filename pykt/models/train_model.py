import os
import torch
import numpy as np
from torch.nn.functional import binary_cross_entropy
from .evaluate_model import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cal_loss(model, ys, rshft, sm, preloss):
    y = torch.masked_select(ys[0], sm)
    t = torch.masked_select(rshft, sm)
    loss = binary_cross_entropy(y.double(), t.double()) + preloss[0]
    return loss

def model_forward(model, data):
    dcur = data

    q = dcur["qseqs"].to(device)
    c = dcur["cseqs"].to(device)
    r = dcur["rseqs"].to(device)

    qshft = dcur["shft_qseqs"].to(device)
    cshft = dcur["shft_cseqs"].to(device)
    rshft = dcur["shft_rseqs"].to(device)

    sm = dcur["smasks"].to(device)

    cq = torch.cat((q[:, 0:1], qshft), dim=1)
    cc = torch.cat((c[:, 0:1], cshft), dim=1)
    cr = torch.cat((r[:, 0:1], rshft), dim=1)

    if "cseqs_multi" in dcur and "cmasks_multi" in dcur:
        c_multi = dcur["cseqs_multi"].to(device)
        c_multi_shft = dcur["shft_cseqs_multi"].to(device)
        c_mask = dcur["cmasks_multi"].to(device)
        c_mask_shft = dcur["shft_cmasks_multi"].to(device)

        cc_multi = torch.cat((c_multi[:, 0:1, :], c_multi_shft), dim=1)
        cc_mask = torch.cat((c_mask[:, 0:1, :], c_mask_shft), dim=1)

        y, reg_loss = model(
            cc_multi.long(),
            cr.long(),
            cq.long(),
            c_mask=cc_mask.bool()
        )
    else:
        y, reg_loss = model(cc.long(), cr.long(), cq.long())

    ys = [y[:, 1:]]
    preloss = [reg_loss]

    loss = cal_loss(model, ys, rshft, sm, preloss)
    return loss

def train_model(
    model,
    train_loader,
    valid_loader,
    num_epochs,
    opt,
    ckpt_path,
    test_loader=None,
    test_window_loader=None,
    save_model=False,
    data_config=None,
    fold=None,
    early_stop_rounds=10,
    early_stop_min_delta=1e-3,
):
    best_auc, best_acc, best_epoch = float("-inf"), 0.0, -1
    patience_reference_auc = float("-inf")
    patience_best_epoch = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        loss_mean = []

        for data in train_loader:
            loss = model_forward(model, data)

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_mean.append(loss.detach().cpu().numpy())

        loss_mean = float(np.mean(loss_mean))

        validauc, validacc = evaluate(model, valid_loader, model.model_name)

        # True-best checkpoint selection is independent of the patience delta.
        if validauc > best_auc:
            if save_model:
                torch.save(
                    model.state_dict(),
                    os.path.join(ckpt_path, model.emb_type + "_model.ckpt")
                )

            best_auc = validauc
            best_acc = validacc
            best_epoch = epoch

        if validauc > patience_reference_auc + early_stop_min_delta:
            patience_reference_auc = validauc
            patience_best_epoch = epoch

        print(
            f"Epoch: {epoch}, "
            f"validauc: {validauc:.4f}, "
            f"validacc: {validacc:.4f}, "
            f"best_epoch: {best_epoch}, "
            f"best_auc: {best_auc:.4f}, "
            f"patience_epoch: {patience_best_epoch}, "
            f"train_loss: {loss_mean}, "
            f"emb_type: {model.emb_type}, "
            f"model: {model.model_name}, "
            f"save_dir: {ckpt_path}"
        )

        if epoch - patience_best_epoch >= early_stop_rounds:
            break

    return -1, -1, -1, -1, best_auc, best_acc, best_epoch
