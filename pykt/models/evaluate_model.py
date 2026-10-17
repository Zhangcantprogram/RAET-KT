import numpy as np
import torch

from .item_level_metrics import (
    compute_item_level_metrics,
    print_item_level_metrics,
    write_benchmark_manifest,
)


device = "cpu" if not torch.cuda.is_available() else "cuda"


def evaluate(model, test_loader, model_name, rel=None, save_path="",
             split_name=None, return_group_metrics=False, manifest_path=None):
    if model_name != "akt":
        raise ValueError(f"Only AKT is supported in this minimal version, got: {model_name}")
    fout = open(save_path, "w", encoding="utf8") if save_path else None
    model.eval()
    y_trues, y_scores, k_counts = [], [], []
    manifest_records = []

    with torch.no_grad():
        for dcur in test_loader:
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

            has_multi = "cseqs_multi" in dcur and "cmasks_multi" in dcur
            if has_multi:
                c_multi = dcur["cseqs_multi"].to(device)
                c_multi_shft = dcur["shft_cseqs_multi"].to(device)
                c_mask = dcur["cmasks_multi"].to(device)
                c_mask_shft = dcur["shft_cmasks_multi"].to(device)
                cc_multi = torch.cat((c_multi[:, 0:1, :], c_multi_shft), dim=1)
                cc_mask = torch.cat((c_mask[:, 0:1, :], c_mask_shft), dim=1)
                y, _ = model(cc_multi.long(), cr.long(), cq.long(), c_mask=cc_mask.bool())
                target_k_count = cc_mask[:, 1:, :].sum(dim=-1)
            else:
                y, _ = model(cc.long(), cr.long(), cq.long())
                target_k_count = torch.ones_like(rshft, dtype=torch.long)

            pred = torch.masked_select(y[:, 1:], sm).detach().cpu()
            true = torch.masked_select(rshft, sm).detach().cpu()
            selected_k_count = torch.masked_select(target_k_count, sm).detach().cpu()
            if not (pred.numel() == true.numel() == selected_k_count.numel()):
                raise AssertionError("AKT prediction/target/full-KC-mask alignment failed.")

            if manifest_path is not None:
                shifted_cidx = dcur.get("shft_cidxs")
                if shifted_cidx is None:
                    raise AssertionError(
                        "Benchmark manifest requires shft_cidxs; refusing to "
                        "write placeholder target identities."
                    )
                shifted_cidx = shifted_cidx.to(device)
                selected_cidx = torch.masked_select(shifted_cidx, sm)
                if torch.any(selected_cidx < 0):
                    raise AssertionError(
                        "Benchmark manifest target_cidx must be non-negative."
                    )
                id_offset = int(getattr(test_loader.dataset, "id_offset", 0))
                for record_index, (batch_index, position) in enumerate(
                        sm.nonzero(as_tuple=False).tolist()):
                    if has_multi:
                        valid_kc = cc_mask[batch_index, position + 1]
                        kc_values = cc_multi[batch_index, position + 1, valid_kc]
                    else:
                        kc_values = cc[batch_index, position + 1].reshape(1)
                    raw_kc_ids = sorted(
                        int(value) - id_offset
                        for value in kc_values.detach().cpu().tolist()
                    )
                    manifest_records.append({
                        "target_cidx": int(shifted_cidx[batch_index, position]),
                        "question_id": int(cq[batch_index, position + 1]) - id_offset,
                        "response": int(true[record_index]),
                        "kc_ids": ",".join(str(value) for value in raw_kc_ids),
                        "k_count": int(selected_k_count[record_index]),
                        "prediction": float(pred[record_index]),
                    })

            y_scores.append(pred.numpy())
            y_trues.append(true.numpy())
            k_counts.append(selected_k_count.numpy())
            if fout is not None:
                for target, prediction in zip(true.numpy(), pred.numpy()):
                    fout.write(f"{int(target)}\t{float(prediction)}\n")

    if fout is not None:
        fout.close()
    if manifest_path is not None:
        write_benchmark_manifest(manifest_path, manifest_records)
    ts = np.concatenate(y_trues, axis=0) if y_trues else np.array([])
    ps = np.concatenate(y_scores, axis=0) if y_scores else np.array([])
    ks = np.concatenate(k_counts, axis=0) if k_counts else np.array([])
    group_metrics = compute_item_level_metrics(ts, ps, ks)
    if split_name is not None:
        print_item_level_metrics(group_metrics, split_name)
    if return_group_metrics:
        return group_metrics
    overall = group_metrics["overall"]
    return overall["auc"], overall["acc"]
