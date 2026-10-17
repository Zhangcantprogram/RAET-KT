"""Item-level training and evaluation for RAPLKT on the V1.5 protocol."""
import os
import json
import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .item_level_metrics import (
    compute_item_level_metrics, print_item_level_metrics, write_benchmark_manifest, KC_GROUPS,
)


def validate_set_loss_weight(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("set_loss_weight must be finite and >= 0.")
    return value


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _log1mexp(x):
    """Stable log(1-exp(x)) for x <= 0, with safe inactive branches.

    Clamping each branch to its own domain avoids 0 * inf in where's
    backward when exp(x) would round to one in the inactive log1p branch.
    """
    log_half = -0.6931471805599453
    return torch.where(x < log_half,
                       torch.log1p(-torch.exp(x.clamp_max(log_half))),
                       torch.log(-torch.expm1(x.clamp_min(log_half))))


def _set_log_correct(kc_probs, c_mask, eps=1e-8):
    if not math.isfinite(eps) or not 0 < eps < .5:
        raise ValueError("eps must be finite and in (0, 0.5).")
    if kc_probs.shape != c_mask.shape:
        raise ValueError("KC probabilities and mask must have the same shape.")
    # At least fp32 arithmetic. In fp32, 1-1e-8 rounds to 1: use a
    # representable upper bound so saturated sigmoid outputs stay finite.
    probs = kc_probs.float() if kc_probs.dtype in (torch.float16, torch.bfloat16) else kc_probs
    upper_eps = max(eps, torch.finfo(probs.dtype).eps)
    safe_p = probs.clamp(min=eps, max=1.0-upper_eps)
    logs = torch.where(c_mask.bool(), safe_p.log(), torch.zeros_like(safe_p))
    return logs.sum(dim=-1)


def compute_set_level_partial_loss(kc_probs, target, c_mask, eps=1e-8):
    """Return per-item NLL [...], from p/mask [...,K] and one target [...].

    The conjunction likelihood supervises the candidate set once. There
    are no per-KC copied labels. Empty unselected padding returns zero;
    the trainer rejects selected targets without a valid KC.
    """
    if target.shape != kc_probs.shape[:-1]:
        raise ValueError("Item target shape must match KC probabilities without K.")
    log_correct = _set_log_correct(kc_probs, c_mask, eps)
    valid = c_mask.bool().any(-1)
    # Never evaluate log1mexp(0) for an empty set, even if masked later.
    safe_log = torch.where(valid, log_correct, torch.full_like(log_correct, -1.))
    loss = torch.where(target.bool(), -safe_log, -_log1mexp(safe_log))
    return torch.where(valid, loss, torch.zeros_like(loss))


def _build_full_sequences(data, device):
    """V1.5 reconstruction, without an auxiliary-supervision mask."""
    def joined(key):
        return torch.cat((data[key][:, :1], data["shft_" + key]), dim=1).to(device)
    cq, cr = joined("qseqs"), joined("rseqs")
    cc, cm = joined("cseqs_multi"), joined("cmasks_multi")
    updates = joined("evidence_update_seqs")
    sm = data["smasks"].to(device)
    for name, value in (("response", cr), ("smasks", sm),
                        ("evidence_update_mask", updates)):
        if torch.any((value != 0) & (value != 1)):
            raise ValueError(f"{name} must be binary.")
    return cq, cc, cm, cr, data["shft_rseqs"].to(device), sm.bool(), updates.bool()


def raplkt_model_forward(model, data, set_loss_weight=0.1, return_losses=False):
    set_loss_weight = validate_set_loss_weight(set_loss_weight)
    cq, cc, cm, cr, shifted_r, sm, updates = _build_full_sequences(
        data, next(model.parameters()).device)
    y, reg_loss, details = model(cc, cr, cq, c_mask=cm,
                                 evidence_update_mask=updates, return_details=True)
    if not sm.any():
        raise ValueError("Training batch contains no selected prediction targets.")
    kt_loss = F.binary_cross_entropy(y[:, 1:][sm].float(), shifted_r[sm].float())
    if torch.any(cm[:, 1:].sum(-1)[sm] <= 0):
        raise AssertionError("Set-level supervised item has no valid KC.")
    # Select supervised shifted items BEFORE the likelihood: unselected items,
    # including the first position, cannot contribute values or gradients.
    set_loss = compute_set_level_partial_loss(
        details["kc_probs"][:, 1:][sm], shifted_r[sm], cm[:, 1:][sm], model.eps).mean()
    total = kt_loss + reg_loss + set_loss_weight * set_loss
    if return_losses:
        return total, {"kt": kt_loss, "set": set_loss, "reg": reg_loss, "n": int(sm.sum())}
    return total


def compute_responsibility_statistics(entropy, effective_k, k_count, tv, max_rho):
    arrays = [np.asarray(v).reshape(-1) for v in (entropy, effective_k, k_count, tv, max_rho)]
    if len({len(v) for v in arrays}) != 1:
        raise ValueError("Responsibility diagnostic arrays must be aligned.")
    entropy, effective_k, k_count, tv, max_rho = arrays
    result = {}
    for name, mask in (("k2", k_count == 2), ("k3plus", k_count >= 3),
                       ("multi", k_count >= 2)):
        values = {"n_incorrect": int(mask.sum())}
        for key, array in (("responsibility_entropy", entropy), ("effective_k", effective_k),
                           ("normalized_effective_k", effective_k / np.maximum(k_count, 1)),
                           ("tv_from_uniform", tv), ("max_rho", max_rho)):
            selected = array[mask]
            values["mean_" + key] = float(selected.mean()) if selected.size else float("nan")
            values["median_" + key] = float(np.median(selected)) if selected.size else float("nan")
        for threshold in (.5, .6, .7, .8, .9):
            values[f"fraction_max_rho_ge_{threshold:.1f}"] = (
                float((max_rho[mask] >= threshold).mean()) if mask.any() else float("nan"))
        result[name] = values
    return result


def compute_relation_statistics(entropy, max_alpha, tv, k_count):
    """Summarize item-side Relation weights on selected multi-KC targets."""
    arrays = [np.asarray(value).reshape(-1) for value in
              (entropy, max_alpha, tv, k_count)]
    if len({len(value) for value in arrays}) != 1:
        raise ValueError("Relation diagnostic arrays must be aligned.")
    entropy, max_alpha, tv, k_count = arrays
    result = {}
    for name, selector in (("k2", k_count == 2), ("k3plus", k_count >= 3),
                           ("multi", k_count >= 2)):
        values = {"n": int(selector.sum())}
        for key, array in (("relation_entropy", entropy),
                           ("max_alpha", max_alpha),
                           ("tv_from_uniform", tv)):
            selected = array[selector]
            values["mean_" + key] = (
                float(selected.mean()) if selected.size else float("nan"))
            values["median_" + key] = (
                float(np.median(selected)) if selected.size else float("nan"))
        result[name] = values
    return result


def compute_calibration_statistics(k_count, max_rho, mean_p, min_p, max_p, p_gap, mastery_gap):
    arrays = [np.asarray(v).reshape(-1) for v in
              (k_count, max_rho, mean_p, min_p, max_p, p_gap, mastery_gap)]
    if len({len(v) for v in arrays}) != 1:
        raise ValueError("Calibration diagnostic arrays must be aligned.")
    k_count, max_rho, mean_p, min_p, max_p, p_gap, mastery_gap = arrays
    probability = {}
    for name, mask in (("k2", k_count == 2), ("k3plus", k_count >= 3), ("multi", k_count >= 2)):
        values = {"n_incorrect": int(mask.sum())}
        for key, array in (("mean_kc_prob", mean_p), ("min_kc_prob", min_p),
                           ("max_kc_prob", max_p), ("kc_prob_gap", p_gap)):
            values["mean_" + key] = float(array[mask].mean()) if mask.any() else float("nan")
            values["median_" + key] = float(np.median(array[mask])) if mask.any() else float("nan")
        probability[name] = values
    strength = {}
    for lo, hi in ((.5, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.)):
        mask = (k_count == 2) & (max_rho >= lo) & ((max_rho <= hi) if hi == 1 else (max_rho < hi))
        label = f"[{lo:.2f}, {hi:.2f}" + ("]" if hi == 1 else ")")
        strength[label] = {"n": int(mask.sum()),
            "mean_kc_prob_gap": float(p_gap[mask].mean()) if mask.any() else float("nan"),
            "mean_pre_response_mastery_gap": float(mastery_gap[mask].mean()) if mask.any() else float("nan")}
    return probability, strength


def evaluate_raplkt(model, test_loader, save_path="", split_name=None,
                         return_group_metrics=False, debug_alignment=False,
                         debug_responsibility=False, responsibility_sample_size=12,
                         responsibility_sample_seed=3407, manifest_path=None,
                         set_loss_weight=0.1):
    set_loss_weight = validate_set_loss_weight(set_loss_weight)
    model.eval()
    first_parameter = next(model.parameters(), None)
    device = (first_parameter.device if first_parameter is not None else
              torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    truths, scores, counts, affected_flags = [], [], [], []
    diagnostics = {key: [] for key in ("entropy", "effective_k", "k", "tv", "max_rho")}
    relation_diagnostics = {key: [] for key in ("entropy", "max_alpha", "tv", "k")}
    raw_relation_diagnostics = {key: [] for key in ("entropy", "max_alpha", "tv", "k")}
    calibration = {key: [] for key in ("mean_p", "min_p", "max_p", "p_gap", "mastery_gap")}
    set_scores, set_nlls, correct_min_p, correct_products = [], [], [], []
    records, samples = [], []
    rng = random.Random(responsibility_sample_seed)
    seen_samples, sequence_offset = 0, 0
    alignment_printed = False
    with torch.no_grad():
        for data in test_loader:
            cq, cc, cm, cr, shifted_r, sm, updates = _build_full_sequences(data, device)
            y, _, details = model(cc, cr, cq, c_mask=cm,
                                  evidence_update_mask=updates, return_details=True)
            pred = y[:, 1:][sm].cpu().numpy()
            true = shifted_r[sm].cpu().numpy()
            k = cm[:, 1:].sum(-1)
            selected_k = k[sm].cpu().numpy()
            if np.any(selected_k < 1):
                raise AssertionError("Every selected target requires a nonempty KC set.")
            truths.append(true)
            scores.append(pred)
            counts.append(selected_k)
            affected_flags.append(details["affected_mask"][:, 1:][sm].cpu().numpy())
            wrong_multi = sm & (shifted_r == 0) & (k >= 2)
            rho = details["rho_wrong"][:, 1:]
            uniform = cm[:, 1:].to(rho.dtype) / k.unsqueeze(-1).clamp_min(1)
            tv = .5 * (rho - uniform).abs().sum(-1)
            maximum = rho.max(-1).values
            valid = cm[:, 1:].bool()
            relation_weight = details["question_weights"][:, 1:]
            raw_relation_weight = details["question_relation_raw_weights"][:, 1:]
            relation_uniform = valid.to(relation_weight.dtype) / k.unsqueeze(-1).clamp_min(1)
            relation_tv = .5 * (relation_weight - relation_uniform).abs().sum(-1)
            raw_relation_tv = .5 * (raw_relation_weight - relation_uniform).abs().sum(-1)
            selected_multi = sm & (k >= 2)
            for key, value in (
                    ("entropy", details["question_weight_entropy"][:, 1:]),
                    ("max_alpha", details["question_weight_max"][:, 1:]),
                    ("tv", relation_tv), ("k", k)):
                relation_diagnostics[key].append(value[selected_multi].cpu().numpy())
            for key, value in (
                    ("entropy", details["question_relation_raw_entropy"][:, 1:]),
                    ("max_alpha", details["question_relation_raw_max"][:, 1:]),
                    ("tv", raw_relation_tv), ("k", k)):
                raw_relation_diagnostics[key].append(value[selected_multi].cpu().numpy())
            probs = details["kc_probs"][:, 1:]
            mastery = details["mastery_mean"][:, 1:]
            min_p = probs.masked_fill(~valid, float("inf")).min(-1).values
            selected_probs, selected_mask = probs[sm], valid[sm]
            set_scores.append(_set_log_correct(selected_probs, selected_mask, model.eps).exp().cpu().numpy())
            set_nlls.append(compute_set_level_partial_loss(
                selected_probs, shifted_r[sm], selected_mask, model.eps).cpu().numpy())
            correct_multi = sm & (shifted_r == 1) & (k >= 2)
            correct_min_p.append(min_p[correct_multi].cpu().numpy())
            correct_products.append(_set_log_correct(
                probs[correct_multi], valid[correct_multi], model.eps).exp().cpu().numpy())
            max_p = probs.masked_fill(~valid, float("-inf")).max(-1).values
            mastery_gap = (mastery.masked_fill(~valid, float("-inf")).max(-1).values
                           - mastery.masked_fill(~valid, float("inf")).min(-1).values)
            for key, value in (("mean_p", (probs * valid).sum(-1) / k.clamp_min(1)),
                               ("min_p", min_p), ("max_p", max_p),
                               ("p_gap", max_p - min_p), ("mastery_gap", mastery_gap)):
                calibration[key].append(value[wrong_multi].cpu().numpy())
            for key, value in (("entropy", details["responsibility_entropy"][:, 1:]),
                               ("effective_k", details["effective_k"][:, 1:]),
                               ("k", k), ("tv", tv), ("max_rho", maximum)):
                diagnostics[key].append(value[wrong_multi].cpu().numpy())
            if manifest_path is not None:
                if "shft_cidxs" not in data:
                    raise AssertionError("Benchmark manifest requires shft_cidxs.")
                cidxs = data["shft_cidxs"].to(device)
                if torch.any(cidxs[sm] < 0):
                    raise AssertionError("Benchmark target_cidx must be nonnegative.")
                offset = int(getattr(test_loader.dataset, "id_offset", 1))
                for index, (b, s) in enumerate(sm.nonzero().tolist()):
                    kc_ids = cc[b, s + 1][cm[b, s + 1].bool()].cpu().tolist()
                    records.append({
                        "target_cidx": int(cidxs[b, s]),
                        "question_id": int(cq[b, s + 1]) - offset,
                        "response": int(true[index]),
                        "kc_ids": ",".join(str(v - offset) for v in sorted(kc_ids)),
                        "k_count": int(selected_k[index]), "prediction": float(pred[index])})
            if debug_alignment and not alignment_printed:
                print("RAPLKT alignment debug (first selected positions):")
                for index, (b, s) in enumerate(sm.nonzero().tolist()[:10]):
                    print(f"    batch={b} shifted_pos={s} target={int(true[index])} "
                          f"pred={pred[index]:.6f} k_count={selected_k[index]}")
                alignment_printed = True
            if debug_responsibility and responsibility_sample_size > 0:
                for b, s in wrong_multi.nonzero().tolist():
                    seen_samples += 1
                    # Reservoir sampling uses its own RNG, never the model RNG.
                    slot = (len(samples) if len(samples) < responsibility_sample_size
                            else rng.randrange(seen_samples))
                    if slot >= responsibility_sample_size:
                        continue
                    valid = cm[b, s + 1].bool()
                    ids = cc[b, s + 1][valid].cpu().tolist()
                    weights = rho[b, s][valid].cpu().tolist()
                    sample = {
                        "sequence_index": sequence_offset + b, "position": s + 1,
                        "question_id": int(cq[b, s + 1]), "response": 0,
                        "candidate_kc_ids": ids,
                        "pre_response_mastery": details["mastery_mean"][b, s + 1][valid].cpu().tolist(),
                        "kc_probs": details["kc_probs"][b, s + 1][valid].cpu().tolist(),
                        "question_weights": details["question_weights"][b, s + 1][valid].cpu().tolist(),
                        "alpha_relation": details["question_relation_raw_weights"][b, s + 1][valid].cpu().tolist(),
                        "rho": weights, "max_responsible_kc": ids[int(np.argmax(weights))],
                        "responsibility_entropy": float(details["responsibility_entropy"][b, s + 1]),
                        "effective_k": float(details["effective_k"][b, s + 1]),
                        "failure_update": details["failure_update"][b, s + 1][valid].cpu().tolist()}
                    if slot == len(samples):
                        samples.append(sample)
                    else:
                        samples[slot] = sample
            sequence_offset += cq.shape[0]
    def concatenate(parts):
        return np.concatenate(parts) if parts else np.array([])
    ts, ps, ks = map(concatenate, (truths, scores, counts))
    affected = concatenate(affected_flags).astype(bool)
    results = compute_item_level_metrics(ts, ps, ks)
    set_ps, set_nll = concatenate(set_scores), concatenate(set_nlls)
    results["set_level"] = compute_item_level_metrics(ts, set_ps, ks)
    for name, selector in KC_GROUPS.items():
        group_nll = set_nll[selector(ks)]
        results["set_level"][name]["bce"] = float(group_nll.mean()) if group_nll.size else float("nan")
    correct_stats = {"n": int(concatenate(correct_min_p).size)}
    for key, parts in (("min_kc_prob", correct_min_p), ("product_probability", correct_products)):
        values = concatenate(parts)
        correct_stats["mean_" + key] = float(values.mean()) if values.size else float("nan")
        correct_stats["median_" + key] = float(np.median(values)) if values.size else float("nan")
    results["correct_multi_kc_probability"] = correct_stats
    results["affected"] = compute_item_level_metrics(ts[affected], ps[affected], ks[affected])["overall"]
    results["affected_ratio"] = float(affected.mean()) if affected.size else 0.0
    results["responsibility"] = compute_responsibility_statistics(
        *(concatenate(diagnostics[key]) for key in ("entropy", "effective_k", "k", "tv", "max_rho")))
    results["question_relation"] = compute_relation_statistics(
        *(concatenate(relation_diagnostics[key])
          for key in ("entropy", "max_alpha", "tv", "k")))
    results["question_relation_raw"] = compute_relation_statistics(
        *(concatenate(raw_relation_diagnostics[key])
          for key in ("entropy", "max_alpha", "tv", "k")))
    results["relation_mix_lambda"] = model.relation_mix_lambda
    results["responsibility_configuration"] = {
        "rule": "normalize(1-p)", "set_level_partial_label_loss": True,
        "set_loss_weight": set_loss_weight}
    results["kc_probability"], results["responsibility_strength_k2"] = compute_calibration_statistics(
        concatenate(diagnostics["k"]), concatenate(diagnostics["max_rho"]),
        *(concatenate(calibration[key]) for key in ("mean_p", "min_p", "max_p", "p_gap", "mastery_gap")))
    if debug_responsibility:
        results["responsibility_samples"] = samples
    if save_path:
        with open(save_path, "w", encoding="utf8") as stream:
            for target, prediction in zip(ts, ps):
                stream.write(f"{int(target)}\t{float(prediction)}\n")
    if manifest_path is not None:
        write_benchmark_manifest(manifest_path, records)
    if split_name is not None:
        print("Responsibility configuration:")
        print("    rule=normalize(1-p)")
        print("    set_level_partial_label_loss=True")
        print(f"    set_loss_weight={set_loss_weight}")
        print_set_level_metrics(results, split_name)
        print_item_level_metrics(results, split_name)
        group = results["affected"]
        print(f"Affected ({split_name}): N={group['n']} AUC={group['auc']:.6f} "
              f"ACC={group['acc']:.6f} affected_ratio={results['affected_ratio']:.6f}")
        for name, values in results["responsibility"].items():
            print(f"Incorrect {name} Responsibility ({split_name}):")
            for key, value in values.items():
                print(f"    {key}={value}")
        for name, values in results["question_relation"].items():
            print(f"Final Question Weight {name} ({split_name}):")
            for key, value in values.items():
                print(f"    {key}={value}")
        for name, values in results["question_relation_raw"].items():
            print(f"Raw Question Relation {name} ({split_name}):")
            for key, value in values.items():
                print(f"    {key}={value}")
        print(f"Relation mix lambda ({split_name}): {results['relation_mix_lambda']}")
        for name, values in results["kc_probability"].items():
            print(f"Incorrect {name} KC probability ({split_name}): {values}")
        for name, values in results["responsibility_strength_k2"].items():
            print(f"Incorrect K=2 strength {name} ({split_name}): {values}")
        if debug_responsibility:
            for sample in samples:
                print(f"Responsibility sample: {sample}")
    if return_group_metrics:
        return results
    return results["overall"]["auc"], results["overall"]["acc"]


def print_set_level_metrics(results, split_name):
    for name, values in results["set_level"].items():
        print(f"Set-level KC evidence ({split_name}, {name}): "
              f"N={values['n']} BCE={values['bce']:.6f} "
              f"AUC={values['auc']:.6f} ACC={values['acc']:.6f}")
    print(f"Correct multi-KC evidence ({split_name}): {results['correct_multi_kc_probability']}")


def train_raplkt_model(
    model,
    train_loader,
    valid_loader,
    num_epochs,
    opt,
    ckpt_path,
    save_model=False,
    early_stop_rounds=10,
    early_stop_min_delta=1e-3,
    set_loss_weight=0.1,
):
    set_loss_weight = validate_set_loss_weight(set_loss_weight)
    best_auc, best_acc, best_epoch = float("-inf"), 0.0, -1
    patience_reference_auc = float("-inf")
    patience_best_epoch = 0
    checkpoint_path = os.path.join(
        ckpt_path,
        model.emb_type + "_model.ckpt",
    )

    epoch_durations = []
    epoch_history = []
    for epoch in range(1, num_epochs + 1):
        epoch_started_at = time.perf_counter()
        model.train()
        loss_mean = []
        component_sums = {key: 0. for key in ("kt", "set", "reg", "total")}
        selected_items = 0

        for data in train_loader:
            loss, components = raplkt_model_forward(
                model, data, set_loss_weight, return_losses=True)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

            loss_mean.append(loss.detach().cpu().numpy())
            n = components["n"]
            selected_items += n
            for key in ("kt", "set", "reg"):
                component_sums[key] += float(components[key].detach()) * n
            component_sums["total"] += float(loss.detach()) * n

        loss_mean = float(np.mean(loss_mean))

        component_means = {key: value / max(selected_items, 1) for key, value in component_sums.items()}
        validation = evaluate_raplkt(model, valid_loader, return_group_metrics=True,
                                              set_loss_weight=set_loss_weight)
        validauc, validacc = validation["overall"]["auc"], validation["overall"]["acc"]
        print_set_level_metrics(validation, "valid")
        epoch_seconds = time.perf_counter() - epoch_started_at
        epoch_durations.append(epoch_seconds)
        epoch_history.append({"epoch": epoch, "selected_items": selected_items,
                              **{"train_" + key + "_loss": value for key, value in component_means.items()},
                              "weighted_set_loss": set_loss_weight * component_means["set"],
                              "validation_set_level": validation["set_level"]})

        # Checkpoint selection always follows the true maximum validation AUC.
        # Even a small strict improvement must replace the saved checkpoint.
        if validauc > best_auc:
            best_auc = validauc
            best_acc = validacc
            best_epoch = epoch
            if save_model:
                torch.save(model.state_dict(), checkpoint_path)

        # Patience deliberately uses its own min-delta reference so validation
        # noise does not indefinitely postpone early stopping.
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
            f"train_total_loss: {component_means['total']:.8f}, "
            f"train_kt_loss: {component_means['kt']:.8f}, "
            f"train_set_loss: {component_means['set']:.8f}, "
            f"train_reg_loss: {component_means['reg']:.8f}, "
            f"weighted_set_loss: {set_loss_weight * component_means['set']:.8f}, "
            f"epoch_seconds: {epoch_seconds:.1f}, "
            f"emb_type: {model.emb_type}, "
            f"model: {model.model_name}, "
            f"save_dir: {ckpt_path}"
        )

        if epoch - patience_best_epoch >= early_stop_rounds:
            break

    print("========== True best validation checkpoint ==========")
    print(f"true_best_valid_auc = {best_auc:.10f}")
    print(f"true_best_valid_acc = {best_acc:.10f}")
    print(f"true_best_epoch = {best_epoch}")
    print(f"checkpoint_path = {checkpoint_path if save_model else 'NOT_SAVED'}")
    print("=====================================================")

    with open(os.path.join(ckpt_path, "training_summary.json"), "w", encoding="utf8") as stream:
        json.dump(_json_safe({"valid_auc": best_auc, "valid_acc": best_acc, "best_epoch": best_epoch,
                   "set_loss_weight": set_loss_weight, "epoch_history": epoch_history,
                   "epoch_seconds": epoch_durations,
                   "mean_epoch_seconds": float(np.mean(epoch_durations)),
                   "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad)}),
                  stream, indent=2, allow_nan=False)
    return -1, -1, -1, -1, best_auc, best_acc, best_epoch
