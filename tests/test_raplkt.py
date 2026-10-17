import ast
import importlib.util
import inspect
import math
from pathlib import Path
from unittest.mock import patch

import pytest
import pandas as pd
import torch
import torch.nn.functional as F

from pykt.models.raplkt import RAPLKT, RAPLKT_VERSION
from pykt.models.raplkt_train_model import (
    compute_set_level_partial_loss, raplkt_model_forward, validate_set_loss_weight)

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SMALL = dict(n_question=8, n_pid=8, d_model=8, n_blocks=1, dropout=0.0,
             d_ff=16, num_attn_heads=2, final_fc_dim=16)


def make_model(**overrides):
    torch.manual_seed(3407)
    return RAPLKT(**dict(SMALL, **overrides)).to(DEVICE).eval()


def load_script(name):
    path = ROOT / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_dataset(directory):
    directory.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({
        "fold": [-1, -1], "questions": ["0,1,1,2,3", "1,2,3"],
        "concepts": ["0,1,2,1,3", "2,1,3"], "responses": ["1,0,0,1,0", "0,1,0"],
        "is_repeat": ["0,0,1,0,0", "1,0,0"], "selectmasks": ["1,1,1,1,1", "1,1,1"],
        "cidxs": ["0,1,2,3,4", "5,6,7"]})
    for name in ("test.csv", "test_sequences.csv"):
        frame.iloc[:1].to_csv(directory / name, index=False)
    frame.to_csv(directory / "test_window_sequences.csv", index=False)
    pd.concat((frame.iloc[:1].assign(fold=0), frame.iloc[:1].assign(fold=1))).to_csv(
        directory / "train_valid_sequences.csv", index=False)
    return dict(dpath=str(directory), input_type=["questions", "concepts"],
                test_original_file="test.csv", test_file="test_sequences.csv",
                test_window_file="test_window_sequences.csv", folds=[0, 1],
                train_valid_file="train_valid_sequences.csv", num_c=8, num_q=8)


@pytest.mark.parametrize("probabilities,response,expected", [
    ([0.8], 1, -math.log(0.8)), ([0.8], 0, -math.log(0.2)),
    ([0.8, 0.9], 1, -math.log(0.72)), ([0.8, 0.9], 0, -math.log(0.28)),
    ([0.1, 0.9], 0, -math.log(0.91)),
])
def test_set_level_partial_label_likelihood(probabilities, response, expected):
    probs = torch.tensor([[probabilities]], dtype=torch.float64)
    mask = torch.ones_like(probs, dtype=torch.bool)
    loss = compute_set_level_partial_loss(probs, torch.tensor([[response]]), mask)
    assert loss.item() == pytest.approx(expected, abs=1e-12)
    if len(probabilities) == 1:
        direct = F.binary_cross_entropy(
            probs[..., 0], torch.tensor([[response]], dtype=torch.float64), reduction="none")
        torch.testing.assert_close(loss, direct)


def test_responsibility_and_evidence_budgets():
    model = make_model()
    probs = torch.tensor([[[0.9, 0.2, 0.0]]], device=DEVICE)
    mask = torch.tensor([[[True, True, False]]], device=DEVICE)
    torch.testing.assert_close(
        model._wrong_responsibility(probs, mask),
        torch.tensor([[[1 / 9, 8 / 9, 0.0]]], device=DEVICE), atol=1e-6, rtol=0)
    concepts = torch.tensor([[[1, 2, 0], [1, 2, 3], [2, 3, 0]]], device=DEVICE)
    responses = torch.tensor([[0, 1, 0]], device=DEVICE)
    valid = concepts != 0
    details = model(concepts, responses, torch.tensor([[1, 2, 3]], device=DEVICE),
                    c_mask=valid, return_details=True)[2]
    torch.testing.assert_close(details["failure_update"].sum(-1)[responses == 0],
                               torch.ones(2, device=DEVICE), atol=1e-6, rtol=0)
    torch.testing.assert_close(details["success_update"].sum(-1)[responses == 1],
                               torch.ones(1, device=DEVICE), atol=1e-6, rtol=0)
    expected_success = valid.float() / valid.sum(-1, keepdim=True).clamp_min(1)
    torch.testing.assert_close(details["success_update"][responses == 1],
                               expected_success[responses == 1])


@pytest.mark.parametrize("temperature,expected", [
    (0.5, [1 / 65, 64 / 65, 0.0]),
    (1.0, [1 / 9, 8 / 9, 0.0]),
    (2.0, [math.sqrt(0.1) / (math.sqrt(0.1) + math.sqrt(0.8)),
           math.sqrt(0.8) / (math.sqrt(0.1) + math.sqrt(0.8)), 0.0]),
])
def test_responsibility_temperature(temperature, expected):
    probs = torch.tensor([[[0.9, 0.2, 0.0]]], device=DEVICE)
    mask = torch.tensor([[[True, True, False]]], device=DEVICE)
    responsibility = make_model(
        responsibility_temperature=temperature)._wrong_responsibility(probs, mask)
    torch.testing.assert_close(
        responsibility, torch.tensor([[expected]], device=DEVICE), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        responsibility.sum(-1), torch.ones((1, 1), device=DEVICE), atol=1e-6, rtol=0)


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf")])
def test_responsibility_temperature_validation(temperature):
    with pytest.raises(ValueError, match="responsibility_temperature"):
        make_model(responsibility_temperature=temperature)


@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0])
def test_relation_mix_limits_k1_padding_and_weighted_question(lam):
    model = make_model(relation_dim=2, relation_mix_lambda=lam)
    with torch.no_grad():
        model.item_relation_embed.weight.zero_()
        model.kc_relation_embed.weight.zero_()
        model.item_relation_embed.weight[1] = torch.tensor([1.0, 0.0], device=DEVICE)
        model.kc_relation_embed.weight[1] = torch.tensor([1.0, 0.0], device=DEVICE)
        model.kc_relation_embed.weight[2] = torch.tensor([0.0, 1.0], device=DEVICE)
        model.q_embed.weight[1] = torch.tensor([1.0] * 8, device=DEVICE)
        model.q_embed.weight[2] = torch.tensor([3.0] * 8, device=DEVICE)
    concepts = torch.tensor([[[1, 0, 0, 0], [1, 2, 0, 0]]], device=DEVICE)
    mask = concepts != 0
    questions = torch.tensor([[1, 1]], device=DEVICE)
    question, candidates, alpha, alpha_relation, alpha_mean = model._relation_question_embedding(
        concepts, mask, questions)
    torch.testing.assert_close(alpha[0, 0], torch.tensor([1., 0., 0., 0.], device=DEVICE))
    expected_relation = torch.softmax(torch.tensor([1., 0.], device=DEVICE), dim=0)
    torch.testing.assert_close(alpha_relation[0, 1, :2], expected_relation)
    expected_mean = torch.tensor([0.5, 0.5, 0., 0.], device=DEVICE)
    torch.testing.assert_close(alpha_mean[0, 1], expected_mean)
    expected_alpha = (1.0 - lam) * expected_mean[:2] + lam * expected_relation
    torch.testing.assert_close(alpha[0, 1, :2], expected_alpha)
    assert not alpha[0, 1, 2:].any()
    assert not alpha_relation[0, 1, 2:].any()
    assert not alpha_mean[0, 1, 2:].any()
    torch.testing.assert_close(alpha[0, 1].sum(), torch.tensor(1., device=DEVICE))
    expected_question = expected_alpha[0] * candidates[0, 1, 0] + expected_alpha[1] * candidates[0, 1, 1]
    torch.testing.assert_close(question[0, 1], expected_question)
    if lam == 0.0:
        assert torch.equal(alpha, alpha_mean)
        torch.testing.assert_close(question[0, 1], candidates[0, 1, :2].mean(0))
    if lam == 0.5:
        torch.testing.assert_close(alpha, 0.5 * alpha_mean + 0.5 * alpha_relation)
    if lam == 1.0:
        assert torch.equal(alpha, alpha_relation)


def test_relation_mix_is_fixed_validated_hyperparameter():
    model = make_model(relation_mix_lambda=0.25)
    assert model.relation_mix_lambda == 0.25
    assert "relation_mix_lambda" not in dict(model.named_parameters())
    assert not hasattr(model, "relation_gate_logit")
    for value in (-0.01, 1.01, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="relation_mix_lambda"):
            make_model(relation_mix_lambda=value)


def test_relation_is_response_and_history_independent():
    model = make_model()
    concepts = torch.tensor([[[1, 2], [3, 4], [1, 2]],
                             [[5, 6], [7, 8], [1, 2]]], device=DEVICE)
    questions = torch.tensor([[1, 2, 1], [3, 4, 1]], device=DEVICE)
    first = model(concepts, torch.tensor([[0, 0, 0], [1, 1, 1]], device=DEVICE),
                  questions, return_details=True)[2]
    second = model(concepts, torch.tensor([[1, 1, 1], [0, 0, 0]], device=DEVICE),
                   questions, return_details=True)[2]
    for key in ("question_weights", "question_relation_raw_weights", "question_mean_weights"):
        torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)
        torch.testing.assert_close(first[key][0, 2], first[key][1, 2], atol=0, rtol=0)
    assert first["relation_mix_lambda"] == model.relation_mix_lambda


def test_alpha_current_side_and_rho_history_side_are_separate():
    model = make_model(relation_dim=2)
    concepts = torch.tensor([[[1, 2]]], device=DEVICE)
    responses = torch.tensor([[0]], device=DEVICE)
    questions = torch.tensor([[1]], device=DEVICE)
    alpha = torch.tensor([[[0.9, 0.1]]], device=DEVICE)
    rho = torch.tensor([[[0.2, 0.8]]], device=DEVICE)
    with torch.no_grad():
        model.q_embed.weight.zero_()
        model.q_embed.weight[1, 0] = 1.
        model.q_embed.weight[2, 0] = 3.
        model.q_embed_diff.weight.zero_()
        model.q_embed_diff.weight[1, 1] = 2.
        model.q_embed_diff.weight[2, 1] = 4.
        model.difficult_param.weight.fill_(1.)
        model.qa_embed.weight.zero_()
        model.qa_embed_diff.weight.zero_()
        for parameter in model.beta_state_encoder.parameters():
            parameter.zero_()
    candidates = model.q_embed(concepts)
    current = model._masked_weighted_sum(candidates, alpha)

    def relation(*_):
        return current, candidates, alpha, alpha, alpha

    def evidence(*_):
        return torch.tensor([[[0.8, 0.2]]], device=DEVICE)

    def zero_akt(query, *_):
        return torch.zeros_like(query)

    with patch.object(model, "_relation_question_embedding", side_effect=relation), \
            patch.object(model, "_compute_kc_evidence", side_effect=evidence), \
            patch.object(model.model, "forward", side_effect=zero_akt):
        details = model(concepts, responses, questions, return_details=True)[2]
    torch.testing.assert_close(details["rho_wrong"], rho)
    current_diff = model._masked_weighted_sum(model.q_embed_diff(concepts), alpha)
    torch.testing.assert_close(details["q_embed"], current + current_diff)
    historical = model._masked_weighted_sum(
        candidates + model.q_embed_diff(concepts), rho)
    torch.testing.assert_close(details["qa_embed"], historical)


def test_relation_mix_does_not_change_responsibility_or_beta_routing():
    mean_model = make_model(relation_mix_lambda=0.0)
    relation_model = make_model(relation_mix_lambda=1.0)
    concepts = torch.tensor([[[1, 2], [1, 2], [2, 3]]], device=DEVICE)
    mask = concepts != 0
    responses = torch.tensor([[0, 1, 0]], device=DEVICE)
    updates = torch.ones_like(responses, dtype=torch.bool)
    probs = torch.tensor([[[0.8, 0.2], [0.7, 0.4], [0.9, 0.3]]], device=DEVICE)
    mean_mix = mean_model._build_responsibility_trajectory(
        concepts, mask, responses, updates, probs)
    relation_mix = relation_model._build_responsibility_trajectory(
        concepts, mask, responses, updates, probs)
    for key in ("route_weight", "rho_wrong", "success_update", "failure_update",
                "success_evidence", "failure_evidence", "beta_state_emb"):
        torch.testing.assert_close(mean_mix[key], relation_mix[key], atol=0, rtol=0)


def test_parallel_exclusive_prefix_and_no_sequence_loop():
    delta = torch.randn(2, 7, 5, 2, requires_grad=True)
    expected = torch.zeros_like(delta)
    for step in range(1, delta.shape[1]):
        expected[:, step] = delta[:, :step].sum(1)
    actual = RAPLKT._exclusive_prefix(delta)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert torch.isfinite(delta.grad).all()
    tree = ast.parse(inspect.getsource(RAPLKT))
    checked = {"forward", "_build_responsibility_trajectory", "_exclusive_prefix"}
    for function in (node for node in ast.walk(tree)
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and node.name in checked):
        assert not any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(function))


@pytest.mark.parametrize("separate_qa", [False, True])
def test_two_pass_causality_and_end_to_end_gradient(separate_qa):
    model = make_model(separate_qa=separate_qa)
    concepts = torch.tensor([[[1, 2, 0], [2, 3, 0], [1, 3, 4], [4, 5, 0]]], device=DEVICE)
    responses = torch.tensor([[0, 1, 0, 1]], device=DEVICE)
    questions = torch.tensor([[1, 2, 3, 4]], device=DEVICE)
    with patch.object(model.model, "forward", wraps=model.model.forward) as akt:
        prediction, _, details = model(concepts, responses, questions, return_details=True)
        assert akt.call_count == 2
    flipped = responses.clone()
    flipped[:, 2:] = 1 - flipped[:, 2:]
    changed, _, changed_details = model(concepts, flipped, questions, return_details=True)
    torch.testing.assert_close(prediction[:, :2], changed[:, :2], atol=0, rtol=0)
    torch.testing.assert_close(details["kc_probs"][:, :2], changed_details["kc_probs"][:, :2], atol=0, rtol=0)
    loss = F.binary_cross_entropy(prediction[:, 1:], responses[:, 1:].float())
    grads = torch.autograd.grad(
        loss, tuple(model.kc_evidence.parameters()), retain_graph=True, allow_unused=False)
    assert sum(gradient.abs().sum() for gradient in grads) > 0
    relation_grads = torch.autograd.grad(
        loss, (model.item_relation_embed.weight, model.kc_relation_embed.weight),
        retain_graph=True, allow_unused=False)
    assert all(torch.isfinite(gradient).all() and gradient.abs().sum() > 0
               for gradient in relation_grads)
    assert ".detach(" not in inspect.getsource(RAPLKT)


def test_mean_checkpoint_is_rejected_by_strict_relation_load():
    reference = torch.load(ROOT / "tests/fixtures/raplkt_formal_reference.pt",
                           map_location="cpu", weights_only=True)
    model = RAPLKT(**reference["config"]).to(DEVICE).eval()
    with pytest.raises(RuntimeError, match="item_relation_embed"):
        model.load_state_dict(reference["state_dict"], strict=True)


def test_public_identity_cli_and_parameter_count():
    assert make_model().model_name == "raplkt"
    assert RAPLKT_VERSION == "responsibility_partial_label_fixed_relation_mix_unit_evidence_v4"
    train = load_script("wandb_raplkt_train")
    predict = load_script("wandb_raplkt_predict")
    args = train.build_parser().parse_args([])
    assert args.save_dir == "saved_model_raplkt" and args.set_loss_weight == 0.1
    assert args.relation_dim == 32 and args.relation_temperature == 1.0
    assert args.relation_mix_lambda == 1.0
    assert args.responsibility_temperature == 1.0
    assert "full_config" not in vars(args)
    assert predict.RAPLKT is RAPLKT
    with pytest.raises(ValueError, match="Relation parameters"):
        predict.main(dict(save_dir="unused", relation_temperature=0.5))
    with pytest.raises(ValueError, match="responsibility temperature"):
        predict.main(dict(save_dir="unused", responsibility_temperature=0.5))
    production = RAPLKT(n_question=101, n_pid=3162, d_model=256, n_blocks=2,
                        dropout=0.2, d_ff=256, num_attn_heads=4)
    base_count = 2694517
    relation_count = (3162 + 1 + 101 + 1) * 32
    assert relation_count == 104480
    assert sum(parameter.numel() for parameter in production.parameters()) == base_count + relation_count


def test_no_relation_responsibility_coupling_or_runtime_switch():
    source = inspect.getsource(RAPLKT)
    forbidden_phrases = (
        "alpha" + " * rho", "rho" + " * alpha",
        "failure_increment" + " * alpha", "success_increment" + " * alpha",
        "route" + " = alpha", "question" + "_aggregation")
    for forbidden in forbidden_phrases:
        assert forbidden not in source
    probs = torch.tensor([[[0.9, 0.2]]], device=DEVICE)
    mask = torch.ones_like(probs, dtype=torch.bool)
    expected = torch.tensor([[[1 / 9, 8 / 9]]], device=DEVICE)
    torch.testing.assert_close(make_model()._wrong_responsibility(probs, mask), expected)


def test_training_objective_and_validation():
    model = make_model()
    data = {
        "qseqs": torch.tensor([[1, 2, 3]]), "shft_qseqs": torch.tensor([[2, 3, 4]]),
        "cseqs_multi": torch.tensor([[[1, 0], [2, 3], [3, 0]]]),
        "shft_cseqs_multi": torch.tensor([[[2, 3], [3, 0], [4, 5]]]),
        "cmasks_multi": torch.tensor([[[1, 0], [1, 1], [1, 0]]]),
        "shft_cmasks_multi": torch.tensor([[[1, 1], [1, 0], [1, 1]]]),
        "evidence_update_seqs": torch.tensor([[1, 1, 1]]),
        "shft_evidence_update_seqs": torch.tensor([[1, 1, 1]]),
        "rseqs": torch.tensor([[1, 0, 1]]), "shft_rseqs": torch.tensor([[0, 1, 0]]),
        "masks": torch.tensor([[1, 1, 1]], dtype=torch.bool),
        "smasks": torch.tensor([[1, 1, 1]], dtype=torch.bool),
    }
    total, parts = raplkt_model_forward(model, data, set_loss_weight=0.1, return_losses=True)
    torch.testing.assert_close(total, parts["kt"] + 0.1 * parts["set"] + parts["reg"])
    total.backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    for value in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="set_loss_weight"):
            validate_set_loss_weight(value)


def test_smoke_train_and_predict(tmp_path, monkeypatch):
    train = load_script("wandb_raplkt_train")
    predict = load_script("wandb_raplkt_predict")
    config = write_dataset(tmp_path / "data")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/data_config.json").write_text(
        __import__("json").dumps({"fixture": config}), encoding="utf8")
    (tmp_path / "configs/kt_config.json").write_text(
        __import__("json").dumps({"train_config": {"optimizer": "adam", "num_epochs": 2}}),
        encoding="utf8")
    monkeypatch.setattr(train, "ROOT", tmp_path)
    monkeypatch.setattr(predict, "ROOT", tmp_path)
    arguments = vars(train.build_parser().parse_args([
        "--dataset_name", "fixture", "--save_dir", str(tmp_path / "saved"),
        "--d_model", "8", "--d_ff", "16", "--n_blocks", "1",
        "--num_attn_heads", "2", "--batch_size", "2", "--max_concepts", "3",
        "--num_epochs", "2", "--early_stop_rounds", "2"]))
    train.main(arguments)
    run_dir = next((tmp_path / "saved").iterdir())
    assert (run_dir / "qid_model.ckpt").exists()
    predict.main(dict(save_dir=str(run_dir), bz=2, debug_alignment=1,
                      debug_responsibility=1, responsibility_sample_size=2))
    report = __import__("json").loads(
        (run_dir / "qid_evaluation_metrics.json").read_text(encoding="utf8"))
    assert report["model_name"] == "raplkt"
    assert report["question_aggregation"] == "fixed_mean_relation_mix"
    assert report["relation_dim"] == 32 and report["relation_temperature"] == 1.0
    assert report["relation_mix_lambda"] == 1.0
    assert report["responsibility_temperature"] == 1.0
    assert report["failure_budget"] == 1.0
    assert report["success_budget"] == 1.0
    assert "set_level" in report["test"]
    assert "question_relation" in report["test"]
    assert "question_relation_raw" in report["test"]
    assert "affected" in report["window_test"]
