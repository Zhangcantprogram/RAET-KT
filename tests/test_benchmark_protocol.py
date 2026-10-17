import os
import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
from torch import nn
from torch.utils.data import RandomSampler, SequentialSampler

from pykt.datasets.data_loader import KTDataset
from pykt.datasets.init_dataset import init_dataset4train, init_test_datasets
from pykt.datasets.init_raplkt_dataset import (
    init_raplkt_dataset4train,
    init_raplkt_test_datasets,
)
from pykt.datasets.multikc_protocol import (
    BENCHMARK_PROTOCOL_VERSION,
    build_raw_q2cs_map,
)
from pykt.datasets.raplkt_data_loader import RAPLKTDataset
from pykt.models.evaluate_model import evaluate
from pykt.models.item_level_metrics import assert_same_benchmark_targets
from pykt.models.raplkt_train_model import evaluate_raplkt
from pykt.models.train_model import train_model

train_model_module = importlib.import_module("pykt.models.train_model")


class _FixedAKT(nn.Module):
    model_name = "akt"

    def forward(self, q_data, target, pid_data=None, **kwargs):
        prediction = torch.full(target.shape, 0.8, device=target.device)
        return prediction, prediction.new_tensor(0.0)


class _FixedRAPLKT(nn.Module):
    model_name = "raplkt"
    eps = 1e-8

    def __init__(self):
        super().__init__()
        self.relation_mix_lambda = 1.0

    def forward(self, q_data, target, pid_data=None, **kwargs):
        prediction = torch.full(target.shape, 0.8, device=target.device)
        zero = prediction.new_tensor(0.0)
        valid = kwargs["c_mask"].bool()
        count = valid.sum(-1).clamp_min(1)
        rho = valid.to(prediction.dtype) / count.unsqueeze(-1)
        entropy = -(rho * rho.clamp_min(self.eps).log()).sum(-1)
        details = {
            "affected_mask": torch.zeros_like(target, dtype=torch.bool),
            "rho_wrong": rho,
            "kc_probs": valid.to(prediction.dtype) * 0.8,
            "mastery_mean": valid.to(prediction.dtype) * 0.5,
            "responsibility_entropy": entropy,
            "effective_k": entropy.exp(),
            "failure_update": torch.zeros_like(rho),
            "question_weights": rho,
            "question_relation_raw_weights": rho,
            "question_mean_weights": rho,
            "question_weight_entropy": entropy,
            "question_weight_max": rho.max(-1).values,
            "question_relation_raw_entropy": entropy,
            "question_relation_raw_max": rho.max(-1).values,
        }
        return prediction, zero, details


def _write_protocol_files(directory):
    directory = Path(directory)
    pd.DataFrame({
        "fold": [-1],
        "questions": ["0,1,1,2"],
        "concepts": ["0,1,2,3"],
        "responses": ["0,1,1,0"],
        "is_repeat": ["0,0,1,0"],
        "cidxs": ["0,1,2,3"],
    }).to_csv(directory / "test.csv", index=False)
    # q=1 is deliberately absent, modelling a min_seq_len-dropped short tail.
    pd.DataFrame({
        "fold": [-1],
        "questions": ["0,2"],
        "concepts": ["0,3"],
        "responses": ["0,0"],
        "selectmasks": ["1,1"],
        "is_repeat": ["0,0"],
        "cidxs": ["0,3"],
    }).to_csv(directory / "test_sequences.csv", index=False)
    # The Window target contains only rep=0; q2cs metadata must restore KC 1,2.
    pd.DataFrame({
        "fold": [-1],
        "questions": ["0,1"],
        "concepts": ["0,1"],
        "responses": ["0,1"],
        "selectmasks": ["-1,1"],
        "is_repeat": ["0,0"],
        "cidxs": ["0,1"],
    }).to_csv(directory / "test_window_sequences.csv", index=False)
    return {
        "dpath": str(directory),
        "input_type": ["questions", "concepts"],
        "test_original_file": "test.csv",
        "test_file": "test_sequences.csv",
        "test_window_file": "test_window_sequences.csv",
    }


class BenchmarkWindowProtocolTest(unittest.TestCase):
    def test_original_test_metadata_restores_short_tail_window_target(self):
        self.assertEqual(BENCHMARK_PROTOCOL_VERSION, "item_multikc_window_v1")
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _write_protocol_files(temp_dir)
            sequence_map = build_raw_q2cs_map(
                os.path.join(temp_dir, "test_sequences.csv"), max_concepts=3
            )
            self.assertNotIn(1, sequence_map)

            akt_test, akt_window, _, _ = init_test_datasets(
                config, "akt", 2, sequence_mode="collapse", max_concepts=3
            )
            rapl_test, rapl_window, _, _ = init_raplkt_test_datasets(
                config, 2, sequence_mode="collapse", max_concepts=3
            )
            self.assertEqual(akt_window.dataset.q2cs_map[1], [1, 2])
            self.assertEqual(rapl_window.dataset.raw_q2cs_map[1], [1, 2])
            for loader in [akt_test, akt_window, rapl_test, rapl_window]:
                self.assertIsInstance(loader.sampler, SequentialSampler)

            akt_target_mask = akt_window.dataset.dori["smasks"]
            rapl_target_mask = rapl_window.dataset.dori["smasks"]
            akt_candidates = akt_window.dataset.dori["cseqs_multi"][:, 1, :]
            rapl_candidates = rapl_window.dataset.dori["cseqs_multi"][:, 1, :]
            self.assertTrue(akt_target_mask[0, 0])
            self.assertTrue(rapl_target_mask[0, 0])
            self.assertEqual(akt_candidates[0, :2].tolist(), [1, 2])
            self.assertEqual(rapl_candidates[0, :2].tolist(), [2, 3])

    def test_akt_and_raplkt_manifests_and_metric_counts_match(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _write_protocol_files(temp_dir)
            _, akt_window, _, _ = init_test_datasets(
                config, "akt", 2, sequence_mode="collapse", max_concepts=3
            )
            _, rapl_window, _, _ = init_raplkt_test_datasets(
                config, 2, sequence_mode="collapse", max_concepts=3
            )
            akt_manifest = os.path.join(temp_dir, "akt.tsv")
            rapl_manifest = os.path.join(temp_dir, "rapl.tsv")
            akt_metrics = evaluate(
                _FixedAKT(), akt_window, "akt", return_group_metrics=True,
                manifest_path=akt_manifest,
            )
            rapl_metrics = evaluate_raplkt(
                _FixedRAPLKT(), rapl_window, return_group_metrics=True,
                manifest_path=rapl_manifest,
            )
            self.assertTrue(assert_same_benchmark_targets(akt_manifest, rapl_manifest))
            for group in ["overall", "k1", "k2", "k3plus", "multi"]:
                for field in ["n", "pos", "neg"]:
                    self.assertEqual(
                        akt_metrics[group][field], rapl_metrics[group][field]
                    )
            manifest = pd.read_csv(akt_manifest, sep="\t")
            self.assertEqual(manifest.loc[0, "target_cidx"], 1)
            self.assertEqual(manifest.loc[0, "question_id"], 1)
            self.assertEqual(manifest.loc[0, "kc_ids"], "1,2")
            self.assertEqual(manifest.loc[0, "k_count"], 2)

    def test_complete_mapping_deduplicates_visible_repeat_kcs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _write_protocol_files(temp_dir)
            pd.DataFrame({
                "fold": [-1],
                "questions": ["0,1,1"],
                "concepts": ["0,1,2"],
                "responses": ["0,1,1"],
                "selectmasks": ["-1,1,1"],
                "is_repeat": ["0,0,1"],
                "cidxs": ["0,1,2"],
            }).to_csv(
                Path(temp_dir) / "test_window_sequences.csv",
                index=False,
            )

            _, akt_window, _, _ = init_test_datasets(
                config, "akt", 2, sequence_mode="collapse", max_concepts=3
            )
            _, rapl_window, _, _ = init_raplkt_test_datasets(
                config, 2, sequence_mode="collapse", max_concepts=3
            )
            akt_manifest = os.path.join(temp_dir, "akt_repeat.tsv")
            rapl_manifest = os.path.join(temp_dir, "rapl_repeat.tsv")
            akt_metrics = evaluate(
                _FixedAKT(), akt_window, "akt", return_group_metrics=True,
                manifest_path=akt_manifest,
            )
            rapl_metrics = evaluate_raplkt(
                _FixedRAPLKT(), rapl_window, return_group_metrics=True,
                manifest_path=rapl_manifest,
            )

            self.assertTrue(assert_same_benchmark_targets(akt_manifest, rapl_manifest))
            self.assertEqual(akt_metrics["k2"]["n"], 1)
            self.assertEqual(rapl_metrics["k2"]["n"], 1)
            self.assertEqual(akt_metrics["k3plus"]["n"], 0)
            self.assertEqual(rapl_metrics["k3plus"]["n"], 0)
            manifest = pd.read_csv(rapl_manifest, sep="\t")
            self.assertEqual(manifest.loc[0, "kc_ids"], "1,2")
            self.assertEqual(manifest.loc[0, "k_count"], 2)

    def test_manifest_requires_shifted_cidxs(self):
        class _OneBatchLoader(list):
            pass

        with tempfile.TemporaryDirectory() as temp_dir:
            config = _write_protocol_files(temp_dir)
            _, akt_window, _, _ = init_test_datasets(
                config, "akt", 2, sequence_mode="collapse", max_concepts=3
            )
            _, rapl_window, _, _ = init_raplkt_test_datasets(
                config, 2, sequence_mode="collapse", max_concepts=3
            )

            akt_batch = next(iter(akt_window))
            akt_batch.pop("shft_cidxs")
            akt_loader = _OneBatchLoader([akt_batch])
            akt_loader.dataset = akt_window.dataset
            with self.assertRaisesRegex(AssertionError, "requires shft_cidxs"):
                evaluate(
                    _FixedAKT(), akt_loader, "akt",
                    manifest_path=os.path.join(temp_dir, "invalid_akt.tsv"),
                )

            rapl_batch = next(iter(rapl_window))
            rapl_batch.pop("shft_cidxs")
            rapl_loader = _OneBatchLoader([rapl_batch])
            rapl_loader.dataset = rapl_window.dataset
            with self.assertRaisesRegex(AssertionError, "requires shft_cidxs"):
                evaluate_raplkt(
                    _FixedRAPLKT(), rapl_loader,
                    manifest_path=os.path.join(temp_dir, "invalid_rapl.tsv"),
                )

    def test_manifest_assertion_validates_and_canonicalizes_targets(self):
        canonical = pd.DataFrame([
            {"target_cidx": 1, "question_id": 11, "response": 1,
             "kc_ids": "2,3", "k_count": 2},
            {"target_cidx": 2, "question_id": 12, "response": 0,
             "kc_ids": "4", "k_count": 1},
        ])
        reordered = canonical.iloc[::-1].copy()
        reordered.loc[reordered["target_cidx"] == 1, "kc_ids"] = "3,2"
        self.assertTrue(assert_same_benchmark_targets(canonical, reordered))

        invalid_cases = [
            ("non-negative", canonical.assign(target_cidx=[-1, 2])),
            ("must be unique", canonical.assign(target_cidx=[1, 1])),
            ("contains duplicates", canonical.assign(kc_ids=["2,2", "4"])),
            ("disagrees", canonical.assign(k_count=[1, 1])),
            ("response", canonical.assign(response=[2, 0])),
        ]
        for message, invalid in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(AssertionError, message):
                    assert_same_benchmark_targets(invalid, canonical)

    def test_orphan_repeat_is_history_not_target(self):
        row = pd.Series({
            "questions": "1,1,2",
            "concepts": "2,3,4",
            "responses": "0,0,1",
            "selectmasks": "1,1,1",
            "is_repeat": "1,1,0",
            "cidxs": "8,9,10",
        })
        akt = object.__new__(KTDataset)
        akt.max_concepts = 3
        akt.q2cs_map = {}
        _, _, _, akt_masks, _ = akt._group_multi_concept_row(row)
        rapl = object.__new__(RAPLKTDataset)
        rapl.max_concepts = 3
        rapl.q2cs_map = {}
        _, _, _, rapl_masks, _, _ = rapl._group_multi_concept_row(row)
        self.assertEqual(akt_masks, [-1, 1])
        self.assertEqual(rapl_masks, [-1, 1])

    def test_different_kt_q2cs_mappings_have_different_cache_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _write_protocol_files(temp_dir)
            path = os.path.join(temp_dir, config["test_window_file"])
            first = KTDataset(
                path, config["input_type"], {-1}, sequence_mode="collapse",
                max_concepts=3, q2cs_map={1: [1, 2]},
            )
            second = KTDataset(
                path, config["input_type"], {-1}, sequence_mode="collapse",
                max_concepts=3, q2cs_map={1: [1, 3]},
            )
            self.assertNotEqual(first.processed_data, second.processed_data)
            self.assertIn("_v5_k3.pkl", first.processed_data)


class LoaderAndCheckpointProtocolTest(unittest.TestCase):
    def test_train_random_and_all_other_samplers_sequential(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            train_path = os.path.join(temp_dir, "train.csv")
            pd.DataFrame({
                "fold": [0, 1],
                "questions": ["0,1", "0,1"],
                "concepts": ["0,1", "0,1"],
                "responses": ["0,1", "1,0"],
                "selectmasks": ["1,1", "1,1"],
                "is_repeat": ["0,0", "0,0"],
            }).to_csv(train_path, index=False)
            all_config = {"synthetic": {
                "dpath": temp_dir,
                "train_valid_file": "train.csv",
                "input_type": ["questions", "concepts"],
                "folds": [0, 1],
            }}
            akt_train, akt_valid, _, _ = init_dataset4train(
                "synthetic", "akt", all_config, 0, 1, sequence_mode="collapse"
            )
            rapl_train, rapl_valid, _, _ = init_raplkt_dataset4train(
                "synthetic", all_config, 0, 1, sequence_mode="collapse"
            )
            self.assertIsInstance(akt_train.sampler, RandomSampler)
            self.assertIsInstance(rapl_train.sampler, RandomSampler)
            self.assertIsInstance(akt_valid.sampler, SequentialSampler)
            self.assertIsInstance(rapl_valid.sampler, SequentialSampler)

    def test_raplkt_train_shuffle_is_reproducible_and_rng_isolated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            train_path = os.path.join(temp_dir, "train.csv")
            rows = []
            for index in range(12):
                rows.append({
                    "fold": index % 2,
                    "questions": f"{index},{index + 1}",
                    "concepts": f"{index},{index + 1}",
                    "responses": "0,1",
                    "selectmasks": "1,1",
                    "is_repeat": "0,0",
                })
            pd.DataFrame(rows).to_csv(train_path, index=False)
            all_config = {"synthetic": {
                "dpath": temp_dir,
                "train_valid_file": "train.csv",
                "input_type": ["questions", "concepts"],
                "folds": [0, 1],
            }}
            first, _, _, _ = init_raplkt_dataset4train(
                "synthetic", all_config, 0, 2,
                sequence_mode="collapse", seed=123,
            )
            second, _, _, _ = init_raplkt_dataset4train(
                "synthetic", all_config, 0, 2,
                sequence_mode="collapse", seed=123,
            )

            first_epoch = list(iter(first.sampler))
            torch.rand(1000)
            second_epoch = list(iter(second.sampler))
            self.assertEqual(first_epoch, second_epoch)

    def test_akt_true_best_checkpoint_ignores_patience_delta(self):
        model = nn.Linear(1, 1)
        model.model_name = "akt"
        model.emb_type = "qid"
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        validation = iter([(0.8400, 0.70), (0.8405, 0.71)])
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            train_model_module,
            "model_forward",
            side_effect=lambda current, data: current.weight.sum() * 0.0,
        ), patch.object(
            train_model_module,
            "evaluate",
            side_effect=lambda *_args, **_kwargs: next(validation),
        ), patch.object(
            train_model_module.torch, "save", wraps=torch.save
        ) as save_mock:
            result = train_model(
                model, [object()], [object()], 2, optimizer, temp_dir,
                save_model=True, early_stop_min_delta=0.001,
            )
        self.assertEqual(save_mock.call_count, 2)
        self.assertEqual(result[-1], 2)
        self.assertAlmostEqual(result[-3], 0.8405)


if __name__ == "__main__":
    unittest.main()
