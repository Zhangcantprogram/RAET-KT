# RAET-KT

PyTorch implementation of RAET-KT for multi-concept knowledge tracing. The model combines masked-mean multi-concept embeddings, response-dependent responsibility allocation, and parallel evidence memory using a shared AKT backbone.

## Setup

The environment is defined in `environment.yml`:

- Python 3.10
- PyTorch 2.11.0
- NumPy 1.26.4
- Pandas 2.2.2
- Scikit-learn 1.5.2

```bash
git clone https://github.com/Zhangcantprogram/RAET-KT.git
cd RAET-KT
conda env create -f environment.yml
conda activate raet
pip install -e .
```

The supplied environment uses the CUDA 12.8 package index. Run the commands below from the project root unless stated otherwise.

## File Structure

```text
RAET-KT/
|-- configs/
|   |-- data_config.json          # Dataset paths and sequence metadata
|   |-- kt_config.json            # Training settings and AKT configuration
|   `-- best_model.json           # Optional checkpoint registry
|-- data/                         # Downloaded and processed datasets
|-- examples/
|   |-- data_preprocess.py        # Preprocessing entry point
|   |-- wandb_raetkt_train.py      # RAET-KT training
|   `-- wandb_raetkt_predict.py    # Checkpoint evaluation
|-- pykt/
|   |-- datasets/                 # Sequence loading and multi-concept alignment
|   |-- models/
|   |   |-- raetkt.py             # RAET-KT model
|   |   |-- raetkt_train_model.py # Training objective and evaluation
|   |   `-- akt.py                # Shared AKT backbone
|   |-- preprocess/               # Three dataset adapters and shared utilities
|   `-- utils/                    # Reproducibility and logging utilities
|-- tests/                        # Model, preprocessing, and protocol checks
|-- environment.yml
`-- setup.py
```

Question embeddings and question-side difficulty embeddings average only valid KCs; padding is excluded.

The Python model class is `pykt.models.raetkt.RAETKT`; its configuration identifier is `raetkt`.

## Data and Data Preprocessing

This release supports Algebra2005, ASSISTments2017, and an EdNet-KT1 subset of 5,000 students. Raw datasets, processed sequences, and trained weights are not included.

Place raw data under `data/`:

| Dataset | Input |
| --- | --- |
| Algebra2005 | `data/algebra2005/algebra_2005_2006_train.txt` |
| ASSISTments2017 | `data/assist2017/anonymized_full_release_competition_dataset.csv` |
| EdNet-KT1 | Student CSV files, question metadata, and the subset manifest described below |

### Algebra2005 and ASSISTments2017

Run preprocessing from `examples/`:

```bash
cd examples
python data_preprocess.py --dataset_name algebra2005
python data_preprocess.py --dataset_name assist2017
cd ..
```

Use `--file_path` to specify a different raw input. Shared preprocessing creates training, validation, test, and window-test sequences and updates `configs/data_config.json`. The default settings are five folds, a minimum sequence length of 3, and a maximum length of 200.

### EdNet-KT1

The adapter is `pykt/preprocess/ednet_kt1_preprocess_20260826_010359.py`. It reads a deterministic subset manifest and the original student and question files.

To create a manifest, place student CSV files in `data/ednet/KT1/` and question metadata in `data/ednet/contents/questions.csv`, then run this Python code from the project root:

```python
from pathlib import Path
from pykt.preprocess.ednet_kt1_preprocess_20260826_010359 import (
    load_question_metadata, select_students, write_subset_manifest,
)

raw_dir = Path("data/ednet/KT1")
questions_csv = Path("data/ednet/contents/questions.csv")
student_files = sorted(raw_dir.glob("u*.csv"))
metadata, _ = load_question_metadata(questions_csv)
selected, candidate_count = select_students(student_files, metadata)
write_subset_manifest(
    selected, candidate_count, raw_dir, questions_csv,
    Path("data/ednet_kt1_5k/subset"),
    raw_student_file_count=len(student_files),
)
```

Then preprocess the selected students:

```bash
cd examples
python data_preprocess.py --dataset_name ednet_kt1_5k --file_path ../data/ednet_kt1_5k/
cd ..
```

Reuse the original manifest to reproduce an existing experiment: rebuilding a subset uses the current RAET-KT selection namespace and can select different students. The `ednet` command-line alias resolves to `ednet_kt1_5k`.

## Training and Testing RAET-KT

### Training

For Algebra2005:

```bash
python examples/wandb_raetkt_train.py \
  --dataset_name algebra2005 --save_dir saved_model_raetkt \
  --seed 3407 --fold 0 --d_model 256 --d_ff 256 \
  --num_attn_heads 4 --n_blocks 2 --dropout 0.1 \
  --learning_rate 0.0001 --batch_size 64 --num_epochs 200 \
  --max_concepts 8 --sequence_mode collapse \
  --responsibility_temperature 1.0 \
  --set_loss_weight 0.1
```

For ASSISTments2017 or EdNet-KT1, use the same command with `--dataset_name assist2017` or `--dataset_name ednet_kt1_5k`. These commands are runnable starting settings; select dataset-specific hyperparameters using validation performance.

The `collapse` sequence mode combines repeated concept rows into one multi-concept question. Set `--max_concepts` to cover the intended concept sets. Use `--fold` values `0` through `4` for five-fold experiments.

The trainer prints `model_save_path` and saves the run configuration, best validation checkpoint, and `training_summary.json` in the run directory.

### Testing

Replace the path below with the run directory printed during training:

```bash
python examples/wandb_raetkt_predict.py \
  --save_dir saved_model_raetkt/algebra2005_fold0_seed3407_RUN_ID \
  --bz 256
```

Evaluation loads the saved configuration and checkpoint. Outputs include:

- `qid_evaluation_metrics.json`: overall and grouped test/window-test metrics.
- `qid_test_predictions.txt` and `qid_window_test_predictions.txt`: question-level predictions.
- `test_manifest.tsv` and `window_test_manifest.tsv`: target-alignment records.

Select hyperparameters and checkpoints using validation data before reporting test results.

## Verification

```bash
pip install pytest
python -m pytest -q tests
```
