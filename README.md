# RAPLKT

本仓库保留 RAPLKT 的最终正式实现。模型入口为 `pykt.models.raplkt.RAPLKT`，训练和预测分别使用 `examples/wandb_raplkt_train.py` 与 `examples/wandb_raplkt_predict.py`。

## 模型

每道题保留最多 `max_concepts` 个 KC。模型根据 question ID 与候选 KC 的独立 Relation embedding 计算 cosine relevance，并用 masked softmax 得到静态题目侧权重 `alpha`：

```text
alpha_relation_qk = softmax(cos(item_relation_q, kc_relation_k) / relation_temperature)
alpha_mean_qk = 1 / K  (valid KC only; padding = 0)
alpha_qk = (1 - relation_mix_lambda) * alpha_mean_qk
           + relation_mix_lambda * alpha_relation_qk
q = sum_k(alpha_qk * kc_embedding_k)
```

`relation_mix_lambda` 是固定超参数，不参与梯度更新，取值范围为 `[0, 1]`，默认值为 `1.0`。`0` 表示 uniform mean pooling，`1` 表示 pure Relation，中间值为线性插值。最终 `alpha` 统一组合当前题目的 KC representation 和 KC difficulty。模型随后执行第一遍共享 AKT，得到 KC 局部成功证据 `p_k`。错误作答的学生侧责任为：

```text
rho_k = softmax_k(log(1 - p_k) / responsibility_temperature)
```

`responsibility_temperature` 必须为正数，默认值为 `1.0`；此时公式严格退化为原来的 `normalize(1 - p)`。小于 `1.0` 时责任分布更尖锐，大于 `1.0` 时更平滑。正确作答按有效 KC 数量均分成功证据，每个 KC 增加 `1/K`；错误作答按 `rho` 分配失败证据。两种情况下单道有效题目的总证据量均恒为 1。历史 Beta 状态通过 `scatter_add` 和 exclusive cumulative sum 并行构造，当前题证据不会进入当前题预测。第二遍共享 AKT 复用 Relation-aware 当前题 query；历史 interaction 及其 difficulty 始终由责任权重路由。训练目标为题级 KT BCE、KC 集合部分标签损失和难度正则项之和：

```text
loss = kt_loss + set_loss_weight * set_loss + regularization
P(question correct) = product_k(p_k)
```

模型没有沿时间步执行的 Python recurrence。两遍 AKT 均保留 causal mask，Beta 使用 exclusive prefix。

## 数据

数据配置位于 `configs/data_config.json`。多 KC 序列由 RAPLKT 数据加载器生成，支持：

- `collapse`：将同一题的 KC 记录合并为一道多 KC 题，正式实验推荐使用。
- `preserve`：保留输入中的题级序列边界。

首次运行会在数据目录生成与参数对应的缓存。更换 `max_concepts`、`sequence_mode` 或预处理数据后会使用新的缓存键。

## AutoDL 训练

在项目根目录执行：

```bash
cd /root/autodl-tmp/RAPLKT
export PYTHONPATH="$PWD"
CUDA_VISIBLE_DEVICES=0 python -u examples/wandb_raplkt_train.py \
  --dataset_name assist2017 \
  --emb_type qid \
  --save_dir saved_model_raplkt_relation \
  --seed 3407 \
  --fold 0 \
  --d_model 256 \
  --d_ff 256 \
  --num_attn_heads 4 \
  --n_blocks 2 \
  --dropout 0.1 \
  --learning_rate 0.0001 \
  --batch_size 64 \
  --num_epochs 200 \
  --early_stop_rounds 10 \
  --early_stop_min_delta 0.001 \
  --max_concepts 2 \
  --sequence_mode collapse \
  --evidence_prior_alpha 1.0 \
  --evidence_prior_beta 1.0 \
  --relation_dim 32 \
  --relation_temperature 1.0 \
  --relation_mix_lambda 1.0 \
  --responsibility_temperature 1.0 \
  --set_loss_weight 0.1
```

进行 lambda sweep 时，分别把 `--relation_mix_lambda` 设为 `0`、`0.25`、`0.5`、`0.75`、`1`，并使用不同的 `--save_dir`。只根据 validation AUC 选择每个数据集的 lambda，再报告该 checkpoint 的 test 与 window-test；不得依据 test 指标选择 lambda。

训练结束时终端会打印 `model_save_path`。它所在目录同时包含 `config.json`、最佳 checkpoint 和 `training_summary.json`。

## AutoDL 预测

将下面的目录替换为训练输出的实际运行目录：

```bash
cd /root/autodl-tmp/RAPLKT
export PYTHONPATH="$PWD"
CUDA_VISIBLE_DEVICES=0 python -u examples/wandb_raplkt_predict.py \
  --save_dir saved_model_raplkt/assist2017_fold0_seed3407_你的运行哈希 \
  --bz 256 \
  --debug_alignment 1 \
  --debug_responsibility 1 \
  --responsibility_sample_size 12
```

预测严格读取训练目录中的配置并使用 `strict=True` 加载权重。结果写入 `qid_evaluation_metrics.json`、题级预测文件和 benchmark manifest；报告包含整体、单 KC、多 KC、受责任历史影响样本、KC 集合似然及责任分布诊断。

## 验证

```bash
python -m pytest -q tests/test_raplkt.py
```


## Data preprocessing

Only Algebra2005, ASSIST2017, and EdNet-KT1 5k adapters are included. Shared splitting and utility modules remain required.

Run from `examples/` with the repository root on `PYTHONPATH`:

```bash
python data_preprocess.py --dataset_name algebra2005
python data_preprocess.py --dataset_name assist2017
python data_preprocess.py --dataset_name ednet_kt1_5k --file_path ../data/ednet_kt1_5k/
```

The EdNet adapter is `ednet_kt1_preprocess_20260826_010359.py`. It reads an existing dataset directory containing `subset/ednet_kt1_5k_v1_manifest.json`, or a manifest JSON supplied via `--file_path`. Use `--file_path` to override the raw input for any dataset.
