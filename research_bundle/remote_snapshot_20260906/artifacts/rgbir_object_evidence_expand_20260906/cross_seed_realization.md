# 跨 seed 随机性实现核验（2026-09-06）

> 结论：seed0 与 seed123 的实际训练 seed 正确生效，初始学生模型有12个张量不同；两者首批输入/学生标签相同，canary前30批的样本顺序相同。该现象来自固定版本 Ultralytics 原生 DataLoader 的固定 generator seed，并非 launcher 丢失 seed 覆盖。本检查未修改方法、数据流水线或运行队列。

## 只读 CPU 检查结果

逐一核验四个 canary 的 `args.yaml`、`launch_manifest.json`、`completion_receipt.json`、worker `effective_config.yaml` 与实际命令 `--seed`，均分别为0或123。IR教师与RGB参考继续固定seed42。

| seed0 对 seed123 | paired | weight0 |
|---|---:|---:|
| 初始 state 中全部张量数 | 499 | 499 |
| 逐张量完全相同 | 487 | 487 |
| 逐张量不同 | 12 | 12 |
| 首批保存的5个输入/标签张量全部相同 | 是 | 是 |
| 首批32张图像的样本路径顺序相同 | 是 | 是 |
| canary全部30批的样本路径顺序相同 | 是 | 是 |

不同的初始权重键均位于 `model.23.cv3.*`，逐键形状、不同元素数量与最大绝对差保存在 JSON。首批实际比较键为 `img`、`strong_img`、`cls`、`bboxes`、`batch_idx`；这里没有额外保存的 IR 标签张量，故不扩展宣称已经逐张量比较 IR 标签。

## 原生代码解释

固定环境为 torch 2.10.0+cu128、Ultralytics 8.4.115。`BaseTrainer.__init__` 按 `self.args.seed + 1 + RANK` 设置全局随机种子；因此不同学生 seed 会产生不同的随机初始化部分。原生 `build_dataloader` 独立创建 generator 并固定为 `6148914691236517205 + RANK`，未把训练 `args.seed` 加入 DataLoader generator。`seed_worker` 再用 `torch.initial_seed()` 派生 NumPy 与 Python worker 种子。

这解释了所测单 GPU、同 workers 配置下首批增强后图像/学生标签及前30批样本顺序一致。原生函数源码快照保存在 `cross_seed_native_source.txt`。

## 结论边界

当前跨 seed 重复覆盖初始化随机性，未覆盖所测数据顺序/增强随机性的变化。固定数据流下仍可研究不同随机初始化的条件重复，后续 mean±SD 应注明该原生数据流水线条件。本检查没有检验统计独立性，也没有逐张量比较完整200轮增强结果；不据首批相同否定 seed 统计。

当前 paired/weight0 的同 seed 对照保持可解释，且本轮研究协议已冻结；因此保留现有队列和方法设置，不因这个原生行为临时改种子或停止训练。本核验不提供性能增益结论。

## 产物路径

- 本地：本目录 `cross_seed_realization.py`、`cross_seed_realization.json`、`cross_seed_native_source.txt` 与本文。
- 94：`/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_object_evidence_expand_20260906/` 下同名文件。
- 读取的 canary：`/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbir_object_evidence_expand_20260906/canary_{paired,weight0}_s{0,123}_attempt1/`。
- 执行设置：空 `CUDA_VISIBLE_DEVICES`，所有 checkpoint/首批张量均 `map_location='cpu'`；没有 GPU 操作，没有写入或覆盖原始训练产物，没有计算哈希。
