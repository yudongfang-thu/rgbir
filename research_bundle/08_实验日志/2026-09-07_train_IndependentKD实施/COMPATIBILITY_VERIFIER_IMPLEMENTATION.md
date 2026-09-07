# N/C0 真实兼容验收器实现回执

**结论：`verify_compatibility.py` 已完成可执行 CLI 和严格回执路径，4 项本地 CPU helper 测试通过；尚未在 94 执行真实 loader/训练验收，因此没有生成任何 ACCEPTED compatibility 回执。**

日期：2026-09-07。只新增此 verifier，不改父流程的 runtime/trainer/criterion，不 SSH、不使用 GPU、不调度任务。真实运行由统一资源 lease 调度。

## 运行接口

```text
python verify_compatibility.py --config <yaml> --output <fresh data-disk directory> --seed 0|42|123|all --arm N|C0
python verify_compatibility.py --self-test
```

真实运行硬固定 E200 recipe 中的 imgsz640、batch32、nbs64、workers4；检查的是开头 30 个完整 batch 以及 24 次真实成功 optimizer updates，**不是整个 E200 等价**。输出强制在 `/mnt/dataset/yudongfang/` 下，并禁止放到源码树中。所有旧原始 trace 保留，不覆盖、删除或重命名。

## 完整验收顺序

1. 使用 native dataset 和 build_dataloader 的同配置两条路径构造独立 old/new loader；旧用 ORIGINAL_DATASET，新用 TrackedDualLabelRGBIRDataset。
2. `RNGObservedDataset` 在原 paired `__getitem__` 前后只读取 Python/NumPy/Torch CPU RNG，collate 附带 worker ID、seed、dataset index 和 RNG 状态；不增加随机采样，也不改原字段。30 批逐张 RGB/IR 像素、双标签、所有原字段和逐样本 worker RNG 精确比较；只允许 tracked pair_info 增加几何元数据。父进程 RNG 同样比较。
3. 两 trainer 顺序运行，避免双份 GPU 模型常驻；两次均从同 seed 启动。真正原 criterion/dataset 类身份在 on_train_start 核验，防止全局 monkey-patch 导致 new-vs-new。
4. 保存/比较全学生参数与 buffers、EMA、optimizer/scaler、trainable names、CPU/CUDA RNG、冻结辅助模型状态。
5. 每次 optimizer attempt 保存/比较全部参数的 scaled 梯度；额外包装实际 optimizer.step 记录真实调用和真正传给优化器的 unscaled+clipped 梯度。AMP skip 不能伪计为成功。
6. 每次 attempt 后比较全学生状态、optimizer state/param groups、EMA及其实际 updates、scaler、CPU/CUDA RNG、成功/尝试/skip 计数。按 pinned native 的真实 EMA 行为比较，不假定 skip 一定不更新 EMA。
7. 两侧均必须恰好完成 24 次实际 optimizer.step；C0 还必须有非零 KD score 梯度和选中对象。超过 96 次 attempt 仍无法完成按技术失败返回。最终教师/参考参数与 buffers 不变、无梯度、不在 optimizer。

旧路径 `.pt` trace 包含 state/gradient/RNG/小体积标签与文件身份，新路径仅加载比较，不保存重复状态。30 批数据检查不保存原图全集；旧 frozen trainer 自身 canary 保存的 `first_batch.pt` 和 `initial_student.pt` 沿用保留。训练 trace 的像素全量等价来自单独 30-batch loader 比较，训练中的每批保存/比较源文件、标签和 worker RNG，不把它写成另外保存了完整训练图像。

## 回执契约

`compatibility_receipt.json` 仅全部完成且逐项通过时写 `status=ACCEPTED`，含：

- seed（单 seed 数值；all 时 seeds 与 records）、arm、dataset/model/teacher/reference；
- successful_updates=24、loader_batches=30、trajectory_exact=true；
- worker_rng_exact、每 seed 详表、资源峰值与 lease；
- source_files 的 relative/accepted_copy：运行前冻结的完整源码副本清单；
- official_test_accessed=false、明确 scope 边界。

失败写 `failure_receipt.json`，保留已有完成子记录和全部失败 trace，不写成功回执、不使用容差替代精确验收。浮点 Tensor 用原始字节比较，包含 AMP NaN/Inf 的位模式；是否真可更新则由优化器真实调用及有限 applied gradient 单独检查。

## 已运行的本地检查

`compatibility_helpers_cpu_attempt2.log`：4 tests / 0 failures / 0 errors，覆盖 tensor dtype/shape/bytes、NaN 位一致、递归差异边界、只允许 pair_info 新增字段、观测 wrapper 不改变原样本或 CPU RNG。

attempt1 在本地 PyTorch 1.8 的不同字节数 dtype-view 限制下报错，保留原日志；改用 NumPy 原始字节读取（BF16 先同宽 int16 view）后通过。没有生成哈希。真实 CUDA、loader/trainer 接口及全参数轨迹需 94 pinned 环境再次执行和独立审阅。
