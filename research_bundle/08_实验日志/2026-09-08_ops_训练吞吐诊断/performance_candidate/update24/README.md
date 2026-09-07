# C1 旧版 / block16：24 次真实更新诊断

**代码与16项CPU比较/闭包真值已通过；本目录准备由root按全局lease运行的旧/新各24次成功更新诊断。没有本地GPU/SSH执行，结果不自动准入E200生产切换。**

入口为 `compare_24_updates.py`。CPU测试 `test_compare_cpu.py`，最新回执 `cpu_checks_attempt2.json`（torch1.8 CPU，CUDA未初始化）；首次测试文件名笔误与失败回执原样保留，见 `CPU_ATTEMPT1_NOTE.md`。固定 atol=1e-6、rtol=1e-5，禁止通过改容差修复失败。

## 运行范围

父协调器保持无CUDA上下文，在**继承的同一个globallease**下，顺序启动两个fresh Python解释器：old、block16。各保留完整formal C1 seed42 cfg、原始yolo11n初始化、E200日程、B32/nbs64/workers4/AMP、实际分类系数0.09227393550836771、定位系数0。只做前24次真实optimizer.step，最多96次attempt；不是独立E20短recipe，也不是20260907自然诊断流。

两边均调用 release `build_trainer(...historical=False,max_steps=24)`，使用TrackedDualLabelRGBIRDataset和原IndependentCriterion逻辑。私有criterion子类的 `__call__` 保留原code/closure，只把其selector引用换成记录包装；selector函数globals副本仅绑定不同pool。旧module、原函数globals及release磁盘源码不变。`build_trainer`内部历史kd_weight=.1不覆盖C1实际使用的classification_coefficient；不调用会改C1系数的verify_compatibility.run。

复用 release verify_compatibility 的clone_cpu、RNGObservedDataset、all_rng_state、梯度/模型快照、辅助模型检查及loader关闭函数；**不调用它的run/run_worker，也不调用train.run、emit/capture evidence binding或新增hash函数**。源文件用逐字节复制核对，输入模型用path/size/mtime；loader原实现保持原状。

## 保留和比较的证据

- 每个实际训练batch：源文件、双标签GT、增强/配对metadata、worker RNG前后/worker id/seed、父进程RNG、图像shape；整批C1选取一次，记录全部base/selected/eligible身份、掩码/region及原始delta/quality/C0统计。
- 首批由原生canary保存的first_batch.pt加入**严格字节比较**，含RGB/IR像素及原保存GT字段。后续图像像素不另存，因此后续的“flow exact”专指source/metadata/GT/worker RNG，不能写成全部像素都重比过。
- 初始student/EMA/optimizer/scaler/RNG、trainable身份及冻结T/R状态精确；每个attempt保留scaled gradients、native+KD loss、原生loss items、实际到达optimizer的unscaled/clipped gradients、更新后student/optimizer/EMA/scaler和计数/RNG。非finite梯度不得实际进入optimizer.step，AMP skips与实调用计数相互核对。
- 每worker结束要求恰好24真实更新、有限非零KD score梯度、T/R参数和buffer不变。T/R完整raw前向张量不逐批序列化；不能把冻结state不变描述为每次raw输出已独立字节核对。
- 文件清单须与完整batch/attempt数逐个闭合。源码快照逐字节、cfg副本逐字节、输入模型stat跨worker一致；缺失state文件拒绝比较。

## 三个精度层级分开

1. **数学同定义**：masked LME表达式是否相同属于源码审阅；24步结果不能证明数学公式。
2. **中间浮点精度**：selection_floating记录S/T/R delta、quality、C0 scalar/stat。固定容差失败单列，即使后续loss/grad恰好一致也不删除。当前v1已有raw teacher_delta约1.907e-6的超差原回执；本次诊断获授权用于观察真实影响，不因此声称全selector数值验收通过。
3. **训练轨迹**：initial/首批像素/flow/selection身份/整数与AMP/RNG控制先要求exact，再分别报告loss、梯度、参数、optimizer、EMA的bitwise_exact与numeric_agreement。比较用float64计算 `abs(new-old) <= 1e-6 + 1e-5*abs(old)`。匹配的非finite位模式另计数，不当作有限可用梯度；浮点容差通过也不称trajectory exact。

浮点或选择差异保留为诊断结果，不修正、不放宽阈值；COMPLETED_COMPARISON_NOT_PRODUCTION_ADMISSION表示比较执行完成，不表示所有数值PASS。任何后续E200切换都不由本入口自动执行。

## 计时与资源

逐batch双同步计时，审计state复制/写盘/辅助检查耗时单列并扣除，固定前6批作为计时热身单列。on_train_batch_start/end跨度不含loader fetch；总trainer跨度扣除审计时间仍含初始化及原生canary检查/保存。仪器化改变正常重叠，**这些耗时不是完整生产吞吐**。CPU旧/新状态比较在两worker均退出后执行，耗时单列。

完整状态和原生首批像素可能达到数GB，只写`/mnt/dataset/yudongfang/`的新诊断目录；root默认只回收小回执，不向本地或系统盘下载全部状态。输入模型前后stat变动立即失败，原结果/失败attempt不覆盖。GPU/NVML/RSS由原guard实测管理；本入口不选择卡、不创建额外lease、没有自行启动权限。

## 调用与依赖

部署新增运行文件只有：

- `performance_candidate/update24/compare_24_updates.py`
- `performance_candidate/pool_block16.py`（本attempt冻结的v1文件）

已有`--reference-dir`必须是完整已验收C1 release（含verify_compatibility.py、runtime.py、train_independent.py、independent_criterion.py、selection_adapter.py、classification_logit.py、gradient_observation.py、protocol.py及task_conditional_reference/legacy_oev1依赖）。pinned Python/torch2.10.0+cu128/Ultralytics8.4.115和原globallease环境由root提供；配置使用原formal C1_s42.yaml，不另造或改系数。test_compare_cpu.py/README/回执供审阅，无需GPU运行依赖。

```text
PINNED_PYTHON performance_candidate/update24/compare_24_updates.py --reference-dir EXISTING_C1_RELEASE --config ORIGINAL_FORMAL_C1_S42_YAML --pool-source performance_candidate/pool_block16.py --output NEW_DATA_DISK_ATTEMPT
```

复核已完成两worker时可只用CPU：

```text
PYTHON compare_24_updates.py --compare-only OLD_DIRECTORY BLOCK16_DIRECTORY --output NEW_CPU_COMPARISON_DIRECTORY
```

本目录只准备诊断，不修改现行长训、原始结果或公共索引。root负责审阅接受后部署；作者不在此调用GPU/SSH。
