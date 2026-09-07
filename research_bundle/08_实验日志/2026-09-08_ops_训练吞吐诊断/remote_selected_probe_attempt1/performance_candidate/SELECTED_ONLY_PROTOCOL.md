# Selected-only 原池化候选

结论：已完成独立候选与 CPU 检查；**尚无真实 GPU/24 更新/生产替换准入**。不修改正在运行的 release，不改任何科学方法、λ 或容差。

`selected_only_v1.py` 通过显式 `make_api(pinned_selection_adapter, pinned_classification_logit)` 绑定。第一遍执行原 `build_classification_selection` 字节码的 globals 副本，只有全类 pool 替换为原 region 验证/count 与私有可微占位。原 GT-only S/T/R evidence、匹配、候选、q、eligible、rho、稳定排序、base 与分母、所有 GT 对背景环的排除和 C0 统计完全保留。第二遍对已选中 region 只执行原逐对象 S/T pool，并填回 `[M_base,L,C]`；原分类 loss 的行数、有效层平均及归约顺序保留。

未选 delta 与全部 reference_delta 在学习存储中是私有占位，不是测量，不得直接调用原完整统计器解读它们。候选自有 loss 接口对未采集字段写 null/列出 missing，尤其不会把原 C0 entropy/clipping 继承成 C1 结果。原 R 的 GT evidence/candidate 与教师/参考前向均保留。

第一/第二 pass 共用一次完整 student score.float()，以保留原 FP32 分支梯度汇合后再转 raw FP16 的位置；不用每区独立转换回半精度。原 raw source/version 绑定明确恢复。该补充前的源码保留于 `selected_only_source_before_shared_float/`。

启用范围：三个真实 raw score 张量均为 finite FP16。其有限绝对值不超过 65504，经 float32 LME 及前景减背景不会发生 float32 溢出，因此不会漏掉原未选 delta 的溢出错误。非 FP16 或 raw 非有限直接执行原完整 selector/loss，保留原异常处理；不把 FP32 极端输入错误静默变成成功。CPU 另检查 FP32 回退。原 pool/mask 校验仍存在。

每批学习 loss 始终来自 selected-only 图。完整统计批使用本次相同 raw 在 no_grad 下运行原完整 selector/loss，不新增模型前向，不消费 RNG；其 wall 双侧同步另计 `full_diagnostics_seconds`。前三批、每 `log_every_batches`（正式配置 100）、sanity 和固定 epoch 首批 shared-gradient-observer 触发完整统计。原 native/actual B/λ、sanity 及 target/off-target 共享梯度观察代码不变。

原 24-update canary 设置 `sanity=True`，因此每批完整统计会遮住薄路径速度。必须同时保留完整 wall 和另列扣除诊断段的时间；插桩改变重叠/缓存，不能把后者叫无插桩生产吞吐。

CPU：`selected_only_cpu_attempt2/receipt.json` 的 18 个比较均通过固定 atol=1e-6、rtol=1e-5；本批真值的 loss、selected S/T delta、学生 score 梯度均 exact。覆盖空集、无选择、多图多类、无效尺度、额外未配对 RGB GT 背景排除、thin/full diagnostic 两模式及 FP32 原路回退，T/R/DFL 不接收 KD 梯度。没有 CUDA 初始化。attempt1 是测试夹具 deepcopy 非叶 Tensor 异常，保留原回执和测试源码；改为明确 detach 的递归 CPU 映射后重跑，无方法或容差修改。

共用完整 FP32 cast 与训练计数补充后再次执行相同 18 项，`selected_only_cpu_attempt3/receipt.json` 全部通过；这是当前候选版本的 CPU 回执。

真实 bundle 入口（由 root 在原 lease 下执行）：

```text
python benchmark_selected_only.py --reference-dir <release_gpu5> --mode bundle --bundle <real_probe_attempt1/capture_attempt1/raw_batch_00.pt> --device cuda --output <fresh_selected_only_replay_attempt1> --iterations 7 --warmup 2
```

同目录依赖：`selected_only_v1.py`、`benchmark_selected_only.py`、原 `benchmark_candidate.py` 和它导入的 `pool_block16.py`（仅复用读取、计时、比较工具；候选学习池化不使用 block16）。只读原 raw bundle，不重跑 capture，不更改原容差。比较 selected S/T、原所有 gates/count/region/source、完整 loss 和 score 梯度；明确不比较未采集 delta。thin 计时包含真实完整 gate/最小统计/score backward，不含模型、loader、optimizer 或完整额外诊断。

可选新解释器训练接口 `make_criterion_type(pinned_independent_criterion, api, selection_observer=None)` 返回独立私有类；调用者须在自己独立的 `build_trainer` globals 副本绑定它。它运行原 criterion.__call__ 字节码，只替换 selector 与 loss 接口，原生产模块不变。累计 `selected_only_diagnostics_seconds_total` 供独立训练计时分层。此接口尚待独立代码审阅/真实训练检查，不会改现有已接受 block16 24-update 入口。

私有 criterion 另累计 `thin_learning_batches`、`full_diagnostics_batches` 和 `fallback_batches`，须写入新的训练回执，证明 sanity 每批重统计时确实走了薄学习图。

数学相同、有限精度通过与逐更新轨迹精确是三层结论。减少零权重图可能改变浮点梯度累加拓扑，CPU exact 不外推 GPU 或长训；真实 loss/grad/update 差异须原样报告。此前 block16 teacher_delta 超容差失败独立保留，不因本候选未测该量而改写其结论。
