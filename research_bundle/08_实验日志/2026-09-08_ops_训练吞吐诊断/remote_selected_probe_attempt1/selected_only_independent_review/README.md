# selected_only_v1 独立审查

**结论：可进入原 raw bundle 的有界 GPU 诊断。** 当前源码和 CPU 证据未发现阻断该诊断的错误；这不构成 GPU 数值检查通过、24-update 接受、正式训练切换或提速倍数结论。

2026-09-08，loc_stress。审查 `performance_candidate/selected_only_v1.py`、`benchmark_selected_only.py`、共享 benchmark 及冻结 selector/loss/criterion/observer。只执行 CPU 小型验证；未运行 GPU、SSH、模型或权重加载，未计算新 hash、未修改候选/生产源码。原失败测试产物全部保留。

## 源码判断

1. `first_pass` 仅克隆原 selector 的 globals，将全类 pool 暂换成保留检查、区域计数和 graph-connected 零存储的 deferred pool。原 C0 matching/evidence/candidate/q/rho/tie/base、全部 GT 背景排除和 region 身份仍由原函数计算。全 base 第一维 M 与 reduction 位置不变；只向 selected 的 S/T 槽位写入原逐对象 pool 真值，没有改用 K 分母。common-valid 校验在 first pass 全量保留，第二遍 selected 的 S/T validity 再与原值比较；R 与 S 共享 RGB 几何，因此无需另算 R 全类 logits 获得 validity。
2. 仅 S/T/R 三者 raw score 均为有限 FP16 时启用薄路径；其他 dtype 或非有限值回原 selector，保留原验证。FP16 值转 FP32 后，当前区域/通道规模的 LME、差值及 loss 归约不存在极端 FP32 输入风险。raw DFL / labels / source 身份验证仍由原实现运行。空集和未选行用连到 S score 的私有零载体，未伪造诊断；T/R score 和所有 DFL 无直接 KD 梯度。
3. loss 始终来自 `payload.learning`。即使 `full_diagnostics=True`，旧完整 selector/loss 仅在同一 raw 的 `torch.no_grad()` 中生成统计，未新增 model forward；完整统计时间单列。非采集字段明确 null，并列 `missing_diagnostics`；C0 entropy/clipping 不冒充 C1 统计。原公共 `ClassificationSelection` 的 base 数量/组件接口没有被 selected-only 张量尺寸替换。
4. `TrainingSelection.__getattr__` 将 `classification_loss_components` 转发到实际 learning payload，保留 target/off-target 图。尾部 factory 克隆原 criterion `__call__` bytecode，仅替换选择和损失函数，原 native/aux forward、actual B/coefficient、sanity 和 shared-gradient observation 流程仍执行。日志 due 使用原调用已递增的 `self.calls`，保留 sanity、前三批、配置 cadence，另含固定 epoch observer due。
5. benchmark old/new 使用同 raw 克隆、相同梯度输入和绑定单 GPU 的既有 lease；旧完整 selection/statistics 与新薄 selection/minimal statistics 的比较属于**实现路径差异**，不单是 pool 算子。计时不含模型、loader、optimizer；full diagnostics 模式只做等价检查，不在新路径保留计时样本中。固定容差继承原 `atol=1e-6,rtol=1e-5`，不能把 CPU 通过预先当 GPU 通过。

## CPU 证据

- [独立复跑 receipt]（服务器/本地保留，未包含于本阶段发布：cpu_attempt1/receipt.json）：作者预定 18 项全部 `PASS_NUMERIC_EQUIVALENCE_ONLY`，含 FP16 thin 普通/完整统计模式、FP32 fallback、empty、单类、多类无效尺度、多图中间空图、全量 region/mask 身份、选中 delta、loss、raw gradient 与日志 cadence。
- [补测脚本 attempt2]（服务器/本地保留，未包含于本阶段发布：check_supplement_attempt2.py） 与 [补测 receipt attempt2]（服务器/本地保留，未包含于本阶段发布：supplement_receipt_attempt2.json）：23 项 PASS。追加 C1/C1_y 的 target/off-target component 与 raw gradient、完整统计逐字段相等、source assertion、M>0/K=0 的可反向零、FP16/FP32 NaN/Inf 与原路径相同拒绝；FP32 极端值确实 fallback。Torch 为 `1.8.0+cu111`，`cuda_initialized=false`；候选小文件在补测前后字节相同，原 selector binding 未变。
- [补测 attempt1]（服务器/本地保留，未包含于本阶段发布：check_supplement.py） 与 [原失败 receipt]（服务器/本地保留，未包含于本阶段发布：supplement_receipt.json） 保留：22 项通过，最后一项失败来自审阅脚本在 fallback 分支误访问仅 thin 提供的 `missing_diagnostics`。attempt2 修正这项测试断言，未改候选。另一个原断言误要求构造的极端 FP32 案例必然报错；该案例实际原路径合法返回，attempt2 改为比较两边一致结果，而非强求错误。

补测的极端案例也限定先前可行性文字：不能把“+3e38 前景、-3e38 背景必然让此实现的 pool 差值溢出”当作已知真值；原 pool 使用有限 sentinel，实际行为需按完整代码判断。全类后续归约、不同 dtype 与原异常域仍需保留，候选的保守 FP16 域限制及 fallback 不依赖该失败构造成立。

部分多对象/未配对 GT fixture 的 selected=0，故它们主要验证完整 masks/身份与空选择；有实际 KD 的单类、多类多图案例负责非零 loss/gradient 比较。CPU 补测不含真实模型 shared-parameter backward，GPU 同 bundle 也不能替代它。

## GPU 与后续 24 update 边界

允许根按原 lease、原 raw bundle、冻结两预热/七计时执行诊断，不放宽门/容差、不重选输入。本审阅不启动 GPU。

**24-update 入口尚需明确记录 `thin_learning_batches`、`full_diagnostics_batches` 和 `fallback_batches`。** 当前所读 factory 仅累计 diagnostics seconds；这些计数能证明 max_steps 导致 sanity=True、每批全统计时仍使用薄学习图。factory 的 `selection_observer` 可以为独立入口提供该记录，但需要真实 receipt 证实。

sanity 24 update 的完整 wall 必须保留；可另列同步测得的完整诊断时间，但其差值受插桩、缓存与重叠扰动，不能直接称为生产吞吐。固定 epoch 0/10/50/100/199 首批的 shared-gradient observer、target/off-target 组件与 sanity 检查不能随日志降频。正式训练和 GPU 迁移仍需后续独立准入。

## 最新共享 cast / 计数补充复核

作者随后保留补充前源码于 `selected_only_source_before_shared_float`，新版本在 first pass 和 selected pool 之间共用一次完整 `student['scores'].float()`；T/R 在 no_grad 中 detach/float。它保留原先先汇总 FP32 score 梯度再回传 raw FP16 的结构，避免分片各自先转回半精度。first pass 使用临时 float 字典，返回 learning 时恢复原 raw `source_tensors/source_versions`；batch 身份未变。独立 source assertion、loss 和组件梯度补测通过。

新增 [CPU 复跑]（服务器/本地保留，未包含于本阶段发布：cpu_attempt2_shared_cast/receipt.json） 18 项 PASS，以及 [补测 attempt3]（服务器/本地保留，未包含于本阶段发布：supplement_receipt_attempt3_shared_cast.json） 23 项 PASS（脚本 `check_supplement_attempt3_shared_cast.py`）；CUDA 未初始化，小文件在该补测前后字节相同，原 binding 不变。历史 receipt 均保留。

factory 已补齐 `thin_learning_batches`、`full_diagnostics_batches`、`fallback_batches`，按成功 loss 调用累计并写 stats，因此上文“计数缺失”的静态待办解除。实际 24-update receipt 仍需核对这些字段与成功 update/batch 数量，不由源码代替实际执行验收。**当前共享 cast 版本 READY：可进入原 bundle 的有界 GPU 诊断；正式训练准入状态未变。**
