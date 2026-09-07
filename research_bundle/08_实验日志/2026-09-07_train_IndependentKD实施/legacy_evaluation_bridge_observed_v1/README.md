# 六个历史端点的实际复评观察桥接

**此工具只生成 DRAFT，由根独立审阅后决定是否接受。它不重新评估模型，不修改 NEW、旧端点或已有回执。**

输入为六个旧 N/C0 的完整 E200 last/EMA 端点、已有候选桥接、2026-09-07 的实际六次完整 dev1469 补评及独立包装器审阅。固定使用 `legacy_diagnostics_snapshot_20260907_161415`，该完成快照含六份 dispatcher 成功资源回执和所有原始补评文件。时间相邻的161459快照是只读收集发生重叠的重复副本，不算另一组实验，生成器不使用它。

`build_observed_bridge.py` 复用旧候选工具解析的正式评价源码顺序、真实 legacy metadata 校验及已接受包装器的旧/新指标比较。它要求每个 checkpoint 的五个总指标实际完全相同、五类 AP 完整、1469 唯一对象行与实际 seen/loader 名单完整、同 seed/arm/配置/原始来源一致，并逐份核对新对象与指标的原字节快照。六份合同和逐图 GT/canvas 也必须一致。缺 seed、源或任何必要执行证据就拒绝生成。

几何只依据 receipt-bound objects 中的实际 canvas/original shape 与 GT 数组；不把配置 imgsz=640 等同于实际输入最长边。首轮 CPU 真数据检查曾因复用旧对象分析器的 max640 断言而失败，原输出保存在 `cpu_tests_attempt1_failure.log`；已移除该依赖。实际六份 eval 的 canvas 为544×672（native rect/pad/stride路径），新桥接不自行重算AP、对象匹配或harm，也不声称 contract 记录了逐batch shape。

canonical 的七份源码从**这六次新执行的实际 eval receipt**取得，并按正式 C1 的 `implementation_files`/receipt emission 顺序核对。实际附加的 `08_evaluator_profile.py` 与 `09_legacy_checkpoint_evaluate.py` 不删去：每个 entry 保存全部九源及其真实 eval receipt，另附独立 wrapper review 和被审源码。附加代码只负责调用冻结原生评价路径、读取/校验输入、采集/序列化既有指标及保存新旧来源；不会把它伪装成历史代码没有执行过。

桥接接受的候选范围是：**同六个固定 checkpoint 在完整 dev1469 上，五个汇总指标对当前已核对的原生评价路径具有观察等价性**。它不证明任意模型或数据上的数学等价，不补造旧时代未保存的原生库源码字节。合同里的 seen/loader 来自本次新执行，entry 显式标出旧 receipt 当时未记录。新逐类/对象证据附带独立来源；旧指标文件不被添字段或覆写。

生成命令（PowerShell，项目根）：

```powershell
& 'D:/Anaconda/envs/KGJ_proj/python.exe' '08_实验日志/2026-09-07_train_IndependentKD实施/legacy_evaluation_bridge_observed_v1/build_observed_bridge.py' --old-manifest '08_实验日志/2026-09-07_train_IndependentKD实施/old_endpoint_manifest_1407.json' --candidate '08_实验日志/2026-09-07_train_IndependentKD实施/evaluation_bridge_candidates_v1/prepared_a1/evaluation_compatibility_candidate.json' --diagnostics '08_实验日志/2026-09-07_train_IndependentKD实施/legacy_diagnostics_snapshot_20260907_161415' --analyzer-root '03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_independent_kd_v2' --wrapper-review '08_实验日志/2026-09-07_train_IndependentKD实施/legacy_endpoint_eval_independent_review_v1/review_receipt.json' --output '<一个新的输出目录>'
```

CPU 验收命令（本目录）：

```powershell
& 'D:/Anaconda/envs/KGJ_proj/python.exe' -m unittest test_observed_bridge -v
```

主要产物为 `evaluation_compatibility_candidate.json`、保留旧 protocol_id 与原臂/seed 的 `manifest_candidate.json`、`summary.json`、每项旧来源独立副本、新实际评估合同/指标/objects/所有九源，以及明确的本地/服务器来源映射。对象只复制一份；`metric_snapshot_independent_copies` 与全局 source_mappings 显式将原 receipt 的每项 metric_snapshot 映射到已确认原字节相同的独立载荷，不改原 receipt 中的路径。原重复快照仍保留在原完成快照内。

生成器没有 ACCEPTED 参数，产物 reviewer=None。它实际调用现有分析器确认六项 DRAFT 均不能升级；不会替根签桥接，也不会自动补签 harm、三seed贡献或论文结论。未来首份 C1 正式评价仍需用原 `prepare_evaluation_bridge.py verify-formal-sources` 核对实际七源顺序和字节。根接受该 observed bridge 后才可以给现有分析器使用对应 accepted 产物；原 DRAFT 保留。

**逐类指标的现行消费接口仍有明确缺口：**NEW 的 `analyze_independent.load_endpoint` 继续从旧 `evaluation_val.json` 读取 `per_class`，而原旧文件没有这些字段。因此，即使本 observed bridge 获得接受，正式分析器也不会自动取得补评中的逐类 AP。后续必须通过显式、独立审阅的 post-hoc 逐类证据入口接入；不得把新逐类值伪造成旧执行时保存的字段，不改原 JSON，也不在本生成器中静默注入指标。此阶段仅完成同 checkpoint 五个旧汇总指标与当前原生评价路径的实际观察桥接。
