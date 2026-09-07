# 自然 64 批选择诊断准备

**有效attempt2完成：LLVIP/Drone各2批技术短测和64批真实自然流，原记录字段逐批exact；L待认证候选207/602，有候选批62/63。真实结果已获[独立限定接受](independent_review/REAL_RESULTS_EXPERIMENT_AUDIT.json)。结果身份为 `UNVERIFIED_GEOMETRY_DIAGNOSTIC`，不提供L1/L_GT或校准准入。**

[完整结果](completed_readout_attempt2/README.md) · [限定解读](completed_readout_attempt2/FINDINGS.md) · [129文件逐字节采集回执](collection_receipt.json) · [启动失败及最小修复](ATTEMPT1_STARTUP_FAILURE.md)。SCP嵌套目录超Windows路径长度导致采集部分失败，使用tar流和extended路径补齐缺失，已有文件字节核对未覆盖；远端诊断结果未改变。

目的：在既有原自然流上补齐真实冻结 C0/C1 与定位选择统计。代码只新增本目录，旧 trainer/选择器、配置和 coverage 产物均未修改。协议见 `PROTOCOL.md`，执行入口为 `diagnose_natural_flow.py`；根任务的 `run_campaign.py` 负责已有 global lease 排程，先 LLVIP 后 Drone。

94 上共同根为 `/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_independent_kd_v2_20260907`。入口参数如下，输出必须为新的数据盘目录：

```text
diagnose_natural_flow.py
  --release <共同根>/release_gpu5
  --config <共同根>/configs_draft_v1/llvip_C1.yaml
  --coverage-dir <共同根>/coverage_llvip_attempt1
  --output <新数据盘artifact目录>
  --batches 2
```

Drone 对应 `drone_C1.yaml`、`coverage_drone_attempt1`；完整诊断仅把 batches 改为 64。脚本不自行预约 GPU，不允许无绑定 lease 执行。两批预约值由 runner 提供，属于资源上限；需实际运行记录 NVML 显存峰值与完整进程树 RSS，再按实测和余量预约 64 批。不能以本地 CPU 检查替代该短测。

真实接口：`coverage_probe.build_natural_loader` 和 `summarize_batch`；`runtime.to_device`；`runtime.legacy.load_frozen/raw_prediction`；整 batch `selection_adapter.build_classification_selection`；`task_conditional_reference/localization_loss.build_localization_selection`。teacher/reference 来自 cfg 原固定端点；特别是 Drone reference 为旧 formal_native RGB42，不等于近期 raw 读出里的新 N42。模型 args、路径、names 和 source 副本随实际运行保存。

完整旧 trace 含 64×32 源图与增强/双标签记录，两数据集均已本地检查可读。loader 使用 seed20260907、B32/workers4/imgsz640；每批 JSON 语义字段 exact 比较后才前向。数据元信息只按原 JSON 序列化比较，避免 names 的整数键与保存后的字符串键被误当变化；数值没有加容差。未保存的像素字节不宣称 exact。整批定位为主结果，按图重复只补 gate 图/组，真实每批都校验 counts、base/selected IDs、anchor、RGB DFL距离、quality mask exact。分类不拆批，C0/C1 共用真实选择集。

输出：`natural_batches.jsonl`、`selection_batches.jsonl`、`summary.json`、输入/模型/source 身份与失败回执。summary 为 `status=COMPLETED`，另存诊断身份；资源接口为原 `resources.per_gpu_peak_vram_mib`、`resources.peak_rss_mib` 和 Torch allocated/reserved peak。定位保留每 gate 的对象数、唯一源图和来源组，base/selected 对象的类别与支持域；解析 gradient proxy/CE/entropy 不发布。本诊断没有 backward、模型更新、校准或 AP。

CPU 结果见 `cpu_checks.json`：原选择器单图/多图含空图与全局 GT index 恢复、confidence-first、C0原stats一致与学生值不参与选择、两组旧64×32 trace可读和增强篡改拒绝、metadata类型兼容与worker漂移拒绝，共6项通过。运行环境 PyTorch1.8 CPU，非94 pinned2.10；只证明这些合约测试。`test_cpu.py` 使用独占新回执，避免覆盖旧检查。

来源分组复用冻结 `diagnose_opportunities.source_groups`：Drone train须有来源TSV，LLVIP读取fit.tsv或使用sequence prefix fallback。它们是数据来源分层，不是标定组。geometry=None/false的全放行诊断可以改变正式geometry约束下的anchor，故其最终selected_count不保证是严格数学上界。实际加载、完整trace、资源测量与独立审阅仍由根任务组织；本目录不制造 readiness、geometry、lambda 或训练receipt。
