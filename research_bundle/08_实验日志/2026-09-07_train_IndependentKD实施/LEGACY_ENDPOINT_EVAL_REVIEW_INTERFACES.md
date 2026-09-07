# 六个 legacy N/C0 last 端点补采：独立审阅接口

**当前是候选实现前的接口核对，不是入队接受。补采应复用已实测的单模型评价路径，并新建真实评估证据；旧训练身份和原结果保持不变。**

日期：2026-09-07；审阅者 `/root/review_c1_spec`。候选作者 `/root/review_l1_spec`。已读 `EVALUATION_BRIDGE_PREPARATION.md`、`prepare_evaluation_bridge.py`、`evaluation_bridge_candidates_v1/prepared_a1` 的候选汇总及 manifest，以及冻结 `evaluate_independent.py`、`evaluator_profile.py`、`analyze_independent.py` 和 LOG 的 `object_error_analysis.py`。任务所称 protocol analyzer 在本地的实际文件名是 `analyze_independent.py`。

## 必须核对的实际接口

1. **六份真实输入。** 固定原 `weight0/paired × seed0/42/123`，只读原 run 的 `weights/last.pt`。原 completion、训练/评估 receipt 与其配置/指标快照必须由已有 legacy loader 校验。目录 `C0` 是旧 C 的 seed0，方法规范名 C0 是整条旧分类分支，两者不可混同。实际 seed 从 receipt/metric 取得，不能把旧共享 YAML 的默认 seed 当成每次执行 seed。
2. **本次配置身份。** 新派生配置可记录真实执行 seed、评价所需 arm/source 与 expected_val_images 等元数据，但必须保留和比对原生评价 recipe。新 receipt 必须明说本次对历史 checkpoint 做诊断评价。原训练文件只读或原字节复制进带路径索引的 origin_evidence；不生成新的 E200 训练回执，不假称历史已保存未保存的 library 源码。
3. **固定推理和指标路径。** 一次完整 evidence 前向，使用冻结 `make_evidence_validator`、`capture_contract`、`verify_population`、`metric_record` 与同一原生 DetectionValidator/NMS/AP。原 kwargs 为 val、640、batch32、workers4、device0、plots=False、save_json=False、verbose=False；实际 runtime 仍记录 FP32/rect/conf/IoU/max_det/single_cls 等，不能只抄配置中期望值。
4. **真实完整产物。** 五个总指标、五类 AP、1469 唯一图像的对象行、真实 loader 顺序/seen/GT/输入尺寸/原始尺寸，均写入新的 attempt。指标、objects 原字节和 contract/config/roster/实际评价源码分别进入本次 eval receipt 的正确快照字段。所有输出不得位于旧 run 内，不覆盖已有 attempt。
5. **profile 与共享 lease。** 使用真实 release_gpu5 的成功 evaluation_profile，核对其实际 binding、配置评价身份、数据 YAML/名单和源码。新薄包装器必须实际调用被该 profile 证明的单模型路径；新增包装源码单独受独立审阅副本绑定。统一 dispatcher 的 stage=evaluation、measured_reservation 和动态全局 lease 不绕过；评价 VRAM 预约实测峰值向上取整 +256 MiB，RSS +4096 MiB 且至少8192 MiB，不直接继承 bootstrap cap。Python venv 入口保留符号链接语义。
6. **新旧差异必须显式。** 本次五个总指标逐项对照同 checkpoint 旧结果并保留差值。任何差异不覆盖旧结果、不把新 objects 冒充旧执行。差异的技术调查/是否可桥接与本次前向是否成功分开记录。

## 两种消费接口不能混同

`object_error_analysis.load_evaluation` 接受本次 metric 及其同级 `eval_evidence`，不要求伪造根训练目录。它要求 receipt/metric/config 的 dataset、seed、arm、source、method_id、checkpoint、endpoint 一致；metric 和 objects 对应 metric_snapshots，contract 对应 config 快照；E200/640、expected_nc、完整1469名单/seen/对象记录均明确。后续配对还比较同 seed、roster、实际 kwargs、evaluator_identity 以及每图 GT 数组顺序和 canvas。缺失分组元数据保持 UNKNOWN。

`analyze_independent.load_endpoint` 当前仍走旧固定 run 布局，读取根 `run_evidence`/completion，并将实际评价来源内容加入比较合同。新薄适配器的额外真实源码不能被删掉来伪装成原 canonical 七源，因此诊断目录不自动成为这个总指标分析器的合格完整端点。补采先补足逐类/对象原始证据；其与旧训练端点的正式桥接或 manifest 入口必须另有清晰、独立验收的来源映射。原六份 DRAFT bridge 不因补采工具存在而变 ACCEPTED。

本审阅没有启动 GPU、执行训练或改动冻结 NEW。候选稳定后再作源码与必要 CPU 已知真值测试，最终回执明确可否进入统一队列及接受范围。

现有候选工具 CPU 检查已独立复跑：`python -m unittest test_prepare_evaluation_bridge -v`，6/6 通过（0.384s）。它们验证 probe/源码顺序/参数及 DRAFT 不被分析器升级；不代表待编写补采包装器已经接受。
