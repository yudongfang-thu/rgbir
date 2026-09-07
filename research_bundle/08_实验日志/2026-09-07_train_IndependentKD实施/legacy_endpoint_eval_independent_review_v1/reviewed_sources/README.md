# 旧 N/C0 的最小诊断补评

状态：**CPU_IMPLEMENTED / NOT_RUN / AWAITING_INDEPENDENT_REVIEW**。六个已完成 E200 的旧 last checkpoint，只做一次完整 Drone dev1469 evidence 评价，不启动训练。所有输出使用新 artifacts attempt；本目录没有运行 GPU，也没有签接受。

`legacy_checkpoint_evaluate.py` 是 LOG 目录的薄入口。它复用冻结 release_gpu5 的 `make_evidence_validator`、`capture_contract`、`verify_population` 和 `evaluator_profile.metric_record`，不自写 AP、NMS 或匹配。推理仍使用已实测的 batch32/workers4/640、FP32、rect、native conf=.001/IoU=.7/max_det300 路径；实际 callback kwargs 必须与已通过 profile 的绑定完全相同。

## 真实旧身份与新执行的区别

运行前用冻结的 legacy receipt loader 验证原 completion、train/eval receipt、实际 cfg/metric/roster 快照及 seed/arm，要求原 E200、last/EMA、Drone5类、全dev1469。再验证物理 checkpoint 确实位于原 run 的 weights/last.pt，读取前后记录 stat（不计算哈希），原小证据通过独立字节副本检查未变化。

旧 YAML 常保留 seed42 默认值，实际启动 seed 来自原训练/评价 receipt。新 `evaluation_config.yaml` 沿用全部原 recipe，只显式补充本次实际 seed、原 arm(weight0/paired)、source=paired、expected_val_images1469 和诊断类型；这些是本次评价的配置元数据，绝不声称旧 YAML 当时包含这些字段。原 train/eval config 本体保存在 origin_manifest 与原字节输入副本中。

新目录不创建根 `completion_receipt.json` 或 `run_evidence`。旧训练 evidence 只复制到有原路径索引的 `origin_evidence/`，不伪造第二次 E200 训练。新 receipt 的 run_kind=eval，明确指向原 training run/checkpoint，并说明 new_training_receipt_created=false。

## 输出与失败

每次新目录包含完整原生逐类 AP50/AP75/mAP50–95、五个总指标、`predictions/objects.jsonl.gz`、实际 contract（roster、loader、seen、kwargs）、`evaluation_val_roster.txt`、原始 observed 结果、新旧五指标精确比较，以及绑定配置/源/metric/objects 的新 `eval_evidence/run_receipt.json`。

任一旧指标与新值不完全相同，保留 `evaluation_observed.json`、objects、contract、comparison 与 failure_receipt，拒绝 `reevaluation_receipt` 完成；不覆盖旧数值，不扩大容差。资源或实现问题也留新失败 attempt，不自动覆盖或重启。参数/指标/对象核对成功但写 receipt 失败时，已有产物仍保留，未完成的 receipt 不可用。

原生库历史源码字节没有保存在旧 receipt 的局限仍保留。本次新评价绑定当前全部实际调用的 evaluator/helper/wrapper 源；它的来源集合包含包装器，不能伪装成旧 canonical 七源。新对象产物可由现有 `object_error_analysis.load_evaluation` 的原字节/配置检查读取；`analyze_independent.load_endpoint` 要求根旧训练资料及相同 evaluator 源，**不能直接把本诊断目录当成它已经接受的完整新训练端点**。总指标桥接和旧训练来源的分析 manifest 仍需独立完成。

## 资源与队列

`prepare_queue.py` 只读九份旧 snapshot，选 N/C0 的六份，生成配置和队列候选；不调用 subprocess、SSH、screen 或 GPU。原 random 不重复评价。队列顺序是 seed42、0、123，每 seed N 后 C0；由冻结 dispatcher 串行运行，所有任务仍共享同一个全局 lease，GPU 动态选择。

资源来自真实 `admission_queue_attempt5/evaluator_profile_a2_resource_profile.json`，实测 NVML1370MiB/RSS6260MiB。沿用已审阅 formal_campaign_gpu5_attempt2 的评价预约规则，分别加256/4096MiB，得到 **1626MiB VRAM、10356MiB RSS**。这不是新的实测；实际 source/config/profile binding 仍由 dispatcher 和包装器运行前验证。旧 config 无硬编码物理 GPU，2GiB余量、70%、共享任务数和总RSS规则均保留，不自己造 lease 池。

GPU 执行命令（待独立审阅、部署并由根启动，不由准备脚本执行）：

```text
screen -dmS ikdv2_legacy_diag_20260907 <pinned-python> <release_gpu5>/resource_dispatch.py --manifest <code-dir>/candidate_bundle_v1/queue_candidate.json --output <BASE>/legacy_diagnostics_v1/dispatch_attempt1 --repo <actual-project-repo>
```

包装器 CLI：

```text
<pinned-python> legacy_checkpoint_evaluate.py --release <release_gpu5> --legacy-run <original-run> --config <derived-evaluation.yaml> --output <new-data-artifact-attempt> --evaluation-profile <actual-resource-profile.json> --review-receipt <independent-review.json> --arm N|C0 --seed 0|42|123
```

运行必须有实际全局 lease 和独立接受回执；缺失就拒绝。review schema=`rgbir-legacy-evaluation-wrapper-review-v1`，status=`ACCEPTED`，有 reviewer，`source_files` 精确四项，relative 为 `legacy_checkpoint_evaluate.py`、`prepare_queue.py`、`test_legacy_endpoint_eval.py`、`README.md`，每项 accepted_copy 指独立源码副本（与本文件不是同一路径且原字节相同）。本包不生成 review，也不能给自己签接受。

CPU 测试有真实旧 snapshot/成功 profile 的只读测试，以及明确标为 synthetic 的临时目录/回执/对象夹具；不称为真实新 AP 或 GPU 验收。正式启动前仍需根安排独立源码审阅，并重新检查实际资源可用情况。C1 已启动的训练不受本准备步骤阻塞。
