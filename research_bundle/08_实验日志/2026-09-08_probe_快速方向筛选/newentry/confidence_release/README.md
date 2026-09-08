# LLVIP 原 C0 置信度 FT3 实现与接口

只准备独立 N/C0 两臂；原 direction release 不修改。scope=`LLVIP_CONFIDENCE_FT3`，endpoint=`LLVIP_CONFIDENCE_FT3_LAST_EMA`。固定本轮新 N 作匹配对照，不借 L2 的 N 或其他 FT 协议冒充本轮完成证据。

固定现有 LLVIP 2048 图子集、64 batch/epoch、B32/nbs64/workers4/640、seed42、SGD/AMP、3轮独立常数 lr=1e-4、warmup=0。共同初始化为已完成 visible42 last/EMA；teacher=infrared42、reference=visible42。fresh optimizer/EMA，BN running buffers 冻结且 BN affine 可训练。原增强/evidence/双 GT mapping 不变；N λ=0，C0 λ=0.1，不校准新系数、不叠加 L/C1 或特征项。

`confidence_criterion.make_type(runtime.ORIGINAL_CRITERION)` 只继承原 OEv1 `EvidenceCriterion`，设置已有 `sanity` 观察开关后调用原 `__call__`。原 `object_evidence_loss`、paired selector、seed+calls、温度/clip8、标量 `native.sum()+B*weight*kd` 全保留。N 通过原 weight0 同计算路径，原检查验证零权重 loss/score gradient exact。每个新 canary 启用原 sanity，正式三轮只首批启用，后续按原前三批/每100批日志频率。

## CLI 与队列契约

- 配置：`configs/llvip_N_s42_FT3.yaml`、`configs/llvip_C0_s42_FT3.yaml`。
- Canary：`train_confidence.py --reference-dir PINNED --config CFG --output NEW_CANARY --canary`。
- 三轮训练：`train_confidence.py --reference-dir PINNED --config CFG --output NEW_RUN --canary-receipt CORRESPONDING_CANARY/canary.json`。
- Full dev：`evaluate_confidence.py --reference-dir PINNED --config CFG --checkpoint RUN/weights/last.pt --output NEW_EVAL --native-config configs/llvip_native_evaluation.yaml`。LLVIP 不传 Drone native-profile-binding。

每臂必须是新路径 canary 成功更新≥24、匹配配置副本，再运行三轮192批。训练检查完整 student state（含 head）与原 EMA 初始化 exact、fresh optimizer/EMA、aux isolation、实际 2048/2406/64/B32/workers4、BN buffers 前后 exact；canary 的非 N 还要求 selected 与非零 KD score gradient。`sample_stream.jsonl` 留前30批双 GT/图像路径/批号，由统一队列比较两臂。资源均由根队列原 globallease 准入，无新资源池。

| 产物 | status | 关键字段 |
|---|---|---|
| `canary.json` | `CONFIDENCE_CANARY_COMPLETED` | 新 scope/endpoint，arm，successful_updates，resources，框架峰值，BN evidence |
| `completion_receipt.json` | `CONFIDENCE_TRAINING_COMPLETED` | last_epoch=epochs_configured=3、batches192、updates/AMP/EMA、checkpoint stat |
| `confidence_failure.json` | `CONFIDENCE_TRAINING_FAILED` | 新 scope、error、traceback、耗时 |
| `direction_evaluation_receipt.json` | `DIRECTION_EVALUATION_COMPLETED` | 保留通用 eval 文件/status，scope/endpoint 为本置信度协议 |
| `direction_evaluation_failure.json` | `DIRECTION_EVALUATION_FAILED` | 同上，scope 为本置信度协议 |

训练 `config_copy` 指向 `confidence_config.yaml`；eval 同时保存该配置和 `training_completion_copy.json`。评估回执保留总体 AP50/AP75/mAP50_95/precision/recall、per_class(class_id/name/3AP)、metric_units=fraction_0_to_1，full_dev_images=observed_images=2406、full_dev_gt_objects=gt_objects_captured=7879，training_model/configuration/completion 与完整 native projection。另含 `training_subset_identity` 的三条路径（student_data_yaml、privileged_data_yaml、paired_train_mapping）和 expected_train_images=2048，供本轮 N/C0 对照身份检查。新 scope 不会被原 direction analyzer 接受。

## 已做的检查与局限

`CPU_CHECKS_attempt1.json` 六组通过：配置/evidence 与原 LLVIP 子集一致；错误系数/scope/dataset/L 拒绝；真实保存的原 criterion AST 加明确 toy evidence kernel 检查继承调用、零/非零标量和原梯度检查；首批 sanity/private globals；BN running freeze/affine梯度；eval scope/配置/API/source 编译。toy kernel 不冒充原完整检测损失真值；实际原 OEv1 计算路径由新 canary 验证。

只做 CPU/源码准备，未读取新 AP、未启动 GPU、未计算新 hash。单 seed 短筛不构成正式增益、论文或 E200 完成；不根据当前方向 AP 改参数。部署仍由根统一队列按资源与先行队列完成条件执行。
