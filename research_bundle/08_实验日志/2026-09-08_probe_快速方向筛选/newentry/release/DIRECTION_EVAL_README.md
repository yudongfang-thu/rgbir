# 双数据集方向筛选独立评价入口

**入口与 11 项 CPU 合同检查就绪；尚未在此任务运行 GPU、评价新 AP 或接受真实端点。**

CLI：

```text
evaluate_direction.py --reference-dir <pinned independent> --config <实际训练配置.yaml> --checkpoint <run/weights/last.pt> --output <新评价目录> --native-config <原完整dev配置.yaml> [--native-profile-binding <Drone既有绑定.json>]
```

读取对应 run 的 `completion_receipt.json`；要求 `DIRECTION_TRAINING_COMPLETED`、`DIRECTION_FT3_BNFROZEN`、固定 `DIRECTION_FT3_BNFROZEN_LAST_EMA`、seed42、3轮/192批、实际 BN running buffers unchanged、模型/T/R/三种系数和 checkpoint stat 一致。`config_copy` 必须是该 run 的 `direction_config.yaml`，并与 CLI 配置字节一致。只读取 last/EMA，不读取 best、不修改权重，不调用原 E200 run/load/publish。

Drone 使用完整 dev1469 图/22462 GT/5 类并验证原已接受 native profile；LLVIP 使用完整 dev2406 图/7879 GT/1 类，创建 `actual_native_evaluation_profile.json`，明确 `prior_binding_validated=false`、`accepted_endpoint_claim=false` 和 `native_vs_capture_parity_rerun=false`，不把 Drone-only 绑定冒充 LLVIP 验收。两数据集都要求 subset YAML 的完整 dev roster 逐项等于 full YAML，full YAML 与显式 auxiliary identity 一致。

实际 `YOLO.val` 使用 pinned 原生指标及原只读 object capture，imgsz640/batch32/workers4/quantize=None（FP32）、conf.001/iou.7/maxdet300/rectTrue，其余固定 kwargs 在实际回调核验。回调核验实际 loader 样本和推理前 GT 总数；结束后核验每图 object capture 的样本/GT 总数。全部类别及三 AP 宏均值必须闭合，指标保留 fraction[0,1]，不换算为增益或 TIDE oracle。不会访问不存在的 `validator.model`。

成功输出 `direction_evaluation_receipt.json`，status `DIRECTION_EVALUATION_COMPLETED`，scope/endpoint 同训练。包含实际 observed_images/gt_objects_captured、完整 dev 总量、原生各类与总体指标、实际 profile、training projection、checkpoint stat 和 lease resource。另存合同、真实源码字节副本及小输入副本；失败独立写 failure，不覆盖旧产物。

[CPU attempt2](DIRECTION_EVAL_CPU_attempt2.json) 是当前版：11/11 PASS，未导入 torch。覆盖两数据集固定臂、配置 schedule、checkpoint/完成身份、profile 适用范围、完整 roster 漂移、单位/有限性/类映射/宏均值与 best 拒绝。当前源码的 CLI --help 也通过。原 attempt1 回执保留；attempt2 补齐与实际训练的 config_copy、BN 证据和 kd_coefficient 对接。
