# 小时级入口最小集成审阅

**READY，限定可进入根任务安排的 canary；未运行 GPU，也不代表新训练已验收。** 2026-09-08，loc_stress。已读实际 `hourly_common.py` / `train_hourly.py` / `evaluate_hourly.py` 及其 pinned builder/criterion/setup 源码。

- 私有 legacy builder 使用独立 globals 副本绑定 `EvidenceCriterion`、tracked dataset、`load_frozen`，不修改在跑 module。新 frozen wrapper 只接受 cfg 指定的 teacher/reference 与对应 subset YAML，传原 full YAML 给既有 frozen provenance 校验；未删除源数据身份检查。
- 实例上挂的 setup/preprocess/step 是显式零参或单 batch 闭包，内部保存原 bound method，符合原 trainer 的调用方式。setup 在原 `_setup_train` 后、首次训练 forward 前比较学生全 state 与 baseline EMA.float，包括 head/buffers，检查 optimizer state 为空、EMA updates=0、EMA 全 state 相同。既有 pinned `resume_training` 在 resume=false 时直接返回，不恢复父 run 状态。实际兼容性由这次运行中的硬检查决定。
- canary 的 criterion.sanity 只在 calls==0 设置 true；之后保留原 first3/every100/shared-gradient observer。只在24次成功更新后停止，不把 AMP 跳步计为成功。没有重新引入每批 fullstats 或大 state 轨迹拷贝。
- 已直接做三处小修：common 固定 `kd_weight=.1`，保留原 `train_independent.build_trainer` 的 C0 行为；source manifest 补本次 RGB/IR YAML、train roster、mapping、auxiliary provenance YAML 及实际 DetectionTrainer 来源；C1 终态读取候选已有 thin/full/fallback 属性并记录 missing 字段名单，没有新增不适用的硬门或造零。
- eval 与 common/训练完成回执/canary 字段已对接；driver 的 `make_job` 使用正确参数与新成功回执文件，完整 dev population/新 E3 identity 检查一致。根已修正原 E8 native contract config 的真实 nested 路径。新的训练身份始终与原完整 dev evaluator identity 分开保存。

原19个 eval 合同样例已用**实际 common 导入**重跑，全部通过：[evaluation_cpu_checks_actual_import.json](evaluation_cpu_checks_actual_import.json)。三份真实配置均通过 common.load_config，四入口 AST 与 common/train/eval 实际导入通过，未导入 Torch：[integration_import_receipt.json](integration_import_receipt.json)。旧 fixture 结果保留；未运行实际评估、checkpoint 读取或新 hash。

生产源码编辑已结束。必要的新初始化、子集流、梯度/有限性与资源证据由受限 canary 产生；不提前标成已通过，不扩展旧 E8/E200 的证据范围。
