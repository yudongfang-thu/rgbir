# 同缓存内部的对象机会与代理覆盖：执行前界定

本次只读既有 baseline `remote_exports/{llvip,dronevehicle}_full_attempt1/objects.jsonl`，执行原 `analyze_baseline_probe.object_view` 的固定 conf=.25、hit IoU=.5、pair IoU=.5 状态定义。train 与 val 分开，背景窗不进入 GT 对象分母。原 frame 的 object_id 必须唯一，图像路径/原 GT 行自洽；以同一 JSON 行内 N42/T42 关联统计，不连接增强训练行。

输出 N 错误桶、IR 候选修复/潜在损伤，以及这些相同行中已有 `anchor_has_reference_candidate` 与 N/T P3/P4 `regions.valid` 的交叉计数。这两个字段仅是**旧静态 probe 的代理**：前者为 P3/P4 一对一空间候选归属，而现行 C 门是稠密 any-candidate；后者有旧公共RGB窗口与padding处理。不能当现行 C1/F 必要/充分条件或上下界，更不能当实际 selected、AP/oracle覆盖。Drone 静态 N42 也不同于当前选择参考R。

已知 dev 总量/正确/IR修复应分别复现 LLVIP643/404/146、Drone3084/2386/461；这些校验仅确认旧对象规则复用。记录中 `actual_selected` 始终 null，并保存无法严格 join 的原因。新结果不修改阈值、mask、训练或 AP；无 GPU、hash、模型重推理。

输出 `summary.json`、`objects.csv`、`README.md` 和执行源副本。若缺字段/同 ID 重复/旧总量不闭合则失败退出，不替换为零。此次执行是现有静态证据的内部交叉诊断，不给训练集过拟合解释。
