# E8 独立副本差异审阅

2026-09-08，loc_stress。对照 `short_screen_draft` 与新 `short_screen_E8_release` 四个入口及三份 YAML，仅做源码/配置读取、AST 语法检查；未运行 GPU、未训练、未计算 hash。

## 初始审阅发现

**初始版本有确定错误，暂不能 READY。** `train_short_screen.py` 两处 `gpu_*_peak_mib` 从 `/2**20` 被 horizon 替换误伤为 `/2**8`，会将 MiB 读数放大 4096 倍；必须恢复 `/2**20`。E8 完成检查本身正确使用 8，但异常文案仍写 exactly20，应同步改为 8。已通知根和实现者，根的批准字节副本须基于修后版本。

其余差异核对结果：

- N/C0/C1 共同 `epochs=8`、`scheduler_horizon_epochs=8`，训练结束、eval completion、driver terminal 检查及 receipt/manifest horizon 都改为 8；身份为 `SHORT_SCREEN_E8_LAST_EMA` / E8 method_id / 独立 E8 admission scope，未冒认为 E200 完成。
- 三臂各自与原 E20 配置相比，训练顶层字段仅 epochs 改变，其余差异是 method/description/protocol/short_screen 元数据。warmup_epochs=3.0、SGD、lr0/lrf=.01、momentum=.937、weight_decay=.0005、B32/nbs64/workers4/imgsz640/seed42，以及数据、匹配门、evidence 和增强字段不变。分类系数分别 0 / .1 / .09227393550836771，定位系数均 0。
- N/C0 原路径与 C1 criterion_factory 的学习源码未另改；warmup 与阶段总长的相对比例因共同采用 E8 而改变，属于明确的独立短程配方，不能称 E20/E200 prefix 等价。
- driver 只对 reference/dispatch/config/admission/native-profile 路径 `.resolve()`；pinned Python 使用 `.absolute()`，保留 venv symlink 调用路径。六阶段固定顺序、预检、资源/receipt 检查和失败退出逻辑继承原已审副本。
- 四入口 AST 均可解析。现有 `E8_CPU_CHECKS.json` 本身不能排除本次发现的错误，因此不得仅据其 PASS 升级结论。

完成状态以随后修复回读补充为准。

## 03:30 最终修复回读

根已修复两处 MiB 换算为 `2**20`，异常文案改为 `exactly8 epochs`。独立回读最终 train 差异：只剩预期的 E8 horizon/身份/完成检查变更，显存单位与原 E20 一致；AST 通过，driver 的 Python `.absolute()` 仍在。未发现其他学习改动。

**最终 READY：新 E8 副本满足本次限定差异审阅。** 三臂共同 E8/8LR、warmup3/系数/数据保留，独立 SHORT_SCREEN 身份和 pinned Python 调用路径正确。仅对根已选择、尚未看新 AP 的 E8 fallback 与实际批准流程有效；不替代资源预算/正式 admission，不授权 E200 或检测增益结论。初始错误记录保留。
