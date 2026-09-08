# 实际 Drone IR42 导出独立验收

**ACCEPTED：历史 IR42 教师完整 dev 原生导出可用于后续限定 CPU 诊断；未接受新的蒸馏增益或跨模态几何结论。**

独立读取 `evidence_1401` 两个 attempt 的全部缓存、population、冻结 spec、实际合同、模型身份、指标和队列资源回执，复算入口及小回执为 `independent_review/review_capture_actual.py`、`independent_review/ACTUAL_CAPTURE_RECEIPT.json`。无新 GPU、forward、NMS、checkpoint 加载或 hash。

| 实际范围 | 图像 | IR GT | 预测行 | 结果 |
|---|---:|---:|---:|---|
| 固定首 32 canary | 32 | 514 | 932 | native/capture 总体及各类指标 exact，实际 kwargs、loader 顺序、模型身份一致 |
| 完整 dev capture | 1469 | 24490 | 61969 | 实际 roster 唯一完整、逐图 GT 数和五类计数闭合 |

完整 IR 五类 GT 为 20588/918/1470/789/725，两幅空 GT 图保留。canary 447/24/30/13/0，只报告出现的四类 AP，模型与数据仍是完整五类。两次输入 checkpoint path/stat 与冻结 spec 一致；历史 seed42/E200/infrared 身份闭合。loader 合同记录逐图 normalized GT 与原 processed 标签 exact；本地缓存独立核对完整人口、类数和两次共用图的 GT 数组、canvas/original shape exact。

实际 FP32、640/B32/workers4/rect、conf .001、NMS IoU .7、max_det 300 与原生合同一致。完整实际 canvas 为 544×672、原图 512×640。所有坐标、置信度和类别合法；保留 19 条越出 canvas 的原生预测框，不另行 clamp。完整 `native_capture_exact=false` 表示未重复第二次 full native 评价，符合“canary 双路径、full 单 capture”的冻结协议，不能误记成数值失败。

完整指标原始单位是 fraction：AP50 **0.8083569915490951**，AP75 **0.7119863600014217**，mAP50–95 **0.5963198259876477**，显示为 **80.835699%、71.198636%、59.631983%**。五类 AP 宏均值与总量最大误差 1.11e-16；canary 四类宏均值 exact。本次未重新计算 AP。

原 RGB N42 缓存与本 T42 的原生版本、全部 effective kwargs、五类顺序、1469 个唯一同名帧和逐帧 canvas/original shape 均闭合。N 是 `full_weight0_s42_attempt1`，T 是历史 `infrared_seed42_native_b32a2`，不能混称旧 RGB reference 或同一训练配方对照。RGB 22462 GT、IR 24490 GT 是独立标签；当前只接受评价接口与帧级范围可对照，逐实例 N/T 联合分析还须接受其显式帧绑定及同类 GT 关联规则，不能将 AP 差视为 KD 收益。

队列两阶段均 COMPLETED、exit 0、monitor_errors 为空，使用原 global lease，项目占卡为 2/4/5，新增进程只用卡 4。canary NVML 峰值 1370 MiB，full 预约 1882 MiB；full 实测同为 1370 MiB。两阶段进程树 RSS 峰值 1560/4017 MiB，均低于 12288 MiB 预约。保留采样中的整卡最小空闲为 15513/14661 MiB，项目 RSS 最大 145335 MiB；采样总显存占比也低于 70%，原 guard 无越界记录。此结论依据现有 guard 和采样，不伪称独立连续监测了所有瞬间或重建全部全局预约账本。

队列 wall 为 **212.534579 秒（3.542243 分钟）**；producer 内计时 canary **24.466730 秒**、full **28.691358 秒**。这些区间含义不同，不能用两项内计时之和替代队列耗时，也不能将其外推为训练吞吐。
