# 静态对象机会 × 缓存代理（已执行 CPU）

不能连接为实际 C1/F selected 覆盖；本表只在同一静态缓存行内交叉统计，不跨 train/dev 或增强帧。

| 数据集/分割 | GT | N错误 | IR修复 | 修复中旧anchor有/无 | 修复中旧ROI有效 | 两旧代理均有 |
|---|---:|---:|---:|---:|---:|---:|
| llvip/train | 2753 | 386 | 303 | 294/9 | 303 | 294 |
| llvip/val | 643 | 239 | 146 | 117/29 | 146 | 117 |
| dronevehicle/train | 15782 | 2208 | 1602 | 1478/124 | 1599 | 1475 |
| dronevehicle/val | 3084 | 698 | 461 | 385/76 | 457 | 382 |

逐状态、逐类结果见 summary.json，逐对象可复核表见 objects.csv。Dev原计数严格复现 LLVIP643/404正确/146修复、Drone3084/2386正确/461修复。

旧anchor是P3/P4的一对一空间归属，现行C门使用全稠密any-candidate；旧ROI还有公共窗口/padding差异。因此“旧anchor无”不能等同当前gate排除，两旧代理均有也不能等同eligible/selected。这份表不是上界、AP贡献、训练修复或不可学比例。

实际selected缺同forward/frame ID、增强GT→原GT映射和每对象q/eligible/selected。现有训练selected使用增强batch全局GT行；静态object_id使用原图局部GT行。只靠同路径或GT行号强连会混淆对象/增强状态。

最小后续接口：固定一个既有32图真实批次，在一次S/T/R前向后共同导出每对象原始/增强双GT ID与坐标、S/R/T各自候选和固定状态、原selector的全过滤链/质量/selected/排名分母；记录同一个forward/frame ID。无需梯度或优化，不换阈值。已有低阈值完整dev输出可作独立错误图谱，但缺全稠密窗口不能恢复原selector。

本次未推理、未GPU、未计算新hash；不依据训练/开发样本差值推过拟合。
