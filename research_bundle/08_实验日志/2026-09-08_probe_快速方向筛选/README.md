# 分钟级方向筛选（2026-09-08）

**状态：LLVIP 三臂已完成，当前 L2-box 未超过 N 或同掩码 GT；Drone 分类在跑，F-rel 因剂量上限未准入、没有 AP。旧 E200 在后台继续。**

最新：[LLVIP 阶段结果](early_llvip_evidence/LLVIP_STAGE_RESULT.md) · [定位覆盖与剂量](early_llvip_evidence/README.md) · [Drone 校准](early_llvip_evidence/DRONE_CALIBRATION_RESULT.md) · [参数/BN 交换结果](STATE_SWAP_RESULT.md)。初始 LLVIP 同当前 native 入口复验 mAP32.878405；本轮 N32.171224，成熟模型短训本身仍降分，不能把其短程排序当作已验证的 E200 排名代理。

独立后续：[LLVIP C0 置信度两臂冻结计划](CONFIDENCE_FOLLOWUP_PLAN.md)，用原 C0/λ0.1 检验单类前景置信度，不改变已结束 L2 和在跑 Drone 配置。下文七臂是首批冻结矩阵，两臂后续有独立 scope、源码和输出目录。

## 冻结目的与范围

用户要求立即推进、快速判断可做方向。本轮使用成熟模型、2,048 对自然分层 train 子集、seed42、3 epochs（192 batch），完整 dev 独立评估 last/EMA。不是正式 E200，也不以一个 seed 宣称论文增益。

上一轮 FT3 的纯监督臂也下降约 0.80 pp。先用无需训练的 parameter_only / buffer_only 交换评估区分参数与 BN 等 buffers 变化。**在读取交换 AP 前固定本轮新训练协议**：SGD lr=0.0001 恒定、warmup=0、冻结全部 BN running statistics（affine 参数可学）、fresh optimizer/EMA；其余数据流、增强、B32/nbs64/workers4/640 沿用。各方向有本轮匹配 N，不复用上一轮掉点 N。

## 最小矩阵

| 数据集 | 臂 | 要回答的问题 |
|---|---|---|
| Drone | N、C1 | 修正短训协议后的纯监督及既有类别控制 |
| Drone | C2 | 同选择/同内容，改为基础集合按类别归一，是否改善类别不平衡 |
| Drone | F-rel | 同对象选择，局部 ROI 的空间特征关系是否比现有类别内容更有信号 |
| LLVIP | N、L2-box、L2-GT | 对象坐标的教师定位信息是否超过同掩码 GT 重监督 |

只跑这七个短训，不自动扩展 E200，不根据本轮 AP 改门槛、系数、子集或训练步数。LLVIP 定位先行；Drone 三种内容随后。原 L1 的物理配准准入仍阻塞，L2 是明确的新对象坐标方法，不能把 GT 关联当物理配准证明。

## 剂量与工程验收

固定各数据集训练流前 8 batch 校准、每批恢复初始全部参数和 buffers，不更新 optimizer。Θ 为检测头 P3/P4 输入来源两模块；C2/F-rel λ 匹配 B×既有 λC1×C1 的梯度范数中位数；L2 λ=min(1,0.1×median(native norm/(B×L2 unit norm)))。至少 4/8 批有限非零；C2/F-rel 未裁剪 λ 须在 (0,1]，不以 epsilon 或换 batch 补救。L2-GT 共用 L2 λ 和对象/anchor/掩码/分母。

这是探索性 8 batch 校准，不能冒充正式 64 batch 准入。新路径先通过小样例和至少 24 成功更新 canary，记录实际 GPU/RSS 后预约短训。每数据集前 30 batch 标签/顺序应一致。全部 GPU 任务使用现有全局 lease；保留旧结果、失败 attempt、权重。不计算新文件哈希，不访问 test。

## 产物位置

- 本地：本目录源码、协议、小体积执行回执及结果。
- 94：`/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_direction_screen_20260908/`。
- LLVIP 子集：`subset_llvip_v1/`；Drone 复用 `rgbir_hourly_screen_20260908/subset_v1/`。
- 方法源码：`newentry/release/`；状态交换：本目录 `evaluate_state_swap.py`。

## 解释边界与下一步

短训练能回答局部优化信号和相对趋势，尚未证明它预测从头 E200 排名。正结果需要匹配基线、多 seed 与四臂归因；负结果只约束此版本和此短训协议。各方向按完整短训后的 AP、逐类变化、控制比较及梯度有效性给出继续/暂缓判断。
