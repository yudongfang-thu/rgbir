# 固定三臂实际终态独立审计

**PASS：LLVIP、seed42、成熟初始化的 FT3 固定矩阵实际完成，端点来源与比较算术闭合。当前 L3-DFL 未超过 N 或同掩码 GT，符合执行前冻结的停止本版本规则。此结论不否定全部 DFL 或定位蒸馏。**

| 臂 | mAP50–95（百分数） | 相对成熟初始化（pp） |
|---|---:|---:|
| 成熟初始化 | 32.878405 | — |
| N | 32.171224 | −0.707181 |
| L3-DFL | 32.160545 | −0.717860 |
| L3-GT | 32.203779 | −0.674626 |

DFL−N 为 −0.0106793605 pp，DFL−GT 为 −0.0432344809 pp，GT−N 为 +0.0325551204 pp。原 fraction、全部五项指标差值和初始差值均从实际回执独立重算，与已接受分析器相同。单 seed 不计算 SD 或显著性。

实际 10 阶段均 clean exit 0、monitor_errors 为空，顺序为固定 8 批校准→三臂 canary→各臂训练/完整 dev 评价，队列耗时 666.8562324 秒。每臂训练 192 batch、96 次 optimizer 尝试、5 次 AMP 跳步、91 次成功更新；EMA 调用 96 次。源和配置绑定 release_v2；18 个已审源文件仍与快照逐字节一致，运行配置只填入校准回执及固定系数。

初始化含检测头 499 个 state tensor，fresh optimizer/EMA，243 个 BN buffer 保持，T/R 隔离。前三十批数据与标签记录跨三臂、训练对 canary 均逐字节相同。15 条保存的诊断记录中，1156 个基础对象、54 次选中的 mask、anchor 与分母完全一致，均满足固定门；第 3、80 批零选中如实保留。673 是全程累计选择次数，不是唯一对象数，也不能代替未保存的 177 批完整 mask 观测。

独立 dev 评价每臂实际 2406 个唯一图、7879 GT。三臂 capture 的 GT 完全相同，actual kwargs、evaluator 版本、canonical/actual loader roster 与成熟参照一致：640/B32/workers4、FP32、conf=.001、IoU=.7、max_det300、rect=True、augment=False。原数据标签是真实 GT；共享 RGB/IR 标签不提供独立物理配准真值。保存的 7218 图记录内所有 GT、预测框及置信度数值有限；三臂有预测的图数为 2400/2399/2398，没有全臂输出失败迹象。这不证明每个未保存的 dense tensor 或权重元素有限。

所有阶段资源记录符合原 lease：样本显存余量至少 2 GiB、项目 RSS 不超过 300 GiB；每次第四卡例外均保留至少两张空卡。三臂训练预算 6144 MiB VRAM / 20480 MiB RSS，NVML 峰 5694 MiB、allocated 4733.0356 MiB、reserved 5162 MiB；RSS 为 17968/17862/18022 MiB。报告 N 训练占卡 0/2/4/5、剩空卡 1/3 的示例与准入回执一致。

**证据边界：**原 eval receipt 的 accepted_endpoint_claim=false 及 profile 的 independent_endpoint_accepted=false 均保留。本审计接受固定比较的执行来源、实际人群/参数和回执算术；没有重新执行 native/capture parity 或从 raw predictions 重算原生 AP，没有下载/重载完整权重。运行源远端副本身份采用既有部署/launcher/运行 manifest 的逐字节回执，并直接复核本地已审快照；未新下载全部远端源码副本。

三臂 results.csv 最后一行是 8 列而表头为 15 列，训练内验证关闭产生的零/缺列不可读作 AP。这是非端点日志格式问题；真正 AP 取自独立完整 dev 评价回执，原 CSV 未改写。GT 与 DFL 共用 λ=0.6900524651944485，但校准实际梯度比例中位数分别为 0.1228374 与 0.1，不能称严格梯度剂量匹配。

三臂均低于成熟初始化，只能说明这个固定微调设置的结果；不能将下降单独归因于 BN、过拟合或教师错位，不能泛化成所有 DFL 无效、物理配准已验证、形状效用已隔离或长训练无效。FINAL_REPORT 的关键数值、十阶段耗时、初始参照和第四卡示例已核；其历史 C0/C1/L2 背景不是本审计新增范围。

本轮仅审阅现有产物：108 项检查通过，258 个声明输入在审阅结束时逐字节未变，没有新 hash、测试、forward 或 GPU 任务。详情见 [EXPERIMENT_AUDIT.json](EXPERIMENT_AUDIT.json)、[RECOMPUTATION.json](RECOMPUTATION.json) 与 [INPUT_MANIFEST.json](INPUT_MANIFEST.json)。审计脚本位于上级 independent_review/audit_final_matrix.py；不覆盖任何原始结果或失败 attempt。
