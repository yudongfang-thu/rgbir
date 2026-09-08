# LLVIP 单批真实DFL信息读出（2026-09-08）

**已完成：唯一一批32图/80GT导出317份真实DFL，生产21.43秒、队列61.70秒，零训练。79共同合法位置教师GT CE均值低于学生；11个既有定位机会亦如此，但尚未隔离分布形状相对更准均值的额外价值。**

[完整结论与方向判断](FINAL_REPORT.md) · [逐对象读出解释](cpu_analysis/RESULT_READOUT.md) · [独立审计](independent_review/EXPERIMENT_AUDIT.md) · [下一项目标接口实现](NEXT_TARGET_INTERFACE_PLAN.md)

执行合同见[RAW_DFL_INFORMATION_PLAN.md](../2026-09-08_probe_定位学习位置与目标/RAW_DFL_INFORMATION_PLAN.md)。保留原批80GT、历史R候选与native匹配anchor；新forward验证图/标签/流身份，不与历史拼成同一次前向。所有旧L1/L2门和任务保持原身份。

本轮只判断实际分布内容与本模型本anchor的GT读出。熵/方差/非零KL不能当效用；不跨anchor同bin直接蒸馏，不重映射/截断概率，不扩大batch或追加训练。已有产物可复用则不启动GPU；缺失时本批自身作为新路径的实测探针，不另跑重复full阶段。

本目录保存源码、固定输入、独立验收及原始小分布；失败attempt保留。原global lease动态选择GPU0，援引四卡例外后仍有两张空卡；执行与资源均正常。两个收集快照来自同一attempt，producer文件逐字节一致，早期RUNNING状态不代表还需重跑。真实原始与CPU读出全量独立通过，不将该范围升级成KD有效或旧L1准入。

原执行位于94的`/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_dfl_information_20260908/attempt1`，小证据镜像为同根`review_v1`。阶段继续发布到既有GitHub分支`research/full-evidence-20260906`。下一步直接实现标签对象坐标中的分布目标接口，原L1几何BLOCKED保持；不再增加完整推理或机会计数。
