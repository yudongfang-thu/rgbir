# 对象坐标分布目标接口（2026-09-08）

**三臂已完成：N／教师DFL／同掩码GT的mAP50–95为32.171224／32.160545／32.203779；DFL−N为−0.010679pp、DFL−GT为−0.043234pp。原lease队列实际11分6.86秒，按冻结规则结束当前L3+成熟初始化FT3版本，不进入长训。**

详细设置、原值、实测耗时、失败修复和结论边界见 [FINAL_REPORT.md](FINAL_REPORT.md)，配对比较见[接受的分析器实际输出](analysis/output_attempt1/README.md)。单seed短筛不等于正式增益或全部DFL方向无效；三臂都低于32.878405的成熟初始化。

固定依据为[上一阶段的下一项计划](../2026-09-08_probe_DFL真实信息读出/NEXT_TARGET_INTERFACE_PLAN.md)。原L1几何BLOCKED保留，原DFL读出不重跑；不把标签对象坐标定义当物理配准。固定80GT、历史R学习anchor，教师同R索引为主输入、原native教师anchor仅作压力检查。

CPU运输耗时0.100秒：同R位置79个可用对象全部支持且均退化为恒等映射；教师独立native位置78个中63个因支撑越界拒绝。这证明固定anchor接口可算，不证明物理配准或蒸馏有效。[完整协议](SHORT_SCREEN_PROTOCOL.md)在新校准/AP前冻结。

短筛固定LLVIP自然2048图、3轮、seed42、共同成熟初始化、BN统计冻结；每臂192个batch、96次optimizer尝试/91次成功/5次AMP skip，完整dev2406图/7879GT评价。三臂各24次成功更新的canary先通过，才训练/评估。校准7/8批有效，λDFL=λGT=0.6900524651944485；全程没有据AP修改门控、剂量、矩阵或预算。

首个GPU attempt在第二批校准因跨批计算图残留触发预定显存门，未开始训练。原失败与源码保留；[v2修复](CALIBRATION_LIFETIME_V2.md)仅释放残留引用，36项pinned CPU检查及独立源码审阅通过后，已在新attempt2重跑原固定8批，显存不再跨批增长。

服务器执行：`/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_object_dfl_screen_20260908/attempt2`；执行源码`release_v2`。只用原全局lease，动态单卡、全部资源门保留。终态副本为`training_evidence_final/`；1716/canaries为同一attempt2的早期快照，不能据旧RUNNING重跑。初始化参照与独立复核见`initial_endpoint_reference/`。正式L1物理几何准入仍BLOCKED。

源码、实际小产物、独立审阅及阶段判断均存本目录。仅通过原global lease使用GPU，旧长训与其他任务提速切换不接管。
