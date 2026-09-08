# 单批真实 DFL 读出：独立实际验收

**`pass`：这一次原首 32 图、80 GT 的真实分布读出与 CPU 派生结果可按限定范围接受。317 份原始分布、720 条对象角色记录和 27 组汇总已全量独立复核；这不是教师分布可迁移、可学或改善检测的证据。**

审阅者 `/root/dfl_readout_review`；可观察模型身份 `unavailable`。实际 forward ID：`raw_dfl_new_forward::probe::1788854451018472481`。审阅者仅执行 CPU 解析/复算，未启动 GPU、主干前向、训练、官方 test 或新 hash。

## 执行与身份

GPU producer 完成耗时 21.426 秒，完整队列含退出/监测 61.705 秒，exit 0、monitor_errors=[]。实际 source copies 四份与七份冻结 release 快照相应文件逐字节相等；原始早期/最终收集的十份 probe 文件逐字节相等。新 first_batch_stream 和完整 GT identity 与旧 witness 按解析值 exact；完整 32 帧有 31 个非空 GT 帧。

实际本批 hook 记录 S/R/T 各一次；三模型 state_dict 前后 exact、无参数梯度、optimizer/EMA 更新 0。S train+BN frozen，R/T eval，实际 AMP=true。通用 check_amp 在 setup 中按原已验证值临时替换，实际只调用一次并恢复；明确不宣称 setup 与旧版本完全等价，也不把模型普通 anchor/stride 缓存归入不变的 state_dict。没有再做 NMS、匹配、loss、训练或完整 dev。

317 份分布为 S83/R83/T151，逐项等于独立预先冻结名单。全部 80 个 GT 的角色、历史 anchor、own-GT、未 clamp 距离、缺失 null 均匹配；教师始终以 IR own-GT 解释。所有原始 logits 确为无损保存的 FP16 值，未由框或期望反构造分布。83 对相同索引 S/R raw 字段 exact。

## 数值与派生复核

|全量复核项目|结果|
|---|---:|
|317 份概率：FP64 独立复算对已存 FP32 最大绝对差|3.251×10⁻⁷|
|期望距离最大差|1.650×10⁻⁶ bin|
|FP64 期望重建框对原 FP32 框最大差|4.166×10⁻⁵ px|
|native 与 FP32 框最大差（分别保留）|0.066895 px|
|native 概率和相对 1 的最大差|3.535×10⁻⁷|
|独立重算的对象角色/分组汇总|720 / 27，全部通过|

实测全部 317 份记录的 dtype 顺序为：raw logits **FP16** → native DFL.conv 输入概率 **FP32** → native DFL 输出距离 **FP16** → native 解码 **FP32**。不能笼统称作“native 概率 FP16”；早期 `SOURCE_FINDINGS.json` 中对低精度概率的泛称由本实测说明限定。原生输出距离与其概率的 FP64 加权和最大差 0.004179 bin，保留 AMP 卷积舍入，不归一或修补概率。

独立复算未导入 producer/analyzer 数学函数，使用 stdlib logsumexp、熵/方差、相邻 bin CE 和分组统计。317 份分布描述、720 条角色及 27 组汇总在绝对/相对 10⁻¹⁰ 容差内闭合。一个 GT 的 5 个角色存在非法距离，非法边/四边均值按合同保持 null，没有偷偷 clamp。另核 9 组共同合法身份连接；S 历史 R 候选与 T 同索引的 79 个共同合法对象具有完全相同的标签框/距离。共享标签及同索引仍不证明物理配准。

这里 CE 是本模型、本 anchor、原未 clamp GT 在 `0≤d<15` 上的相邻 bin 诊断，熵单位 nat、方差 bin²；不是实际训练 DFLoss（后者还含 clamp/分配/权重）。各角色合法集合可能不同，不能任意相减其均值；没有跨不同 anchor/bin 做 KL，也没有借此选择阈值或 λ。

## 资源与证据范围

实际 GPU0，任务 NVML 峰值 1,684 MiB、进程树 RSS 峰值 16,336 MiB；全卡观测最低余量 22,378 MiB，项目总 RSS 抽样峰值 157,503 MiB。原 dispatcher 援引四卡例外，加入后项目 GPU 为 [0,2,4,5]，仍有 [6,7] 两张空卡；监测回执完整。以上是采样及 guard 证据，不声称证明每个瞬时资源值。

可接受的结论是：真实完整 4×16 概率信息已保存，身份/坐标语义可追溯，并能给出 own-GT 读出。合成真值证明相同期望可有不同分布形状；这不等于实际教师的额外内容更可靠、更可学或优于 GT 重监督。本次只有一个训练批次和 seed42，固定 11 个正向/1 个反向对象，不产生 AP、梯度或 KD 收益结论，不解除空间几何或训练准入条件。

## 复核入口与边界

- 原始结果：`../evidence_1602_final/probe/`；派生结果：`../cpu_analysis/output_attempt1/`。
- 独立原值检查：`verify_actual_cpu.py` → `ACTUAL_JSON_CPU_attempt1.json`。
- 独立派生复算：`verify_analysis_cpu.py` → `DERIVED_CPU_attempt1.json`；共同分母补项：`APPENDIX_CPU_attempt1.json`。
- 最终源码/身份/资源复核：`FINAL_BINDING_RESOURCE_CPU.json`；源码准入及七文件快照：`SOURCE_REVIEW.json`、`reviewed_source/`。
- `SOURCE_FINDINGS.json`、初始独立检查失败及源快照作为历史保留，不与最终验收混淆。

`audited_input_hashes: not computed`。按明确任务边界未计算 hash；使用实际来源路径、stat、逐字节快照和执行回执。未重新加载 checkpoint、重新运行 GPU 或比较历史完整像素/DFL；参数/state 事实来自已绑定的实际执行 guards。通过本审计不等于排除所有可能的完整性缺陷，也不建立未测试的科学效用。
