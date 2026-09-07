# 训练吞吐与快速验证排程诊断

**后续实测（06:36核对）**：[E8三臂已完成](../2026-09-08_train_分类快速反馈E8/three_arm_summary_20260908/README.md)，持久队列六阶段共9604.804秒，约2小时40分钟；C1未显示早期优势。实际完整epoch计时见该阶段的THROUGHPUT.md，不能把共享负载下的预算差解释为严格代码加速。下文保留性能诊断与03:31启动时的历史记录。

**selected-only通过旧新各24次真实更新逐位一致性；正常频率N/C0/C1实测0.858/0.891/2.476秒每批，E20三臂仍约13.2纯训练小时。已在看AP前采用预留E8方案并于03:31启动，预算约7.3小时，N已开始训练。原block16失败与现有E200均保留。** 结果见 [SELECTED_UPDATE24_RESULT.md](SELECTED_UPDATE24_RESULT.md)，当前队列见 [分类快速反馈E8](../2026-09-08_train_分类快速反馈E8/README.md)。

起点2026-09-08 01:55：C1按42/0/123为34/33/34轮，近期每轮约1110/1157/1120秒，外推仍需52–54小时。旧C0 shuffled/same-modal为79/73轮。此处仅吞吐估计，不使用中途AP选择方法。

本条目采集GPU利用率、项目进程CPU/I/O/等待状态和历史吞吐；只读正在运行的源码及checkpoint状态。现有run、冻结科学参数、样本流与原始产物保持原状。任何新GPU基准仍须全局lease及资源检查，不能用猜测峰值放行。

并行责任：主任务采集实时性能；baseline_feature_analysis审查计算热点；ap_error审查历史吞吐及短recipe比较限制；loc_stress审查恢复/迁移可行性。

产物根为本目录。性能采集采用独立包装脚本，不向正在运行的训练代码插桩；修改/创建文件清单随结果回执记录。

## 已完成的诊断

1. **真实吞吐明显退化。** C1 每轮约18–19分钟，历史完整C0为约3–4分钟，约5.25–5.54倍时间。历史负载不相同，这个比例不单独证明代码因果。证据见 [THROUGHPUT_HISTORY.md](THROUGHPUT_HISTORY.md)。
2. **短时主机采样不支持全机CPU或数据读取饱和。** GPU利用率高不能等同有效算力利用率；原始20秒采样见 `host_profile_20260908_015752.json`。
3. **原池化存在逐对象小张量循环。** 分块16对象的独立实现不改对象选择、门限、前背景掩码、分母或损失定义。在两个固定GPU测试形状中，原池化前向加分数梯度耗时29.70→3.98毫秒、92.18→7.91毫秒，即约7.47倍、11.66倍；仅数值容差内等价，不是整轮训练加速倍数。见 [POOL_MEASUREMENT.md](POOL_MEASUREMENT.md)。
4. **两个自然训练batch已完成原完整路径诊断。** 使用已完成N42的私有学生副本和原冻结T/R，保持原64批诊断流的前两批，源图/增强/双标签逐字段核对。第二批的同步计时：loader 0.0042秒，S/T/R前向合计0.0602秒，native loss 0.0239秒，对象选择1.7460秒，KD与统计0.0663秒，combined backward 0.2299秒。原始串行跨度扣除额外KD探针为2.2475秒。第一批含CUDA首次开销，不能用于稳定吞吐。额外同步与诊断使这些计时不等于生产每步耗时。
5. **该capture有效且资源合规。** 两批基础对象435/423、选中124/93；KD到原始分类分数梯度L2为0.00764/0.00755，有限非零；原生+KD的学生梯度255个张量有限。没有optimizer更新，不证明独立共享参数KD梯度或24次训练更新等价。NVML测得5546MiB、进程树RSS7577MiB；使用原全局lease在GPU4运行，项目物理卡仍为2/4/5。

远端原始证据：

`/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_throughput_20260908/real_probe_attempt1/`

其中 `capture_attempt1/receipt.json` 为阶段计时；`raw_batch_00.pt` / `raw_batch_01.pt` 留在服务器，存真实raw logits/DFL、标签和明确的feature形状占位，不含原图和实际特征值。分块完整选择回放在其后顺序执行，各自必须通过旧新选择、损失、统计与raw-score梯度比较。

### 首个真实回放的结果：有速度收益，但未通过预设数值门

`real_probe_attempt1/replay_00_attempt1/receipt.json` 的完整选择+KD统计+raw-score反向交替7次计时，中位数1.66072→0.91760秒，约 **1.81倍**。这仍排除模型、DataLoader、optimizer，并非整轮加速比。

该回执状态为 **FAIL_EQUIVALENCE**：仅部分 `teacher_delta` 在接近零的数值上超出预设 `atol=1e-6, rtol=1e-5`，最大绝对差1.9073486e-6。学生/参考delta通过；对象ID、选择、质量、统计、最终loss和学生raw-score梯度在此batch逐项exact，教师/参考梯度为空。不能将这份回执写成通过，也没有据此放宽容差。第二batch回放被顺序准入门阻止，未运行。

原失败回执、全部时间样本和资源记录保留在 [remote_real_probe_attempt1](remote_real_probe_attempt1/collection_manifest.json)，不重用目录。后续不再靠更改归约形状追逐该中间误差，改为验证保留原逐对象算术的 selected-only 路径；原失败不能改写成通过。

二维归约候选的同一bundle复验也保留同一失败（最大差仍1.9073486e-6），其耗时1.62346→0.80066秒；没有把改变reshape当作修复成功。见 `remote_rank2_probe_attempt1/`。停止继续盲试归约形状。

### 02:34 已启动完整更新诊断

`update24_probe_attempt1` 已使用原全局lease和独立screen `rgbirperf_update24_42` 启动。旧C1与分块C1各在新解释器执行24次成功optimizer update（最多96次尝试），沿用原通用初始化、实际系数0.09227393550836771、原样本流与AMP设置。逐步保存损失、梯度、模型、optimizer、EMA、AMP跳步、随机状态、双标签与首批真实像素；CPU比较器的16项真值/负例已通过独立检查。

这是评估浮点差异和真实训练耗时的受限诊断，不自动准入正式替换；此前中间值失败依然保留。审计拷贝开销与训练阶段计时分列。源码审阅见 `performance_candidate/update24/INDEPENDENT_CODE_REVIEW.md`，新峰值按本次执行独立测量。

**该诊断已完成：两臂各30批、24次成功更新及6次AMP跳过。** 初始化、30批源图/增强/双标签/worker RNG、首批RGB/IR像素、对象选择及控制状态逐项exact。但24步训练轨迹不满足原固定数值容差：实际总损失最大差0.41510，未加权KD最大差0.000634，student状态（含buffers）最大差0.00771；因此不能把分块实现当作在训轨迹的无损替换。AMP非有限梯度发生于正常跳过阶段，不能仅凭这个计数判训练失效，也不能把后续轨迹差异解释成方法收益或失败。

去除单列审计时间后的24个热身后观测batch，中位数2.4517→1.1634秒，约2.11倍；这包含同步观测且不含loader等待，仍非生产epoch外推。详细原值与CPU回执闭合见 [update24_readout_attempt1/README.md](update24_readout_attempt1/README.md) 和 `summary.json`，原始小产物在 `remote_update24_probe_attempt1/`，大state张量留94。此结果推动下一步转向保留原逐对象归约和cast位置的selected-only实现。

### 更高优先级的训练薄路径

源码显示R的全类相对logit只服务统计，未选中的S/T对象也不贡献KD。新候选先原样完成C0资格、quality、全局选择与全GT背景排除，再只对选中对象计算S/T全类内容，并保持全base分母和原有效尺度平均；完整统计在原日志/观察批次采集，未采集字段明示缺失。它可继续用原逐对象归约，不必先解决分块中间值差异。

这个方案先在有限合法FP16输入域完成源码与数学范围审阅，再进入以下实测；有限性检查、既有共享梯度观察和空集反向不能删除。见 [THIN_PATH_FEASIBILITY_REVIEW.md](THIN_PATH_FEASIBILITY_REVIEW.md)。

**02:50，两批真实raw回放已通过。** 保留原逐对象pool及全score共用FP32 cast的selected-only版本，在两批的选择/背景mask/分母/选中S-T内容、最终loss和全部1,344,000维student score梯度上均exact，纯薄路径和同raw额外完整统计两种模式均通过。没有启用FP32或异常域回退；参考/未选中的全类诊断明确未采集。两批中位数分别1.74569→0.92691秒、1.61963→0.84459秒，即约1.88、1.92倍；这是选择+相应统计+raw-score反向，仍不含模型/loader/optimizer。

原小产物见 `remote_selected_probe_attempt1/`。此后 `performance_candidate/update24_selected_only/` 已完成独立旧/新各 24 次更新验证，结果来自实际训练，未由 raw 通过或同公式推导。

**selected-only 的真实训练验证通过：两臂各 30 批、24 次成功更新、6 次 AMP 跳步；新臂 thin/full-diagnostics/fallback=30/30/0。** 初始状态、首批像素、全部批次的 source/增强元数据/双标签/worker RNG、选择与有效尺度、selected S/T delta、native/KD/target/off-target 损失、全 scaled/applied 梯度、逐次模型状态（含 buffers）、optimizer、EMA、scaler 与控制均字节 exact，原容差通过，max abs=0。共享参数梯度观察与 sanity JSON 也一致。完整统计仍在同 raw 的 no_grad 路径另外采集，实际学习始终使用薄图。该结论不覆盖旧 block16 失败，不外推 E200 或检测增益。

本次完整 canary span 为 134.085→187.674 秒，新臂因每批额外完整统计而更慢；额外诊断实测 81.048 秒。去前 6 批后，扣审计和额外诊断的观察 batch 中位数为 3.23796→2.28345 秒，约 1.42 倍；完整 wall 与全部样本均保留，不能当作无插桩生产 epoch 吞吐。两臂 NVML 峰值均 6510 MiB，进程树 RSS 峰值 29367/29397 MiB；guard 无错误，整卡最低剩余 4236 MiB。两臂 65 对源码快照及配置副本逐字节一致，实际 candidate/harness 与冻结本地版本也一致。详见 [SELECTED_UPDATE24_RESULT.md](SELECTED_UPDATE24_RESULT.md) 与 [selected_update24_readout_attempt1](selected_update24_readout_attempt1/README.md)。

短程反馈以12小时为准备目标，成本和匹配条件见 [FAST_FEEDBACK_OPTIONS.md](FAST_FEEDBACK_OPTIONS.md)。E20/E40是独立短日程，不能与旧E200直接比较；尚未启动新的短程训练矩阵。

三臂 E20 配置和独立入口已在 `short_screen_draft/` 准备，仍标 DRAFT，尚未启动。03:18 以原全局 lease、独立 screen `rgbirperf_cadence_42` 启动 N→C0→C1 顺序性能探针，每臂24次真实更新，连续计时包含loader等待和正常统计。N/C0各30批、24次成功更新、6次AMP跳步，热身后分别0.858475/0.890618秒每批，NVML各6304MiB，进程树RSS28738/28737MiB；C1尚待完成。这是短窗口吞吐，非完成epoch。远端为 `artifacts/rgbir_throughput_20260908/short_profile_attempt1`。原每小时 heartbeat 先读本诊断的最新记录、优先跟进加速验证，保持无变化时静默及原服务器资源规则。

## 处理决定与限制

- selected-only 已完成至少 24 次成功 update 的真实学习/梯度/状态验证；当前工程优先级是核对训练 cadence 下三臂实测吞吐和短程预算，不得用算子或扣诊断倍数承诺 E200 完成时间。
- 当前已运行C1与两个旧归因臂未停止、未热替换代码。checkpoint不能恢复完整原随机状态与训练轨迹，详见 [RESUME_OPTIONS.md](RESUME_OPTIONS.md)。新实现若需要重启，应明确登记新attempt，不能写成原run连续训练。
- 下一代想法先采用独立、匹配的短程N/C0/候选对照获得方向反馈；完整三seed和归因仍承担正式结论责任。短程轮数/LR与比较集合必须预先登记，不与旧E200直接比较。
- 性能验收应加入以后每个新路径的准入：热身后多批吞吐、完整阶段占比、同卡并发吞吐及显存，而不仅是正确性和显存通过。

## 插桩与新增文件

### 正常频率测量完成与实际排程

`remote_short_profile_attempt1/` 三臂均完成30批、24次成功更新、6次AMP跳步。热身后连续24批、含loader等待与原正常日志开销，N/C0/C1分别20.6034/21.3748/59.4189秒，即0.858475/0.890618/2.475788秒每批；C1 thin/full/fallback=30/3/0。NVML6304/6304/6368MiB，进程树RSS28738/28737/28703MiB，原lease监控无越界。该测量不能与另一负载时段旧C1直接相除得到正式提速比，也没有覆盖完整epoch或每100批统计周期。

根据实际成本，三臂E20纯训练预算13.2145小时，未放行E20。读取任何新AP前采用已预留E8早期筛查，纯训练5.2858小时，20%波动余量加1小时其他开销合计7.3430小时；共享排队时间仍未知。E8独立release差异审阅发现并修复了仅影响回执的MiB单位误替换，最终READY见 `selected_only_independent_review/E8_RELEASE_REVIEW.md`。原E20草稿保持。

E8已经通过实际准入并在03:31由原global lease入队，N在GPU4开始；具体配置、限制、运行身份及小证据副本统一见 [新的训练条目](../2026-09-08_train_分类快速反馈E8/README.md)。本页上方的“准备中/未启动”段落是此前过程记录，当前状态以本节和顶部为准。不把短程启动写作研究冲刺完成。

`collect_profile.py` / `collect_runtime_sources.py` 只读主机、源码和checkpoint元数据；`analyze_throughput_history.py` 读取历史结果；`performance_candidate/` 存新池化、旧新基准与诊断capture；`run_pool_probe.py` / `run_real_probe.py` 复用既有dispatcher；`deploy_*` 仅写新远端attempt并字节比较；`collect_real_probe.py` 仅收小产物。原release、正式配置、原始结果及checkpoint均未修改；未计算新哈希。代码和计量范围审阅见 [candidate_review.md](performance_candidate/candidate_review.md)。
