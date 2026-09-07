# C0 / C1 / C1_y 独立审阅

**结论：接受“保持 C0，固定选择后测试对象×尺度×类别相对 logits 的 C1”作为下一轮类别线；内核公式没有发现阻断性错误，但不能直接复用旧校准器，也不能把 C1_y 的梯度剂量写成必然降低。先补完整训练轨迹兼容验证，再独立完成 C1 三 seed。**

审阅日期：2026-09-07。范围：只读新规格、ZIP 内源码与测试、当前 frozen OEv1 和既有梯度校准器，以及前次 native/载体审计；不执行 ZIP 内程序，不使用 GPU，不修改在跑任务。包中“18 项 CPU 测试通过”为材料自带回执，本审阅未将其冒充现场复验。

## 1. 读入依据

- `07_研究分析/RGBIR_Independent_Class_Loc_Formal_Spec_20260907.md`，重点 §3–4、§8–10、§12–14。
- 同名 `_Package.zip` 中 `reference_kernels.py`、`test_reference_kernels.py`、README 和测试/文档回执。ZIP 共 8 个条目，提供的是低层算子参考，未包含真实 trainer/selection adapter/梯度校准器。
- `03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_task_conditional_v1/legacy_oev1/object_evidence_loss.py`。
- `03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_task_conditional_v1/calibrate_gradient_scale.py`。
- `08_实验日志/2026-09-07_audit_native与蒸馏载体/README.md`。

## 2. 可接受的定义及它实际回答的问题

| 臂 | 每对象的监督载体 | 可回答的问题 | 不能单独回答 |
|---|---|---|---|
| C0 | 正确类 P3/P4 相对证据先平均成一个标量，SmoothL1 | 原 OEv1 在匹配 N 上的净效用 | 完整类间关系是否有用 |
| C1 | 每有效尺度保留所有类别相对 logits，逐类 Bernoulli KL；目标类 1，非目标类平均后乘 .25 | 一套更丰富的对象级类别目标能否优于 C0 | 全部差值都是“信息量”导致 |
| C1_y | C1 中仅保留目标类，保持相同全局系数 | 在此固定结构化损失中加入非目标类项的效果 | 每单位梯度的信息增益，或独立拆清尺度与损失形状 |

C1 是对象级相对 logit 蒸馏，Drone 两尺度均有效时最多 10 个数；不是 raw anchor confidence、不保留对象内部空间，也不是局部 feature KD。规格把它与当前讨论明确对应，避免同时改变选样和载体，是合适的快速路线。LLVIP 单类别下 C1=C1_y，因此不应为 LLVIP 重复跑这两臂；C1 相对 C0 只能检验尺度聚合与损失形状。

`sigmoid(Delta/tau)` 的数学语义是人为定义的相对证据目标。它不是 detector confidence，也不能凭 Bernoulli 形式称为校准概率。对每类共同加性 logit 偏置不变；对乘性尺度、前景区域内容变化和跨模态 logit 校准差异并不不变。当前规范对此边界表述正确。

## 3. 需在实施计划中明确的两项修正

### 3.1 删除非目标类项不保证降低梯度范数

规格 §4.6、§8.3 的“降低总梯度剂量 / 实际剂量降低”不成立于一般共享参数梯度。设加权目标类分量为 `g_y`、非目标类分量为 `g_nt`，则 C1 为 `g_y + g_nt`，C1_y 为 `g_y`。例如一维梯度 `g_y=1, g_nt=-0.8`，C1 范数为 .2，删除非目标类后为 1，反而增大。

应改成：**保持目标类项系数，删除非目标类项会改变实际梯度剂量与方向，变化需测量。**记录同一批次/参数集合上的 `||g_y||`、`||g_nt||`、二者余弦、`||g_y+g_nt||` 和相对 native 的范数/余弦。不新增动态权重，不据结果补改 eta。

数学上即使目标/非目标 raw logit 通道分开，在共享骨干上梯度也可抵消，不能以分类通道不同排除此问题。C1_y 同 lambda 是明确的“删除项”消融，但不是剂量严格相等的内容消融。

### 3.2 旧校准器不满足新状态契约

现有 `calibrate_gradient_scale.py:46` 把 R 权重学生副本设为 `eval()`；循环中没有新规格要求的逐 batch 恢复参数及 BN buffers，也只支持旧 L 校准路径。因此不能仅替换被求导 loss 就宣称完成新 C1 校准。

新校准器应在隔离进程和固定真实增强 batch 中：

1. 读取与 N/C0 完全同一有效 recipe 和 pinned 源码；保存 R 状态快照。
2. 每批恢复模型参数、buffers，学生使用实际训练模式；T/R 固定 eval/no_grad。
3. 在同一个学生前向图上算 C0 与 C1，均保留前景和背景梯度，避免两个前向导致不同 BN 状态。
4. 固定 P3/P4 检测头输入模块的实际 parameter names，确认 head.f 指向的模块及 trainable 参数；保存未使用参数信息，不能以 allow_unused 静默掩盖断路。
5. 对 `B*.1*L_C0` 与 `B*L_C1` 求未缩放 FP32 梯度，记录 64 批，至少 16 批双方非零且有限。median 比值获得 lambda；未经裁剪须在 (0,1] 内才自动准入。
6. 保存逐 batch 副本状态恢复检查、batch ID/增强身份、原生与三种类别损失梯度；不 optimizer.step，不污染正式训练 RNG。

一次局部梯度范数配平不代表整个 E200 更新剂量相等。规格要求的预定训练时点剂量观察应保留，观察不改变 lambda。

## 4. 当前代码衔接最容易出错的位置

现行 OEv1 先聚合全部 matched objects，然后以 `base = region_ok & reference_ok` 形成 E_C（旧源码约 295 行），分母是 `base.sum()`，selected 的索引仍指向全部 matched objects。参考 C1 kernel 却要求输入第一维 M **恰好是 E_C 全体**。

因此 selection adapter 必须做显式索引映射：全部 matched ID → E_C 紧凑序号 → selected mask，不能把全部 matched 数量或 selected 数量当分母。保存 RGB/IR GT 全局 ID、image/augmentation ID、valid levels 和映射表，校验 `M==旧 stats.base_count`、selected 原 ID 顺序完全相同。

旧 C0 前景是本对象 GT 框，背景是外环排除本模态全部 GT；教师与学生区域不能偷共用。旧 C0 q 在跨尺度平均及 clipping 前定义；C1 不应使用新 per-class KL 反过来排序。稳定 tie-break、ceil(.5*eligible)、粗候选阈值和 teacher-correct 必须保持。

保留旧 C0 调用作为 oracle；不为了共享新 pooling 函数修改旧 C0。参考新 pooling 是逐对象循环实现，与旧向量化 masked LME 可能有细小浮点差异；把它用于 C1 可以，但 C0 兼容模式须以实际回归通过为准。

## 5. C1 新增内容的风险诊断

“教师 GT 类别判断正确”只能支持选择该对象，不能证明该对象全部非目标类别关系可信。尤其本对象 GT 前景框可包含重叠的其他类别 GT；当前代码只对背景排除全部 GT，前景并未排除其它对象。C1 的非目标响应可能编码相邻/重叠物体及场景布局，而不是该对象的语义混淆；两模态自身 GT 区域又可能不同。

建议加入只读诊断，不改变首版选择规则：按前景与其它 GT 的覆盖率/重叠类别分桶，报告非目标项强度、T/R/S 相对证据、原生 raw confidence、目标/非目标梯度方向。保留标签及 ROI 来源，使异常强非目标项能定位到具体对象。若 C1 失败，这比笼统解释为“更多信息产生负迁移”更有证据。

补充核验 C1 的单尺度 raw Delta clipping 与 C0 的跨尺度平均后 clipping 是不同操作；这是 C1_y−C0 的额外结构差异，应在方法差异表中列出，不能把两者等同。

## 6. 内核静态检查与待补验收

包内 `classification_relative_kd` 的 KL(T||S)、双方温度、tau²、teacher detach、目标类系数、非目标类平均、有效尺度平均与 E_C 分母一致；学生不 clamp、空集合连到学生图、单类别退化处理也合理。未发现需改变数学定义的阻断错误。

现有测试覆盖低层温度与梯度、mask、分母、单类与空集，但并未验证真实对象选择、模型生命周期、优化器/EMA 或数据流。除规格已有测试外，建议优先补：

- 两尺度误差方向相反的例子：C0 平均可能抵消，C1 分尺度 divergence 不应抵消，确保未错误先平均。
- 将非目标教师通道单独改变：C1 值/梯度改变，C1_y 严格不变；同时验证共享参数梯度可以相消。
- 非零 clipping、极大正负学生 Delta、无效尺度和未选对象组合，验证有限 loss/gradient 及正确 mask；不要仅测试小幅随机张量。
- 真实含 unmatched GT、重叠对象、单有效尺度和末尾小 batch 的 adapter 映射/分母回归。
- 同一 RGB 输入和双标签 batch，旧与新 N/C0 完整参数 loss/gradient、成功 optimizer updates、AMP skips、EMA keys/值及更新轨迹。首 30 批覆盖 workers 分发差异，不能用首 4 批一致代替完整验收。

参考函数内多处 `bool(tensor)` 和逐对象循环可能产生 GPU 同步，需实际 canary 测速度/显存/RSS；不可用现有 C0 资源峰值直接放行 C1，更不能为了快擅改 worker/batch 后复用旧 N。

## 7. 最小实验矩阵及复用条件

**阶段 A（无新 E200）：**绑定现有 N/C0 六个独立端点；标准 native 与新 weight0 的工程差异继续保留说明。完成新 wrapper 的 N/C0 真实训练轨迹兼容、selection 回归、固定 C1 校准、C1/C1_y 各至少 24 次成功 optimizer update 的 canary。

**阶段 B（固定新增 3 个类别 E200）：**C1 seed42 先启动，技术验收后按 0/123 补齐，不因 seed42 AP 不利改 seed 或提前截断。独立比较 C1−N、C1−C0。以三 seed 同向且平均至少 +0.10 pp 作为升级实际价值门，记录它不是显著性标准。

**阶段 C（条件新增 3 个类别 E200）：**C1 值得保留才 C1_y×3；同一 lambda/selector/recipe，只删除非目标项，报告剂量与方向变化。若 C1 不胜 C0，保留 C0，停止本版自动载体扩展。ROI feature、dense C2、eta sweep、dose 对照均不塞入首轮。

现有 C0-random、C0-shuffled、C0-same-modal 按原身份收尾。C1 改了表示/损失，不能借 C0 的 null 对照完成 C1 跨模态归因；最终胜出版本仍需遵守工作区 paired/shuffled/same-modal/N 四臂与三 seed 规则。该后续预算要独立列出，不能宣称本轮 3 或 6 个类别新训练已覆盖全部论文要求。

复用 N/C0 的必要条件：同一 split、初始化文件、教师/参考、workers、增强与数据流协议、优化器/调度/AMP、全部有效 args、原始代码身份、last/EMA 独立 evaluator；新兼容路径通过全链路验收。不要为消除旧 native 的观察差而悄悄改新的 data seed 后仍复用现有 N。若决定建立独立数据流的三 seed，应另立协议并将匹配 baseline 预算单列。

## 8. 推荐结论措辞

可说：“C1 在与 C0 相同对象选择下测试更多类别和尺度结构；KL 目标是相对证据，是否更优由匹配三 seed 决定。”

不说：“完整 logits 一定更有知识”“C1_y 总剂量更低”“18 个内核测试说明训练可用”“原生损失等价已经证明旧 native 全程等价”“C0 的四臂可以直接证明 C1 的跨模态机制”。
