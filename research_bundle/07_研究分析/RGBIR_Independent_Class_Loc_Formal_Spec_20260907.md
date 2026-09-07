# RGB–IR 单任务蒸馏正式实施方案
## 类别证据 / 类别 logit 与定位 DFL：分别实现、分别验证，暂不融合

> **日期：2026-09-07｜方案版本：INDEPENDENT-CL-v2.0｜读者：研究负责人、现场 Codex**  
> **本轮交付目标：**保留已经有正面证据的 OEv1 类别单蒸馏；实现一个可比较的类别 logit 升级候选和一个独立定位 DFL 蒸馏候选。先让类别、定位各自相对匹配 baseline 获得可重复收益。**不实现 C+L，不实现跨任务路由，不用融合结果掩盖单分支无效。**  
> “正式实施方案”表示公式、接口和判读规则已明确到可编码，并不表示新方法已经验证有效、真实配准已经通过或长训自动获准。

---

## 0. 开工前的两项说明

### 0.1 已知结果与使用前提

本轮用户提供的新前提是：**现有类别单蒸馏已取得小幅、三 seed 稳定提升。**本方案据此将 OEv1 作为应保留的有效参照，而不是要求推翻它重做。

同时，本次实际读取的 GitHub 分支 `research/full-evidence-20260906` 固定在提交：

```text
7bfaf2fe837d2ddfea4b0bde4e7ede0a21434af9
```

该提交的 `TASK_CONDITIONAL_STATUS_20260907.md` 标注快照时间为 **2026-09-07 03:22 +08:00**，仍记录：N/C/C-random 共 4/9 独立端点，只有 seed42 完成严格 C−N 配对，ΔmAP50–95 为 +0.144554 pp；定位 L 尚未在已接纳的几何覆盖内获得合格对象。**因此，用户提供的更晚结果与仓库可见快照要分开记录，不能自行填出另外两个 seed 的数值。**[P1]

现场 Codex 只需补齐实际最新的六份 N/C 独立终点评估、训练/评估回执及同 seed 配对表。若这些证据已存在，直接登记并复用；不要仅因本报告读取到旧快照而重训。若缺失，标记 `C0_THREE_SEED_EVIDENCE_PENDING`，可以继续实施新代码和诊断，但不在报告中写“本次已经独立核实三 seed 成功”。

本文不把任何历史约 +1 pp 的跨 campaign 参照当作正式 C−N，也不以两个 P 的均值减另一个 seed 的 N。

### 0.2 与上一份 MD 的优先级关系

本文件是对 `RGBIR_Task_Conditional_KD_Codex_Spec_20260907.md` 的**增补和下一阶段策略替换**，不是历史预注册勘改。

| 原计划 / 最近讨论 | 本轮采用的策略 |
|---|---|
| 优先 N/C/L/C+L，关注 CL−C | **分别跑类别线和定位线，先关注 C−N、L−N；不新增 CL/CGT 训练** |
| 类别固定为 OEv1 标量 | 保留 **C0=原 OEv1**，增加 **C1=按尺度保留完整类别相对 logit 向量**作为升级候选 |
| 定位点估计或 DFL 二选一 | **L1 正式候选固定为同物理 anchor 的 DFL 分布 KL；不自动回退成框 IoU KD** |
| 最后设计双任务路由 | 本轮不做；只有单分支证据成立，才单独制定下一阶段协议 |
| 把更多 logit 信息视作自然升级 | **信息更多不保证更好，C1 必须和 C0 比较，失败就保留 C0** |

不自动终止已经启动的历史训练，不修改其配置、教师、loss 或回执。对于尚未启动的旧 CL/CGT 扩展，按新计划登记 `DEFERRED_BY_INDEPENDENT_CL_V2`，而不是继续旧的大矩阵。资源或作业变更仍按现场用户授权执行。

---

## 1. 最终要交付的两条方法线

### 1.1 方法身份表

| ID | 方法 | 蒸馏知识 | 是否带另一项任务 KD | 当前证据身份 |
|---|---|---|---|---|
| N | 同流水线零 KD | 无 | 否 | 匹配 baseline |
| C0 | 原 OEv1 | 每对象一个正确类别前景−背景证据标量 | 无定位 KD | 用户已报告正面三 seed；现场补绑定原始证据 |
| C1 | Object-relative Multi-class Logit KD | 每对象 × 有效尺度 × 全部类别的相对 logit | **无定位 KD** | 新候选，未验证 |
| C1_y | C1 的目标类内容消融 | 与 C1 相同，但只监督 GT 类别通道 | 无定位 KD | 验证非目标类信息的增量 |
| L1 | Aligned Localization Distribution KD | 每选中对象一个同物理 anchor 的四边 DFL 分布 | **无类别 KD** | 新独立候选，几何和剂量待准入 |
| L_GT | L1 的 GT 内容对照 | 同对象、同 anchor、同 mask 的 RGB GT 两 bin 分布 | 无类别 KD | 检验教师分布是否优于额外 GT 监督 |

**C0/C1 中“只有类别蒸馏”并不等于不训练定位。**所有实验都保留完整 RGB 原生检测损失，包括分类、框和 DFL。类似地，L1 仍然有原生分类监督，只是没有额外类别 KD。

### 1.2 核心问题只有三个

1. **类别线：**现有 C0 的收益是否可复用；C1 能否在不依赖定位 KD 的情况下，比 C0 更好？
2. **定位线：**L1 在没有类别 KD 帮助的情况下，能否比 N 更好？
3. **内容解释：**C1 的非目标类信息和 L1 的教师分布，是否确实提供超出简单替代目标的价值？

不能用 `C1+L1>N` 来替代任何一项。当前 trainer 应拒绝同时启用类别 KD 与定位 KD。

### 1.3 为什么类别正式首版不是密集逐 anchor 全类别模仿

最近讨论提出过“每个前景 anchor 保留相对类别 logit”。本轮为了可实施、可归因，先固定为**对象级、分尺度的完整类别向量**，暂不做逐位置硬对应。

这是明确的保守选择：现有对象 GT 对应可以支持各模态各自在自身目标区域池化；但同一类别 ID 不意味着 RGB 与 IR 的两个任意空间位置可以直接比较。直接密集模仿会同时引入空间配准、候选定位质量和多 anchor 剂量三个新因素，定位线当前也正被几何覆盖阻塞。[P1][P1]、[P2][P2]、[P3][P3]

在 DroneVehicle 的 5 类、P3/P4 两尺度都有效时，C0 每个对象最终是 **1 个数**，C1 是 **2×5=10 个数**；两个尺度分别计算损失，不先平均成一个向量。C1 仍然丢弃目标内空间布局，**不是密集蒸馏，不能在论文中写成每 anchor 分布迁移**。这是当前正式候选的定义，而不是声称它必然优于密集方法。

后续只有确实需要空间结构、且有相应对齐证据时，再另立 dense C2；本轮不实现 dense C2。

---

## 2. 共用训练、数据和版本契约

### 2.1 三个模型的角色

- **T：冻结 IR 教师。**沿用已核实的原 seed42 教师。
- **R：冻结 RGB 参考。**沿用原 seed42 RGB reference，只用于固定的候选和质量判断。
- **S：待训练 RGB 学生。**从原通用 YOLO11n 预训练初始化，不从 R 接着训练。

冻结 T/R 的参数和 buffers 行为，固定 `eval()`，前向采用 `no_grad()`；它们不得进入 optimizer、student EMA、部署 state_dict 或导出图。学生保留完整可训练检测器，不因为实验名 C/L 就冻结另一个头。当前工程的这些基本角色见原 OEv1 与 Task-Conditional 计划。[P2][P2]、[P3][P3]、[P4][P4]

### 2.2 主数据集和训练 recipe

主线 DroneVehicle：RGB-only 推理、IR→RGB 训练辅助。沿用现场已验证 OEv1 的完整有效配置，历史配置为 train 17,990 / dev 1,469，YOLO11n，640，E200，batch32/nbs64，workers4，SGD，lr0/lrf=.01，momentum=.937，wd=.0005，warmup3，AMP，deterministic，seeds0/42/123，固定 last/EMA 独立评估。[P2][P2]、[P4][P4]

这些数值不能仅靠目录名推断；Codex 应读取最新 run 的实际 args 并生成共用 `effective_recipe.yaml`。沿用已接纳的增强和双标签 RNG 回放；不为了新 loss 升级 PyTorch/Ultralytics、不改为 SSL 初始化、不换学生容量、不顺便调整 batch、worker、packing 或验证端点。

读取 IR 独立训练 GT 的权限必须在回执中明确记录。它参与对象对应、背景排除和教师正确性，不可再声明“KD 未使用教师标签”。

### 2.3 原始张量契约

```text
scores: [B, C, A]           原始类别 logits，未 sigmoid
boxes:  [B, 4*R, A]         原始 DFL logits，不是解码后的 xyxy
feats:  P3/P4/P5            提供真实网格 H/W
strides: 与上述网格一致
R:      reg_max，从真实模型读取，禁止任意 reshape 后猜测
```

DFL 明确按下式恢复：

```python
x = raw_boxes.float().reshape(B, 4, reg_max, A).permute(0, 3, 1, 2)
# [B, A, 4, R]，四边顺序 l,t,r,b
```

类别顺序必须在 T/R/S 中相同。anchors 要核对每层偏移、中心 `+0.5`、拼接顺序、stride 和数据增强后的输入坐标；形状相同不构成几何相同的证明。

---

## 3. 类别部分 C0：原 OEv1 完整保留

### 3.1 公式

对象 i、模态 m、有效尺度 l、正确类别 y_i：

$$
\Delta^m_{il,y_i}
=\operatorname{LME}(z^m_{l,y_i}[F^m_{il}])
-\operatorname{LME}(z^m_{l,y_i}[B^m_{il}]),
$$

其中 $\operatorname{LME}(x)=\log\sum_j\exp(x_j)-\log n$。前景为本模态 GT 框内 anchor 中心；背景为同心 2 倍框内排除本模态**全部** GT 后的环带。

$$
e_i^m=\frac{1}{|V_i|}\sum_{l\in V_i}\frac{\Delta^m_{il,y_i}}{2}.
$$

每层至少 1 个前景点、4 个背景点，只用两模态共同有效尺度。基础集合 E_C、reference coarse candidate、teacher-correct 检查、原 q 排序、eligible 的前一半、稳定 tie-break 全部保持旧实现。

$$
q_i=[\operatorname{softplus}(-e_i^R)-\operatorname{softplus}(-e_i^T)]_+,
$$

$$
L_{C0}=\frac{1}{\max(1,|E_C|)}\sum_{i\in\mathcal S_C}
\operatorname{SmoothL1}\left(e_i^S,\operatorname{sg}(\operatorname{clip}(e_i^T,-8,8))\right).
$$

组合为 `native_total.sum() + actual_B * 0.1 * L_C0`。[P2]

### 3.2 必须原样保留的边界

不把 R 的粗候选 IoU 0.1 改成 0.5，不把“教师优于冻结 R”改成“教师优于当前 S”，不修改 top-rho、原 background mask、valid level、温度、分母、clipping 或 SmoothL1 beta。

原文中 clipping 前的 q 与 clipping 后的目标可能有边界差异；本轮只增加统计，不悄悄修入 C0。修改这种规则必须产生新方法 ID，不能继续复用 C0 的结果。

从旧函数抽取选择逻辑时，必须证明同一 raw T/R/S/batch 下：对象对、E_C、eligible、selected ID、e、KD、native items 和 raw-score 梯度保持等价。**重构不是免费的科学无变更假设。**

---

## 4. 类别部分 C1：对象相对、多类别、分尺度 logit 蒸馏

### 4.1 固定知识定义

C1 **复用 C0 的对象集合与选择结果**。对于每个被匹配对象，在每个共同有效尺度，分别计算所有 C 个类别：

$$
\Delta^m_{ilc}
=\operatorname{LME}(z^m_{lc}[F^m_{il}])
-\operatorname{LME}(z^m_{lc}[B^m_{il}]),\quad c=1,\ldots,C.
$$

保留 `[M, 2, C]`，M 是 E_C 的对象数；用 `valid_levels[M,2]` 标记 P3/P4。**先逐尺度、逐类别计算 divergence，再对有效尺度平均；不得先把 P3/P4 平均后再做 KL。**

S/R 使用各自 RGB 区域；T 使用其 IR GT 的区域，背景仍排除本模态所有对象。T 与 S 的池化区域可以包含不同数量的点，LME 各自除本区域点数。这里假设是“同一对象、同一尺度、同一类别的区域证据可比较”，不是假设像素完全对应。

C0 的 selected mask 仍由正确类别证据计算。**不要因为 C1 有更多类别通道，就重新用向量 KL 来选样；本轮只改变被传递的内容及其损失。**

### 4.2 为什么不用类别 softmax KL

当前 YOLO 是 Sigmoid/BCE 类别协议。[P2][P2]、[P3][P3] C1 对每个类别分别构造一个 Bernoulli 目标，不要求所有类别的概率和为 1。对 C 个类别直接 softmax 会额外引入类别互斥归一化，尤其单类别时 softmax 恒为 1，不能这样实现。

定义：

$$
u^T_{ilc}=\frac{\operatorname{clip}(\Delta^T_{ilc},-16,16)}{\tau_C},\qquad
u^S_{ilc}=\frac{\Delta^S_{ilc}}{\tau_C},\qquad \tau_C=2,
$$

$$
p^T_{ilc}=\sigma(u^T_{ilc}),\qquad p^S_{ilc}=\sigma(u^S_{ilc}).
$$

这里 clipping 在**原始差值**上做，之后才除温度；C0 的 ±8 是已除 2 的 scalar，不能把这两个空间混用。学生差值不 clamp，避免把大误差位置的梯度截断。

注意：$p=\sigma(\Delta/\tau)$ 是我们定义的**相对证据目标**，不是 detector 原始 confidence，也不是校准后的“此处是某类目标”的概率。局部背景相减可消除同类别的共同加性 logit 偏置，但不保证消除乘性尺度、区域统计或跨模态校准差异。

### 4.3 每类二元 KL

$$
D_{ilc}=p^T\log\frac{p^T}{p^S}+(1-p^T)\log\frac{1-p^T}{1-p^S}.
$$

实现用 `logsigmoid`，或用 BCE 减去教师熵得到同样的 KL；BCE 接收 logits，teacher target 是 sigmoid 概率。**学生与教师必须都除同一个温度**。只把教师除温度、学生仍喂原始 logit 的写法不采用。[R2][R2]、[R3][R3]

教师 target 全部 detach。学生的前景池化和背景池化都保留梯度；背景是本方法判别关系的一部分，不在本轮改成 stop-gradient。

### 4.4 正确类与非目标类的权重

固定首版：目标类系数 1；其余类别的平均 divergence 乘 $\eta=0.25$：

$$
\ell^C_{il}=D_{il,y_i}+\eta\frac{1}{C-1}\sum_{c\ne y_i}D_{ilc},\quad C>1.
$$

C=1 时仅保留 $D_{il,y_i}$，不出现除零，也不虚构额外类别知识。LLVIP 上 C1 相比 C0 主要改变了尺度聚合和损失形状，不可声称提供“多类别 dark knowledge”。

这是一份**待验证的固定设计**，不是已调优超参。选择 0.25 是为了让新增非目标类项成为有限增量，不让 C−1 个通道按求和主导。不要把它写成已发现的最优值。

不再除以 `1+eta`。这样 C1_y 只需 eta=0，即可保持目标类系数不变，删去非目标类监督；对照含义明确。

### 4.5 对象级归一化与总损失

$$
L_{C1}=\frac{\tau_C^2}{\max(1,|E_C|)}
\sum_{i\in\mathcal S_C}\frac{1}{|V_i|}\sum_{l\in V_i}\ell^C_{il}.
$$

$$
L_{\mathrm{train},C1}=\operatorname{sum}(L_{native})+B\lambda_{C1}L_{C1}.
$$

E_C 是 C0 教师质量筛选之前的基础集合，不是 selected 数量；也不能在 loss helper 内把 selected 子集长度误当 E_C。只选少量对象时不除以 selected 数重新放大。没有入选对象返回连到学生 scores 的可导零。

C1 是**替换 C0 的知识项进行比较**，不是 `C0+C1`。不得通过叠加两个类别 KD 再宣称向量比标量好。

### 4.6 C1_y 的准确身份

C1_y 与 C1 使用相同的 E_C、selected、valid levels、tau、teacher clipping、reference、trainer 和全局系数，只令 eta=0。

它回答“在这个结构化类别损失上，加入非目标类项是否有收益”，不单独证明“每单位梯度信息更多”。删除项会降低总梯度剂量，必须测量剂量变化。若论文要把收益严格归因于内容而非剂量，后续可新增一个梯度配平的 C1_y-dose 对照；**本轮不自动启动该额外分支。**

C1 对比 C0 同时改变了类别维度、尺度聚合、SmoothL1→Bernoulli KL，不能把整个差值全归给“信息数量”。C1_y 是第一项低成本拆解，而不是完整因果分解。

### 4.7 类别线必须记录什么

记录 E_C / eligible / selected 及逐类、目标尺度、亮度、场景组成；teacher/reference/student 的目标类与非目标类证据；teacher clipping 比例；target binary entropy（标注为相对证据熵）；每尺度、目标/非目标类 KD；和原生 cls/box/DFL 共享参数的梯度比。

评估不能只看 margin 上升。C1 的 region pooling 仍可能让模型通过压背景，或提升 GT 内定位较差位置的分数来降低 KD。要同时看正确目标召回、错类、重复框、背景误检和 AP。**logit 更丰富不自动修复 OEv1 的这些代理目标风险。**

---

## 5. 定位部分 L1：独立同-anchor DFL 分布蒸馏

### 5.1 本轮不重新造定位框架

仓库已经存在：

```text
experiments/rgbir_task_conditional_v1/
    localization_loss.py
    geometry_contract.py
    geometry_audit_tools.py
    diagnose_opportunities.py
    calibrate_gradient_scale.py
    train_task_conditional.py
```

本轮应复用、核验并接入 **L-only**，不要重写一套与实际 D1/D2 脱节的选样器。目前 `localization_loss.py` 已定义单对象单 anchor、原生候选唯一归属、DFL 支撑检查、教师/参考质量门和 GT 内容对照。[P3][P3]、[P5][P5]

### 5.2 DFL 的语义与正式损失

同一 anchor a、每条边 d 的 logits：

$$z^m_{ad}\in\mathbb R^R,\quad d\in\{l,t,r,b\}.$$

softmax 后得到距离 bins 的预测分布。用于解码框和判断 IoU 时，使用**原生温度 1**的分布期望；蒸馏时才用 $\tau_L=2$。不要用 softening 后的期望重新定义质量门。

$$p^m_{ad}=\operatorname{softmax}(z^m_{ad}/\tau_L).$$

$$\ell^L_i=\frac{\tau_L^2}{4}\sum_d KL\left(p^T_{a_i d}\Vert p^S_{a_i d}\right).$$

这是预测分布迁移，保留多 bin 相对偏好；不是只对解码框做 IoU，也不是对 raw logits 直接 MSE。定位分布 KD 已有 LD 等先例，本轮不把 KL 算子本身当作新颖贡献。[R1]

### 5.3 几何严格限定为已核验的近似同坐标网格

L1-v1 只支持：同 augmented input 像素坐标、同 grid/stride、同 anchor 中心、同四边方向和距离 bin 支撑。采用近似恒等对应前，必须有独立几何证据，经过 `geometry_contract.py` 生成对象/anchor 准入 mask。

以下均不构成几何准入：两侧分辨率相同、同一条 augmentation RNG、GT IoU 高、LLVIP 标签复制重合、某个全局 CKA 高、任意置 `verified=true`。

**修正前面讨论里容易引起误解的说法：**仅在 W(a) 对教师 DFL logits 做最近邻或双线性采样，不保证 bins 已被正确转换。参考点、距离尺度、边方向或映射形变改变后，还需要分布的重参数化/重采样。首版不实现非恒等 W，也不利用每对 GT 强拟合局部变换后声称是教师知识。

形状一致但物理参考点不同：直接拒绝。配准证据只覆盖局部区域：只在覆盖该目标及相应边界附近的位置启用 L。不能拿远处背景结构点替所有目标背书。

### 5.4 基础集合 E_L 的形成顺序

在 `no_grad()` 下使用 T/R 的固定输出和各模态训练 GT，沿用已有配置：[P3][P3]、[P4][P4]

1. RGB/IR GT 按同类、IoU≥0.5 作最大有效匹配数优先的一对一对应；再要求该对象对 IoU≥0.8。后者只减少标注/框差异，不代替独立配准。
2. 候选位置限 P3/P4，落在已接纳的几何区域；同时位于两侧 GT 内。
3. 两侧 GT 相对 anchor 的未裁剪 l/t/r/b 距离全部落在 `[0, reg_max−1−0.01]`。
4. 必须对**所有 RGB GT**计算 pinned native TAL 几何候选，包括未配对 GT、其他类 GT 和实际实现的小框扩展。只保留有唯一几何归属、且属于当前对象的位置，避免对重叠对象重复施加定位监督。
5. R 的任意类最大分数≥0.05，解码框与该 RGB GT IoU≥0.1。
6. 每个对象只选一个 anchor：**R confidence 最大 → R IoU 最大 → 全局 anchor ID 最小**，严格稳定破平局。此选择不看当前 S，也不去追逐教师最佳 anchor。

这之后得到 E_L，固定分母 `max(1, len(E_L))`。几何、标注和参考粗候选本身已经筛掉一部分对象，因此 E_L 不是纯 RGB 数据集合，也不是全部 GT。

“唯一几何归属”不等于“当前学生 TAL 最终正样本”。最终 assignment 还依赖分数/框质量；要记录 selected anchor 被当前学生原生 assigner 实际分配的比例，不能在术语上把两者等同。

### 5.5 在实际 a_i 上做固定质量门

用温度 1 解码，令：

$$u^R_i=IoU(b^R_{a_i},g_i^{RGB}),$$
$$u^{T,RGB}_i=IoU(b^T_{a_i},g_i^{RGB}),\qquad
u^{T,IR}_i=IoU(b^T_{a_i},g_i^{IR}).$$

在已核验近似同坐标假设下，首版门固定为：

```text
R 的 argmax 类别 = GT 类别，且 confidence >= 0.25
T 的 argmax 类别 = GT 类别，且 confidence >= 0.25
R 对 RGB GT 的 IoU < 0.70
T 对 RGB GT 的 IoU >= 0.60
T 对自身 IR GT 的 IoU >= 0.50
T_RGB_IoU - R_RGB_IoU > 0.05
```

这是已有 task-conditional 配置的固定过滤，不是本轮按新 AP 调参。[P3][P3]、[P4][P4] 通过为 1，否则为 0；不额外加熵门控、连续 IoU 权重或 top-rho。**首版只选择，不做双任务路由。**

教师必须在同一个实际 a_i 上通过，不能用“教师另一个 anchor 有好框”替当前位置通过。单类别数据中 argmax 本身无信息，前景 confidence 和定位门仍必须保留。

### 5.6 损失与直接梯度路径

$$L_{L1}=\frac{1}{\max(1,|E_L|)}\sum_{i\in\mathcal S_L}\ell^L_i.$$

$$L_{\mathrm{train},L1}=\operatorname{sum}(L_{native})+B\lambda_{L1}L_{L1}.$$

只取 S 在这些 anchor 的 DFL logits；T 概率 detach，mask 和候选 detach。L1 对 `student_raw['scores']` 没有直接 KD 梯度，定位分布有梯度；共享骨干仍可能让分类结果变化。

一个对象只一个 anchor，避免无意按目标面积增加剂量。若未来改成多 anchor，必须先对象内平均，再除 pre-teacher base_count；这是新版本，不在本轮顺手扩展。

### 5.7 教师 IoU 更好不保证其 DFL 分布更好

同一个预测均值可以来自不同分布；高 IoU 不保证教师 GT-DFL CE 更低，分布尖锐也不保证正确。已有 D2 诊断已经记录这种差异。[P1][P1]、[P3][P3]

因此记录 T/R/S 对 RGB GT 两 bin 目标的 CE、teacher entropy 和边界误差；**不把 entropy 作为本轮质量门，也不声称现有 IoU 门保证无负迁移**。若 teacher IoU 好但分布监督冲突很多，它是 L1 失败后要分析的具体机制，不是提前加上多个额外门来保证训练正结果。

---

## 6. L_GT：定位分支必须保留的内容对照

### 6.1 对照意义

RGB 原生 GT 已经告诉模型框在哪里。L1 的额外收益可能来自教师分布，也可能只是对挑中的对象多做一份定位监督。因此 L_GT 与 L1 保持：**同 E_L、同 a_i、同 quality gate、同分母、同 tau、同 lambda、同训练和评估规则**，只替换 target。[P3][P3]、[R4][R4]

注意 L_GT 仍继承 IR GT 和教师选样，**不是完全 RGB-only 基线，也不能隔离辅助标注的全部贡献**。

### 6.2 目标的精确定义

对每条边，RGB GT 到同一个 a_i 的真实未裁剪 bin 距离为 d。设 k=floor(d)：

$$q_k=1-(d-k),\qquad q_{k+1}=d-k,$$

其余 bins 为 0。支撑范围不合法直接拒绝，不把原生 `clamp` 后的数重新当作合法证据。

为了沿用现有温度契约，对 GT 分布作：

$$q^{(\tau)}_j=\frac{q_j^{1/\tau_L}}{\sum_v q_v^{1/\tau_L}}.$$

零概率保持零；整数 d 时仍是单 bin 点质量。不对任意 logits 作 softmax 来凭空构造“GT logits”。再计算：

$$L_{L\_GT}=\frac{\tau_L^2}{4\max(1,|E_L|)}
\sum_{i\in\mathcal S_L}\sum_d KL(q^{(\tau)}_{id}\Vert p^S_{a_i d}).$$

该目标和原生温度 1 的标准 GT DFL 不完全相同，应命名 `same_mask_tempered_gt_dfl`，不要直接写成“原生 DFL 原封不动再算一次”。L1/L_GT 同系数不等于实际梯度剂量相同，记录两者梯度范数；必要的剂量匹配属于后续单独对照。

---

## 7. L 当前阻塞如何处理：先补几何，而不是偷换实验

本次仓库快照显示，在未核验几何的诊断假设下，Drone train 2,048 图 / 31,931 GT 中实际 anchor 门选中 626 个；LLVIP train 2,048 图 / 5,592 GT 中选中 211 个。但已接纳的 LLVIP 局部几何帧没有覆盖可用 GT，**可正式用于 L 的对象仍为 0**。[P1]

因此必须区分：

```text
UNVERIFIED_OPPORTUNITY > 0     在同坐标假设下有定位机会
VERIFIED_TRAIN_SELECTION = 0  当前证据还不允许正式蒸馏
```

这不是“L 训练失败”，也不是“没有定位知识”；更不能用非零的假设诊断把正式 gate 放开。

### 7.1 有限范围的补证方案

复用已经冻结的几何 roster 和已有 48 对查看结果，不重跑一套同名探针。剩余审核优先覆盖**训练图中潜在 L 候选目标的局部邻域**，而不是只在背景上加控制点。沿用既有独立结构点 / region coverage、复核划分和尺度容差；若需要改变抽样重点或证据模型，登记新的几何审核版本，但不根据 AP 改阈值。[P4]

几何准入需要保存：image/group ID、证据方法、局部有效区域、独立检查点、误差统计、审核者/来源、增强后的误差传播和覆盖 mask。框 IoU 只能是额外条件，不能替代这些证据。

不在 full train 上为了“凑够对象”放宽 geometry mask。只用已证明覆盖的区域；未覆盖记 `UNKNOWN_GEOMETRY`，不是 `BAD_GEOMETRY`。

### 7.2 正式准入的最小量化检查

固定 64 个**训练**增强 batch（规则见第 8 节），要求至少 16 批出现有限、非零 L 梯度，且入选对象来源不集中在单一图或单一场景。保存全部过滤漏斗和唯一对象/场景数；16/64 是本方案的工程可测性门，不是统计显著性或性能保证。

若不足：返回 `BLOCKED_GEOMETRY_OR_SIGNAL_COVERAGE`，交付现有产物和明确缺口，不开 L1/L_GT E200、不改成“有标签就视为配准”、不把 CL 当作替代。

Drone 不满足而 LLVIP 满足时，可把 LLVIP 作为定位机制 pilot，但须建立 LLVIP 自己的 N/L1/L_GT，同模态标签来源限制明确报告。**C 在 Drone 成功、L 在 LLVIP 成功，不等于两部分已经在同一任务成立，也还不是融合准入。**

---

## 8. 权重、温度和剂量：不把相同 lambda 当公平

### 8.1 固定设计参数

| 参数 | C0 | C1 / C1_y | L1 / L_GT |
|---|---:|---:|---:|
| loss | 原 SmoothL1 | Bernoulli KL | 四边 DFL KL |
| temperature | 原 2 | 2（双方） | 2（双方分布） |
| scale | P3/P4 按原规则 | P3/P4 各算 loss 后平均 | P3/P4 中每对象一个 anchor |
| teacher clip | e 空间 ±8 | 原始相对差值空间 ±16 | 不剪 DFL logits |
| off-target weight | 不适用 | 0.25 / 0 | 不适用 |
| 分母 | 原 E_C | 相同 E_C | 固定 pre-teacher E_L |
| 系数 | **0.1 不改** | 梯度配平后一次冻结 | 梯度校准后一次冻结 |

这些新常量是有边界的初始设计，不是搜索出来的最优参数。

### 8.2 校准数据与状态

使用 64 个固定 train batch，seed=20260907；沿用真实增强、batch32，T/R 全程冻结。校准在 R 权重加载到的**学生结构副本**上进行，不用它继续正式训练，不读取 dev/test AP。

每个 batch 前恢复这个副本的参数和 buffers（包括 BN）；学生以实际训练模式前向，确保校准点固定且与训练接口相容。用 `autograd.grad` 分别取梯度，不调用 optimizer.step，不污染正式学生或训练 RNG。

共享参数集合 Θ 固定为已接纳校准器选取的 P3/P4 输入特征相关可训练模块，保存**实际 parameter names**和模块来源。若无法解析，不退回任意整个模型悄悄改变度量。完整 pipeline 测试仍要核验全模型参数梯度。

### 8.3 C1 的配平目标是现有 C0，而不是任意绝对 loss 值

每个有效 batch j：

$$g_{0,j}=\nabla_\Theta\left(B\cdot0.1\cdot L_{C0}\right),\qquad
g_{1,j}=\nabla_\Theta(BL_{C1}).$$

$$\lambda_{C1}=\operatorname{median}_j\frac{\|g_{0,j}\|_2}{\|g_{1,j}\|_2}.$$

至少 16 批两者非零且有限；分母为零的批单独记录，不能靠 epsilon 得出一个巨大系数。该系数未经裁剪地落在 `(0,1]` 才自动准入；越界返回 `CALIBRATION_REVIEW_REQUIRED`，不默默截断然后宣称已配平。

C1_y 默认使用同一个 lambda_C1，让目标类项系数不变。记录其实际剂量降低，避免过度解释。

### 8.4 L 的校准沿用已有方式

$$g_{N,j}=\nabla_\Theta\operatorname{sum}(L_{native}),\qquad
g_{L,j}=\nabla_\Theta(BL_{L1}),$$

$$\lambda_{L1}=\min\left(1,0.1\operatorname{median}_j\frac{\|g_{N,j}\|_2}{\|g_{L,j}\|_2}\right).$$

这是与现有 L 计划一致的初始梯度尺度方案，不是理论保证。[P4] 记录上界是否触发、实际 ratio 的中位数与分位数、valid batch 数及唯一对象数。全零时没有合法系数，返回 `NO_LOCALIZATION_SIGNAL`。

L_GT 使用与 L1 完全相同的 lambda，另报实际梯度比。两条线不为了“对称”强制同一个数值系数。

### 8.5 正式训练中只观察，不在线调权

在固定 batch 索引或 epoch 0/10/50/100/199 的预定样本上记录加权 KD / native 梯度范数比和余弦；不按观察结果动态改 lambda，不用 AP 搜超参。

BN/AMP、最终梯度 clipping 和优化器动量都会影响更新，loss 数值比不是实际更新比；梯度范数配平也只是局部度量。记录 AMP 跳过次数、optimizer attempts、成功 updates 和 EMA updates，不能把 E200 自动解释成所有更新数完全相同。

---

## 9. 实验矩阵：类别与定位两条线分别跑

### 9.1 阶段 A：实施与资格检查，不开新的大矩阵

完成当前 C0/N 三 seed 证据绑定；实现 C1；复用 L1 代码并完成几何/D2/校准准入；跑 CPU 与真实 batch canary。已有 C0-shuffled、C0-same-modal、C0-random 继续按其原身份登记，不能自动变成 C1 的对照。

**类别线无需等 L1 几何准入。定位线也不以 C1 是否优于 C0 为准入条件。**两者独立推进。

### 9.2 类别主实验

| ID | seed | 数据/端点 | 主要比较 | 复用规则 |
|---|---|---|---|---|
| N | 0/42/123 | Drone E200 last/EMA | baseline | 只复用已匹配且完整的结果 |
| C0 | 0/42/123 | 同 N | C0−N | 已有稳定结果优先复用 |
| C1 | 0/42/123 | 同 N/C0 | **C1−N、C1−C0** | 新增 3 runs |
| C1_y | 0/42/123 | 同 C1 | **C1−C1_y** | C1值得保留后新增 3 runs |

第一次正式新 run 固定 seed42，只能用于技术 smoke 和预先登记的开发 pilot；不因为它的 AP 低就换 seed。完成技术验收后按固定 seeds 0/42/123 收齐 C1，三 seed 主比较不按先完成的有利结果截断。

C1 若不稳定优于 C0：类别线继续使用 C0，**不需要为了叫“logit 蒸馏”强行换掉有效标量方法**。可以结束 C1 版本并保留结果，而不是无上限调 eta/temperature/target selection。

### 9.3 定位主实验

| ID | seed | 数据/端点 | 主要比较 | 约束 |
|---|---|---|---|---|
| N | 0/42/123 | 与定位线完全匹配 | baseline | 验收后复用；换数据集不能借用 |
| L1 | 0/42/123 | E200 last/EMA | **L1−N** | lambda_C=0，几何与 D2 必须已准入 |
| L_GT | 0/42/123 | 同 L1 | **L1−L_GT，L_GT−N** | 同选择和剂量 convention，仅换 target |

L1 先完成一套独立三 seed，再决定是否补齐 L_GT 全部 seeds。若 L1 对 N 无稳定收益，本轮不继续 CL“救结果”，也不称整个定位蒸馏领域失败。

### 9.4 计算预算与分批顺序

在 N/C0 的 6 个终点都能严格复用的前提下：

- 核心新训练：C1×3 + L1×3 = **6 runs**。
- 两条核心线有保留价值后：C1_y×3 + L_GT×3 = **最多再 6 runs**。
- 总计最多 **12 个新增学生长训**，而不是默认一次提交 12 个；教师加载、诊断、真实 canary 和几何审核另计。

没有可复用 N/C0 时，先修复/绑定证据，必要补训单列。定位被几何阻塞时，本轮只推进类别的 3 或 6 个新 runs，不把 L0 长训当作填空。

后续若准备声称“跨模态独有价值”，胜出分支还需同协议有效 RGB-only 教师对照；同数据 teacher/helper 权限必须公平。该追加实验属于**单分支归因**，仍不是融合，但不包含在当前 12 个上限中，不自动排队。

### 9.5 现在明确不跑

不新跑 C+L、CL、CGT、joint logit、learned router、动态教师切换、动态 EMA 参考、共享/私有特征分解、熵门控、teacher ensemble、OS-SSL+KD 或全数据集大范围并行探索。

---

## 10. 如何判断“稳定提升”，以及何时继续

### 10.1 所有差值都在相同 seed 上计算

$$\Delta_s=100\left(AP^{method}_s-AP^{baseline}_s\right),\quad
s\in\{0,42,123\}.$$

报告三个原始数、mean 和 sample SD（ddof=1）。**不减两臂 SD，不用 batch/image 数代替训练 seed 数，不用不同 campaign 的近似 AP 填缺 seed。**

本方案建议将 `min_practical_gain_pp=0.10` 作为新增版本继续投入的默认门槛，在读出新结果前记录。它是成本/实际价值门，不是期刊要求或显著性标准；不得追溯套到用户已有 C0 来改变其历史结论。

### 10.2 分类分支的两个不同判定

| 判定 | 条件 |
|---|---|
| C1 本身可用 | 三 seed C1−N 全正，mean≥0.10 pp；无未解释的明显错误/主要类别损伤 |
| C1 替代 C0 | **三 seed C1−C0 全正且 mean≥0.10 pp**；背景误检和主要类别退化没有抵消其价值 |
| C1 与 C0 基本持平 | 保留更简单且已验证的 C0，不把接近零写成升级成功 |
| 非目标类知识确有增量 | C1−C1_y 方向及幅度可重复，并结合实际剂量解释；未胜出则采用更简单候选 |

小正差但未过实际价值线可以记录为“方向性改善，暂不升级”，不是武断地判成无效。三 seed 方向一致也不意味着已经通过某个统计显著性检验。

### 10.3 定位分支的两级结论

| 判定 | 条件与允许主张 |
|---|---|
| L1 独立有效 | 三 seed L1−N 全正，mean≥0.10 pp；AP75 与固定对象定位诊断总体不矛盾 |
| 有定位机制支持 | 原定位不足对象的边界误差/IoU 得到改善，同时记录漏检、FP、类别变化；不能只凭 AP75 |
| 教师定位分布有增量 | L1 稳定优于 L_GT；否则只能说选中对象上的额外定位监督有效 |
| 未正式训练 | 几何/信号 gate 不通过，记 `BLOCKED`，不统计为负训练结果 |

默认 harm review 条件：AP50 平均下降超过 0.20 pp、主要类别出现明显系统性损伤，或开发集背景误检明显增加，必须专门分析，不自动过关；这些不是统计证明。

**不要求分类 logit 新版和定位新版都必须最终保留。**C0 有效而 C1 无效时，类别可保留 C0；L1 无效时不凑第二模块。未来才考虑把同一部署任务上有效的类别方法与有效 L 组合。

### 10.4 统计边界

T/R 固定 seed42，学生三 seed 只覆盖其实际发生变化的随机因素。现有框架原生 DataLoader 可能固定数据流；保留历史条件，不偷偷改新方法的 data seed 后仍复用旧 N。若全量 checkpoint 覆盖全部初始化且数据流又相同，名义 seed 可能没有有效变化，必须报告。[P1][P1]、[P4][P4]

跨 seed 初始 tensor/BN buffers、首批图像与双模态标签、固定前 30 批索引、augmentation/worker 状态都要记录。分类头少量权重不同可以构成“固定数据流下的初始化重复”，但不是完整训练随机性覆盖。

图片/场景 bootstrap 只能描述固定模型在采样总体上的差值不确定性，不能把它说成重新训练的不确定性。需要 bootstrap AP 时重算整套检测 AP，不能平均所谓每图 AP 代替。

---

## 11. 分任务评估：要看修复了什么，也要看伤害了什么

所有主 AP 在同一完整开发集、同一 evaluator、同一 NMS/precision/imgsz/batch、固定 last/EMA 上计算。禁止只在被教师挑中的对象上报告主要 AP。

### 11.1 类别线

记录 mAP50–95、AP50、AP75、逐类 AP、固定 conf 下 precision/recall；基于冻结 R 定义的“正确位置低分”“错类”“背景误检”“原本 R 更强”等对象组，报告修复与新增错误。

固定配对/候选规则，避免每个模型自己重新挑有利对象。低分正确候选增加不等于 AP 一定提高；重复框和背景分数也要保留。S 的 pooled margin、target entropy 和 loss 只是训练诊断，不是检测指标。

### 11.2 定位线

用冻结 R 和固定对象 ID 建立定位缺口分组：R class-correct 且 IoU 0.1–0.5 / 0.5–0.7 / 已较好，另列无候选，不把无候选随意填成 IoU=0 混入同组。

同时保存两个视角：

- **同一 reference anchor 视角：**S/T/R 解码框对 RGB GT 的 IoU、四边归一化误差、GT-DFL CE、分布熵、native 最终 assignment 命中率。
- **完整检测视角：**正常 NMS 后目标命中、定位错误、漏检、重复和背景 FP；一对一匹配规则、阈值及完整预测都保存。

如果只有同-anchor loss/IoU 改善而最终 AP 不变，就不能说完成了有效检测提升。若 AP75 增加，也不能单独断言原因全是几何更准，因为排序变化同样影响 AP。

---

## 12. Codex 应修改哪些代码，哪些不能动

### 12.1 推荐组织

先检查现场已有 `rgbir_task_conditional_v1` 的实际可导入路径，不要只修改 GitHub 镜像却运行其他副本。建议新增隔离目录：

```text
experiments/rgbir_independent_kd_v2/
    README.md
    selection_adapter.py         # 导出并核验原 C0 选择结果
    classification_logit.py       # C1 / C1_y 的 pooling 与 Bernoulli KL
    localization_adapter.py       # 包装已验收 L1/L_GT，不重造选样
    criterion.py                 # 单任务组合，拒绝同时 C 与 L
    calibrate_single_task.py
    diagnose_single_task.py
    train_single_task.py
    evaluate_single_task.py
    analyze_single_task.py
    configs/
    tests/
```

实际文件名可随现场规范调整，但职责分离不能丢失。旧 `legacy_oev1`、旧 train/eval receipt、已跑队列与配置保持原样；新配置登记 method ID、父版本及偏差。

### 12.2 C1 选择缓存的接口

建议返回下列 detached 元数据和各模态区域信息，不能只返回 selected IDs：

```text
ClassificationSelection:
    base_object_ids[M]             # E_C 的有序 ID
    rgb_gt_indices[M]
    ir_gt_indices[M]
    gt_classes[M]
    selected[M]                   # C0 原选择结果
    valid_levels[M,2]
    base_count = M
    per_modality_foreground_masks
    per_modality_background_masks
    source_method_id / selection_version
```

region mask 必须继承 C0，teacher background 排除 teacher 全部 GT；不以 student GT 覆盖 teacher GT。缓存仅在同一 batch/相同增强状态中复用，不能把 raw-image cache 当成所有随机增强之后的正确选样。不得按 batch 内对象行号跨 batch 错取缓存。

### 12.3 Trainer 单任务路由是工程选择，不是研究中的双任务路由

CLI 的 `--arm` 只允许 N/C0/C1/C1_y/L1/L_GT。每次 run 只产生一个额外 KD 标量：

```python
if arm == 'N':
    kd, weight = diagnostic_zero_kd, 0.0
elif arm == 'C0':
    kd, weight = frozen_oev1_loss(...), 0.1
elif arm in {'C1', 'C1_y'}:
    kd, weight = class_logit_loss(...), lambda_c1
elif arm in {'L1', 'L_GT'}:
    kd, weight = standalone_localization_loss(...), lambda_l1
else:
    raise ValueError('Fusion and learned routing are outside this phase')

total = native_total.sum() + batch['img'].shape[0] * weight * kd
```

这是选择实验 ID，不是按对象在 C/L 间学习分配。**不同时存在两个非零 KD 权重，不写 C0+C1，也不写 C1+L1。**

不要直接使用旧 trainer 中“只要不是 weight0 就加 C”的分支来运行 L-only。检查显式开关、实际 loss 组成和梯度；run 名称 L 不证明 C 已关闭。

### 12.4 模型生命周期与等价验证

在 student/optimizer/EMA 建立后，把 T/R 作为训练专用对象附加，避免注册进 student graph；用参数 ID、state keys 和实际导出文件检查，而不只看 `requires_grad=False`。

N/C0 兼容模式必须与旧实现逐项比较：相同初始化、相同双标签 batch、相同 native loss/items、raw score/DFL 和共享参数梯度、更新尝试与 AMP skips、EMA state keys。零 KD 等价性要覆盖真实完整参数，不只一个 score tensor。

teacher/ref 额外前向不应推进数据增强 RNG。任何新增缓存、import、随机 null、调试 probe 都使用隔离状态；确认真实模块 `__file__`、版本和实际加载源码快照。

---

## 13. 配置规格：必须物化有效配置，不能把 null 当默认值

下面是建议的新 adapter schema，不是现有 Ultralytics 原生参数。Codex 要实现解析与校验，再生成展开后的 `effective_config.yaml`；不认识的字段拒绝，禁止悄悄忽略。

```yaml
schema: rgbir-independent-kd-v2
method_id: RGBIR-C1-RELATIVE-MULTICLASS-v1
arm: C1
source_commit: 7bfaf2fe837d2ddfea4b0bde4e7ede0a21434af9
inherit_verified_oev1_recipe: PATH_TO_FROZEN_EFFECTIVE_RECIPE
teacher_identity: PATH_TO_VERIFIED_IR_TEACHER_RECORD
reference_identity: PATH_TO_VERIFIED_RGB_REFERENCE_RECORD
student_init_identity: PATH_TO_VERIFIED_GENERIC_INIT_RECORD
seeds: [0, 42, 123]

classification:
  representation: object_scale_multiclass_relative_logits
  selection: exact_oev1
  levels: [0, 1]
  foreground_pool: logmeanexp
  background_pool: logmeanexp
  background_scale: 2.0
  exclude_all_modality_gt: true
  min_foreground: 1
  min_background: 4
  temperature: 2.0
  teacher_raw_delta_clip: 16.0
  off_target_weight: 0.25
  divergence: bernoulli_kl
  normalization: pre_teacher_base_objects
  lambda: null
  calibration_receipt: null

localization:
  enabled: false

run_policy:
  allow_joint_kd: false
  allow_new_routing: false
  allow_formal_train: false
  checkpoint_endpoint: fixed_budget_last_ema
  test_access: false
  calibration_batches: 64
  calibration_min_nonzero_batches: 16
  min_practical_gain_pp: 0.10
```

C1_y 只更换 ID/arm 和 `off_target_weight: 0.0`，复用 C1 calibration receipt/lambda，其余内容相同。

定位版本：

```yaml
schema: rgbir-independent-kd-v2
method_id: RGBIR-L1-ALIGNED-DFL-v1
arm: L1
inherit_verified_oev1_recipe: PATH_TO_FROZEN_EFFECTIVE_RECIPE
classification:
  enabled: false
localization:
  implementation: verified_task_conditional_localization_loss
  mode: teacher
  levels: [0, 1]
  temperature: 2.0
  gt_match_iou: 0.5
  pair_iou: 0.8
  reference_coarse_conf: 0.05
  reference_coarse_iou: 0.1
  reliable_conf: 0.25
  reference_iou_max: 0.70
  teacher_rgb_iou_min: 0.60
  teacher_ir_iou_min: 0.50
  relative_iou_margin: 0.05
  support_epsilon: 0.01
  anchors_per_object: 1
  require_unique_native_geometry_owner: true
  geometry_contract: null
  accepted_coverage_manifest: null
  normalization: pre_teacher_base_objects
  lambda: null
  calibration_receipt: null
run_policy:
  allow_joint_kd: false
  allow_new_routing: false
  allow_formal_train: false
  checkpoint_endpoint: fixed_budget_last_ema
  test_access: false
```

L_GT 只更换 ID/arm 和 `mode: gt`。正式运行前必须继承/填入与 C1 配置同样的 T/R/init/seed 身份字段，并通过完整 schema 校验。上面的 YAML 是两段关键 schema 示例，不是可以缺字段直接启动的配置文件。

`allow_formal_train` 只能由完整 recipe、权重身份、D2/geometry（L）、校准、canary 与已授权资源共同准入；不能由 agent 为消除报错单独改 true。

---

## 14. 验收测试：哪些现在已测，哪些必须现场完成

### 14.1 本报告附带内核已执行的测试

附录 A 的参考内核在本地 **Python 3.13.5 / PyTorch 2.10.0+cpu / CPU** 上执行 **18 项测试，18 passed，0 failed**。测试代码与结果在配套包中。它们覆盖池化、KL、温度、分母、梯度和单任务组合，**没有加载你们的模型/图片，没有验证真实几何或检测收益**。

### 14.2 现场必须再做的工程测试

| 类别 | 必须证明的内容 |
|---|---|
| C0 回归 | 新 wrapper 的 C0 与旧版本对象选择、scalar loss、native items、scores/DFL/共享参数梯度等价 |
| C1 pooling | 每模态各自区域；排除全部本模态 GT；有效尺度正确；空区域不产生 NaN；先逐尺度 loss 后平均 |
| C1 信息路径 | 改 teacher 非目标类可改变 C1，不能改变 C1_y；修改 teacher 相同类别加性偏置不改变相对 logits；乘性变化不声称不变 |
| C1 loss | T/S 均除温度；target-only clip；teacher/ref 无梯度；目标类、非目标类和背景梯度符合公式；C=1 有效 |
| L layout | 精确四边/bin/anchor 排列，真实 stride/support；同均值不同分布仍有非零 KL；物理错位尽管 shape 相同也被外层拒绝 |
| L selection | 最大有效匹配数优先；所有 GT 的唯一几何归属；R 预选；T 同 anchor 校验；不读取动态 S 决定 mask |
| L geometry | 没有证据时正式入口 fail closed；局部覆盖外拒绝；翻转/缩放后的 mask/误差正确传播；不允许用插值 logits 绕过 |
| L_GT | 未裁剪距离校验；两 bin 与整数边界；温度化保留零支撑；同选样且只换目标 |
| 单任务 | C-only 无直接 DFL KD 梯度；L-only 无直接 score KD 梯度；N 与 native 等价；C+L/双非零权重被拒绝 |
| 模型生命周期 | T/R 不进 optimizer/EMA/export；student 仍有完整 native cls/box/DFL 梯度 |
| 随机性 | 新旧同 seed 数据流兼容；随机 null 不改变训练 RNG；不同 seed 的有效随机性有实证 |
| 评估 | fixed last/EMA，完整 dev roster，指标单位正确，缺端点不补零；ddof=1；非法三 seed 聚合拒绝 |

### 14.3 真实 canary

每条新路径至少 24 次**成功 optimizer update**，记录尝试数与 AMP skips；出现非有限值、批量自动缩减、无实际 KD 梯度、teacher 泄漏、配准未通过，均技术停止并保存 attempt。

C1/C1_y/L1/L_GT 逐条测试；L 未准入时不能拿合成张量 canary 写成真实 L canary。两路的显存、RSS 与参数更新检查交给现场统一 guard；不抢占、不 kill 其他训练、不硬编码 GPU ID。

---

## 15. 结果、日志与报告格式

每个 run 至少保存：

```text
launch_manifest.json
resolved_model_identities.json
effective_config.yaml
implementation_snapshot/
initial_state_comparison.json
first_batch_comparison.json
calibration_receipt.json
geometry_contract.json          # L only
canary_receipt.json
kd_diagnostics.jsonl
completion_receipt.json
run_evidence/
evaluation_val.json
per_class_metrics.json
prediction_records             # 用于固定错误分析，格式按现场 evaluator
evaluation_val_roster.txt
eval_evidence/
```

历史工程不允许新写哈希时，沿用实际源码副本、文件元信息和直接字节/tensor 比较；不要以“本方案要求可追溯”为由违反现场约束。现有历史 digest 原样保留，不伪造复核。

每次日志保存 `actual_B`、`base_count`、`selected_count`、`lambda`、raw KD、加权 KD、native 各项、total、成功/尝试/EMA updates。最后小 batch 使用真实 B，不固定乘 32。`selected_objects` 若是跨 batch 累计出现次数，不能称为唯一训练对象数。

训练禁用 val 后的 AP=0 或末行列错位不用于效果表；独立 eval JSON 和对应 receipt 是端点依据。不修改原 CSV 来伪装修复。训练完成和正式实验有效性分开判定。

结果表必须分别列 C 线、L 线：

```text
classification: N, C0, C1, C1_y
contrasts: C0-N, C1-N, C1-C0, C1-C1_y

localization: N, L1, L_GT
contrasts: L1-N, L_GT-N, L1-L_GT
```

不得输出 `joint_summary`、`router_gain` 或 CL 的空白预期增益。

---

## 16. 单分支内容和同模态对照的后续优先级

先完成各自效用验证，不机械地做完所有 null。C0 的现有归因实验继续有价值，但 C1 改了知识表示，C0-shuffled/C0-same-modal 的结果不能替 C1 证明机制。

若要推进类别论文主张，优先“C1_y”与真正有效的 RGB-only teacher；若要推进定位论文主张，优先 L_GT 和同模态定位教师。对照 teacher 若与 R 完全相同，而选择又要求 teacher 优于 R，会退化成零 KD，必须禁止或另行定义成熟的同模态协议。

跨模态 teacher-specific content 的 null 可采用同类别/相近尺度的对象级目标置换；固定接收方 E/K/分母、donor 来源与随机种子，明确它只检验条件实例内容。全图 shuffled 会破坏对象位置，尤其定位场景不能把它受损当作唯一配对证据。

同 mask 内容对照继承了教师选样和 IR 标签，RGB-only 对照不应继承这些信息。两种对照回答不同问题，不能互相替代。

本轮不再开展十几篇方法的重现。LD 提供定位知识背景，CrossKD 提醒监督冲突；既有调研中的 CoLD、任务解耦和分区 DFL 工作仍是投稿前的重点比较对象。本文件不是新的全面文献搜索，也不凭“logit”一词主张新颖。[R1][R1]、[R4][R4]

---

## 17. 到什么状态才允许讨论下一步融合

只有满足以下条件，才另起后续设计任务：

1. 同一主任务上存在一个已验证类别分支：可以是 C0，不强制必须 C1。
2. L1 在该任务上单独胜 N，且改善与定位错误减少相容。
3. 两者的 matched recipe、教师/参考身份、剂量、实际有效随机性和评估端点都闭合。
4. 若要宣称跨模态具体知识有贡献，内容/同模态对照提供了相应支持；没有则收窄主张。

**这四项只表示可以设计下一阶段，不表示联合训练一定更好。**共享骨干的梯度交互、互补对象覆盖和预算都需要新的实验；本轮不预先编写融合阈值或启动组合训练。

---

## 18. 分批交付给 Codex 的任务

### 交付 A：状态、源代码与基线绑定

读取现场最新 HEAD/实际模型路径和 N/C0 六份完整终点评估；核对用户报告的三 seed 正收益。记录当前仓库快照不足之处，但不以旧快照覆盖现场新结果。登记本轮相对于旧 CL/CGT 计划的策略变更。

输出：`BASELINE_EVIDENCE.md`、`baseline_pairs.json`、源码对接清单、blocked 清单。此阶段不重训。

### 交付 B：类别 C1 与回归测试

从已验收 OEv1 导出 selection/region 信息；实现对象×尺度×类别 logits、Bernoulli KL、C1_y、单任务 trainer 分派。N/C0 完整等价检查、CPU tests、冻结配平与真实 canary；形成 C1-ready 配置。

### 交付 C：定位 L-only 准入

复用已有 localization_loss/geometry/D2；检查 L-only 真的不含 C。补几何覆盖，不篡改诊断假设。只在真实非零合格覆盖、剂量和 canary 完整后产生 L1/L_GT-ready 配置；不足时交付可定位的 BLOCKED，不虚构 λ。

### 交付 D：按阶段运行、独立评估

有既定用户 GPU 授权与 lease 时，按第 9 节的 3-seed 顺序执行；无资源或无准入则只交付就绪配置，不自行扩大权限。类别与定位分别输出结果和判读，不自行升级融合。

### 交付 E：研究决策

给出 `KEEP_C0 / PROMOTE_C1 / C1_INCONCLUSIVE / L1_EFFECTIVE / L1_NO_GAIN / L1_BLOCKED`；保留所有负 seed、technical attempts 和非显著/幅度不足结果。没有新方法成绩时不填写预期 AP。

---

## 19. 可直接复制给 Codex 的执行说明

```text
实施 RGBIR independent-KD v2，当前只做单分支，不做融合/路由。

1) 先核验最新 N/C0 同代码三 seed 完整 E200 last/EMA，绑定实际结果；
   用户报告C0小幅稳定提升，但GitHub旧快照可能未齐，不凭旧表补数。
   保留所有旧OEv1结果和运行代码，不修改在跑实验。

2) 类别：C0原样保留。新增C1：复用C0 E/selected/region/valid-level，
   各自模态GT区域的全类别前景−背景LME，保留[M,2,C]，不先平均尺度。
   T raw delta clip ±16，再双方除tau=2，以逐类Bernoulli KL蒸馏。
   GT类系数1，其余类别KL平均乘eta=.25；C1_y只令eta=0。
   对象内有效尺度平均，再sum_selected/base_count，乘tau²。
   不做dense逐anchor类别强配，不同时叠加C0与C1。

3) 定位：复用已验收localization_loss，以L-only运行。
   仅已独立核验的近似同物理网格：真实support、P3/P4、全部GT的
   native几何唯一归属、R选一个anchor、同anchor的T质量门。
   使用KL(softmax(T/tau)||softmax(S/tau))*tau²/4，tau=2。
   L_GT保持同mask/anchor/分母/lambda，替换为RGB GT两bin的温度化分布。
   没有合格几何覆盖或非零真实L梯度则BLOCKED，不能开E200或改成框KD。

4) 单任务总损失= native_total.sum()+actual_B*lambda*kd_scalar。
   CLI只允许N/C0/C1/C1_y/L1/L_GT；拒绝CL/CGT/joint/router。
   保留完整native cls/box/DFL，T/R/mask detach，辅助模型不进optimizer/EMA/export。

5) 做64个固定train batch的梯度校准并冻结，C1对齐原C0加权梯度尺度，
   L沿已有0.1 native梯度比例/上界1规则。至少16批非零，零信号不填epsilon系数。
   无AP网格调参，不中途改tau/eta/选样。

6) 验收：C0/N新旧全链路等价，loss形状/温度/分母/teacher梯度/几何拒绝、
   raw score与DFL梯度路径，真实24成功updates canary，资源峰值和seed有效性。

7) 实验两条线分别做：C1×3对N/C0；L1×3对N；
   各自有价值后再C1_y×3、L_GT×3。不自动一次启动全部12个。
   原N/C0严格匹配时复用，否则先解决证据或配置差异。

8) 所有正式run固定E200 last/EMA、同dev/evaluator、逐seed和ddof=1；
   不读CSV占位AP、不填缺seed、不碰封存test选方法。
   当前结果分别判断，C1不能稳定胜C0就保留C0，L不胜N不做融合救结果。

先交付源码、测试、基线证据和准入状态，再遵守已有授权与统一guard运行。
```

---

## 20. 来源与本轮核验边界

**项目来源固定到 7bfaf2f：**

- [P1] [Task-Conditional 阶段状态][P1]：本轮读取 03:22 快照；三 seed 结果缺口、几何零准入、现有 D1/D2 与作业状态。
- [P2] [原 OEv1 冻结协议][P2]：对象证据、选择、双标签与 loss convention。该定义也已在前次对话逐项核验。
- [P3] [现有 localization_loss.py][P3]：本轮读取 1–390 行，含 single-anchor selection、T/R 门、DFL/GT 两种 target。
- [P4] [现有 Task-Conditional 冻结计划][P4]：本轮读取；原 CL/CGT 计划、几何/校准/资源规则。本文只替换下一阶段策略，不倒改原预注册。
- [P5] [现有独立模块 README][P5]：本轮读取，用于确认已实现工具，避免重造训练栈。

**本轮重新核验的外部来源：**

- [R1] [Localization Distillation，CVPR 2022][R1]：官方摘要确认定位知识/定位区域蒸馏背景。本次没有重读全文公式或重训论文。
- [R2] [PyTorch BCEWithLogitsLoss][R2]：BCE 与 logits/sigmoid 的接口和稳定性。网站当前为2.13文档，不意味着升级现场2.10运行环境。
- [R3] [PyTorch KLDivLoss][R3]：KL 输入约定和 reduction。实现自行按对象/类别/边定义 reduction，不能无条件使用默认 mean。
- [R4] [CrossKD，CVPR 2024][R4]：官方摘要讨论教师预测与 GT 监督冲突，作为检查风险的背景，不作为本方法已解决冲突的证据。

**实际完成：**读取上述项目与文献页面，编写本实施方案及参考内核，在本地 CPU 跑过18项内核测试。  
**本轮未完成/未声称完成：**未加载服务器权重、未重新推理图像、未核验新增三 seed 成绩、未执行真实几何审核、未完成完整trainer集成、未启动GPU训练、未修改或提交远端仓库。

**最终决策原则：现有 C0 有效就保留；更丰富的 C1 用独立对照争取升级；L1 必须自己证明有用。先形成两条可信的单任务结论，再考虑如何组合。**


---

## 附录 A：可交给 Codex 的参考内核

以下代码是完整可导入的内核文件，不是完整 trainer。选择/区域/几何由第 4–7 节规定的工程产生；其中 C1 输入是 E_C 全体对象的未除温度相对 logit，而 L 输入是已经通过几何核验的所选对象 logits。不要只复制 KL 代码而跳过选择和准入。

文件名建议：`reference_kernels.py`。真实训练调用应通过工程 wrapper 做版本和形状核验；下面的低层函数不可能仅凭 Tensor 判断跨模态是否配准。

```python
"""CPU-testable kernels for an implementation specification, not a trainer.

C1 inputs are per-object/per-level pooled raw logit differences (before T).
L1 inputs must ALREADY have verified physical anchor/stride/side correspondence.
This file cannot establish correspondence, select samples or validate AP gains.
"""
from __future__ import annotations
import math
import torch
from torch import Tensor
from torch.nn import functional as F


def _positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')


def _finite_float(tensor: Tensor, name: str) -> None:
    if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f'{name} must be finite floating point')


def pool_relative_logits(scores: Tensor, foreground: Tensor, background: Tensor,
                         minimum_foreground: int = 1,
                         minimum_background: int = 4) -> tuple[Tensor, Tensor]:
    """One scale: [C,A], bool[M,A], bool[M,A] -> delta[M,C], valid[M].

    A different modality may have different region masks/counts. The caller
    supplies matched OBJECTS and the exact frozen OEv1 background exclusions.
    No temperature is applied here. Student foreground AND background get grads.
    """
    _finite_float(scores, 'scores')
    if scores.ndim != 2 or scores.shape[0] < 1:
        raise ValueError('scores must be [C,A], C>=1')
    if foreground.dtype != torch.bool or background.dtype != torch.bool:
        raise ValueError('region masks must be bool')
    if foreground.ndim != 2 or foreground.shape != background.shape or foreground.shape[1] != scores.shape[1]:
        raise ValueError('masks must have matching [M,A] shapes')
    if foreground.device != scores.device or background.device != scores.device:
        raise ValueError('masks and scores must share device')
    if minimum_foreground < 1 or minimum_background < 1:
        raise ValueError('minimum region counts must be positive')
    if bool((foreground & background).any()):
        raise ValueError('foreground and background overlap')
    valid = (foreground.sum(-1) >= minimum_foreground) & (background.sum(-1) >= minimum_background)
    z = scores.float()
    values = []
    for fg, bg, ok in zip(foreground, background, valid):
        if bool(ok):
            f, b = z[:, fg], z[:, bg]
            values.append(f.logsumexp(-1) - math.log(f.shape[-1])
                          - b.logsumexp(-1) + math.log(b.shape[-1]))
        else:
            values.append(z.sum(-1) * 0.0)
    if not values:
        return z[:, :0].T, valid
    return torch.stack(values), valid


def classification_relative_kd(student_delta: Tensor, teacher_delta: Tensor,
                               valid_levels: Tensor, selected: Tensor,
                               labels: Tensor, temperature: float = 2.0,
                               raw_teacher_clip: float = 16.0,
                               off_target_weight: float = 0.25) -> Tensor:
    """C1: [M,L,C] deltas for the entire pre-quality base E_C.

    Per-object: mean(valid levels)[KL_y + eta*mean(KL_non_target)].
    Sum selected objects / M. Target coefficient is one even if eta changes.
    M must be the ORIGINAL base count, not the count after quality selection.
    Teacher target is clipped before division by temperature; student is not.
    """
    _positive(temperature, 'temperature')
    _positive(raw_teacher_clip, 'raw_teacher_clip')
    if not math.isfinite(off_target_weight) or off_target_weight < 0:
        raise ValueError('off_target_weight must be finite and nonnegative')
    _finite_float(student_delta, 'student_delta')
    _finite_float(teacher_delta, 'teacher_delta')
    if student_delta.ndim != 3 or student_delta.shape != teacher_delta.shape:
        raise ValueError('deltas must share [M,L,C] shape')
    m, levels, classes = student_delta.shape
    if levels < 1 or classes < 1 or teacher_delta.device != student_delta.device:
        raise ValueError('invalid dimensions/device')
    if valid_levels.shape != (m, levels) or valid_levels.dtype != torch.bool:
        raise ValueError('valid_levels must be bool[M,L]')
    if selected.shape != (m,) or selected.dtype != torch.bool:
        raise ValueError('selected must be bool[M]')
    if labels.shape != (m,) or labels.dtype != torch.long:
        raise ValueError('labels must be long[M]')
    if any(t.device != student_delta.device for t in (valid_levels, selected, labels)):
        raise ValueError('selection, labels and deltas must share device')
    if m and not bool(((labels >= 0) & (labels < classes)).all()):
        raise ValueError('class index out of range')
    if bool((selected & ~valid_levels.any(-1)).any()):
        raise ValueError('selected object has no valid levels')
    if m == 0:
        return student_delta.float().sum() * 0.0
    s = student_delta.float() / temperature
    t = teacher_delta.detach().float().clamp(-raw_teacher_clip, raw_teacher_clip) / temperature
    lp, lq = F.logsigmoid(t), F.logsigmoid(-t)
    p, q = lp.exp(), lq.exp()
    binary_kl = p * (lp - F.logsigmoid(s)) + q * (lq - F.logsigmoid(-s))
    target_mask = F.one_hot(labels, classes).to(binary_kl.dtype)[:, None, :]
    target = (binary_kl * target_mask).sum(-1)
    if classes > 1:
        non_target = (binary_kl * (1.0 - target_mask)).sum(-1) / (classes - 1)
        point_loss = target + off_target_weight * non_target
    else:
        point_loss = target  # No nonexistent non-target classes for LLVIP.
    per_object = (point_loss * valid_levels).sum(-1) / valid_levels.sum(-1).clamp_min(1)
    return (per_object * selected).sum() * (temperature ** 2 / m)


def localization_from_target(student_logits: Tensor, target_probability: Tensor,
                             base_count: int, temperature: float = 2.0) -> Tensor:
    """[K,4,R] selected single-anchor objects; target already at temperature T.

    The enclosing trainer, not this kernel, must verify geometry and base_count.
    """
    _positive(temperature, 'temperature')
    if not isinstance(base_count, int) or base_count < 0:
        raise ValueError('base_count must be a nonnegative integer')
    if student_logits.ndim != 3 or student_logits.shape[1] != 4 or student_logits.shape[-1] < 2:
        raise ValueError('expected [K,4,R], R>=2')
    if student_logits.shape != target_probability.shape or student_logits.device != target_probability.device:
        raise ValueError('student and target shape/device mismatch')
    if student_logits.shape[0] > base_count:
        raise ValueError('selected objects exceed original base_count')
    _finite_float(student_logits, 'student_logits')
    _finite_float(target_probability, 'target_probability')
    target = target_probability.detach().float()
    if bool((target < 0).any()) or not torch.allclose(target.sum(-1), torch.ones_like(target.sum(-1)), atol=1e-6, rtol=1e-6):
        raise ValueError('target must be normalized and nonnegative')
    return F.kl_div(F.log_softmax(student_logits.float() / temperature, -1), target,
                    reduction='sum') * temperature ** 2 / (4 * max(1, base_count))


def aligned_dfl_kd(student_logits: Tensor, teacher_logits: Tensor,
                   base_count: int, temperature: float = 2.0) -> Tensor:
    _positive(temperature, 'temperature')
    _finite_float(teacher_logits, 'teacher_logits')
    if student_logits.shape != teacher_logits.shape:
        raise ValueError('teacher/student localization shapes differ')
    target = F.softmax(teacher_logits.detach().float() / temperature, -1)
    return localization_from_target(student_logits, target, base_count, temperature)


def gt_dfl_target(distances: Tensor, reg_max: int, temperature: float = 2.0,
                  support_epsilon: float = 0.01) -> Tensor:
    """RGB GT distances in bins, BEFORE clamp. Two-bin q, then q**(1/T).

    This supplies the same-mask GT content control, not ordinary native DFL.
    """
    _positive(temperature, 'temperature')
    _finite_float(distances, 'distances')
    if distances.ndim != 2 or distances.shape[-1] != 4 or reg_max < 2:
        raise ValueError('expected distances[K,4] and reg_max>=2')
    if not 0 < support_epsilon < 1:
        raise ValueError('support_epsilon must be in (0,1)')
    d = distances.detach().float()
    if not bool(((d >= 0) & (d <= reg_max - 1 - support_epsilon)).all()):
        raise ValueError('GT distances outside unclamped support')
    lo = d.floor().long()
    fraction = d - lo
    target = d.new_zeros((*d.shape, reg_max))
    target.scatter_add_(-1, lo.unsqueeze(-1), (1 - fraction).unsqueeze(-1))
    target.scatter_add_(-1, (lo + 1).unsqueeze(-1), fraction.unsqueeze(-1))
    softened = target.pow(1 / temperature)
    return softened / softened.sum(-1, keepdim=True)


def combine_single_task(native_total: Tensor, kd: Tensor, actual_batch: int,
                        weight: float, task: str) -> Tensor:
    """API deliberately has only ONE KD scalar: no C+L path in this phase."""
    allowed = {'N', 'C0', 'C1', 'C1_y', 'L1', 'L_GT'}
    if task not in allowed:
        raise ValueError('unsupported task; fusion/routing not implemented')
    if not isinstance(actual_batch, int) or actual_batch < 1:
        raise ValueError('actual_batch must be positive integer')
    if not math.isfinite(weight) or weight < 0 or kd.ndim != 0:
        raise ValueError('weight must be nonnegative; KD must be scalar')
    if task == 'N' and weight != 0:
        raise ValueError('N must have exactly zero KD weight')
    _finite_float(native_total, 'native_total')
    _finite_float(kd, 'kd')
    if native_total.device != kd.device:
        raise ValueError('native and KD device mismatch')
    return native_total.sum() + actual_batch * weight * kd
```

## 附录 B：本地内核验证与使用

配套包包含 `reference_kernels.py`、`test_reference_kernels.py`、`kernel_test_results.json`、原始 pytest 日志和本 MD。使用已具备 torch/pytest 的环境执行：

```bash
python -m pytest -q test_reference_kernels.py
```

本轮实际结果：

```json
{
  "scope": "local CPU synthetic kernels only; no trainer, data, geometry or AP validation",
  "python": "3.13.5",
  "torch": "2.10.0+cpu",
  "device": "cpu",
  "tests": 18,
  "failures": 0,
  "errors": 0,
  "skipped": 0
}
```

本地 CPU 版本与服务器 CUDA 构建不是同一个执行环境。这些18项测试不能替代 pinned CUDA 环境的真实数据、梯度、RNG、模型生命周期和几何验收；本文没有声称这些真实检查已通过。

[P1]: https://github.com/yudongfang-thu/rgbir/blob/7bfaf2fe837d2ddfea4b0bde4e7ede0a21434af9/TASK_CONDITIONAL_STATUS_20260907.md
[P2]: https://github.com/yudongfang-thu/rgbir/blob/7bfaf2fe837d2ddfea4b0bde4e7ede0a21434af9/research_bundle/08_%E5%AE%9E%E9%AA%8C%E6%97%A5%E5%BF%97/2026-09-06_train_RGBIR%E5%AF%B9%E8%B1%A1%E5%88%A4%E5%88%AB%E8%92%B8%E9%A6%8F%E9%A6%96%E8%BD%AE/EXPERIMENT_PLAN.md
[P3]: https://github.com/yudongfang-thu/rgbir/blob/7bfaf2fe837d2ddfea4b0bde4e7ede0a21434af9/research_bundle/03_%E7%8E%B0%E8%A1%8C%E5%B7%A5%E7%A8%8B/SpaceNet6_OTD_official_reproduction/experiments/rgbir_task_conditional_v1/localization_loss.py
[P4]: https://github.com/yudongfang-thu/rgbir/blob/7bfaf2fe837d2ddfea4b0bde4e7ede0a21434af9/research_bundle/08_%E5%AE%9E%E9%AA%8C%E6%97%A5%E5%BF%97/2026-09-07_train_TaskConditional%E9%A6%96%E8%BD%AE/EXPERIMENT_PLAN.md
[P5]: https://github.com/yudongfang-thu/rgbir/blob/7bfaf2fe837d2ddfea4b0bde4e7ede0a21434af9/research_bundle/03_%E7%8E%B0%E8%A1%8C%E5%B7%A5%E7%A8%8B/SpaceNet6_OTD_official_reproduction/experiments/rgbir_task_conditional_v1/README.md
[R1]: https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_Localization_Distillation_for_Dense_Object_Detection_CVPR_2022_paper.html
[R2]: https://docs.pytorch.org/docs/2.13/generated/torch.nn.BCEWithLogitsLoss.html
[R3]: https://docs.pytorch.org/docs/2.13/generated/torch.nn.KLDivLoss.html
[R4]: https://openaccess.thecvf.com/content/CVPR2024/html/Wang_CrossKD_Cross-Head_Knowledge_Distillation_for_Object_Detection_CVPR_2024_paper.html
