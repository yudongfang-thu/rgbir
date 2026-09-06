# CGA-KD 预注册与代码快照独立审查（2026-09-05）

> 当前快照不应作为已经实现预注册 CGA-KD 的正式训练入口：G 没有 DFL 分布蒸馏；C 的正标量特征门控被损失归一化抵消；I 的目标与模块生命周期有关键问题。此结论是静态实现审查，不是方法效果实验失败。

仅新增本文件，不修改实现，不启动训练，不使用GPU，不联网服务器。代码来源为本审计目录 `train_cga_kd.remote_snapshot.py`；损失依赖 `E:/SHARE/光sar/03_现行工程/SpaceNet6_OTD_official_reproduction/yolo_osssl/rgbt_cmdistill_kd.py`；设计来源 `E:/SHARE/光sar/07_研究分析/方法预注册_CGA-KD_20260905.md`。下文 S=trainer快照，D=损失依赖，P=预注册。行号为本次读取版本。

## 1. 确定发现：G不是预注册的DFL分布蒸馏

- P L8/L23：H1/G明确定义为DFL分布蒸馏，G限制P4/P5。
- S L134–141：实际kd=terms.iou+terms.cls；L315回执也写iou/cls。
- D L114–120：raw DFL logits经softmax后只取期望偏移；L147–160解码xyxy；L176–208使用IoU与分类BCE。
- D L243–254使用全部预测anchor；S此分支没有P4/P5 anchor mask。

因此：这是CMDistill的逻辑项消融，不是DFL分布间KL/交叉熵，也不等于纯几何。具有相同DFL期望、不同bin分布的师生预测在IoU项上可以完全相同；保留class BCE又直接保留了教师类别/外观相关输出。不能用该臂检验“定位分布更抗负迁移/构造性排除外观冲突”。这不是命名小问题，而是estimand改变。

附带：调用cmdistill_terms仍计算PCCFD/SLRD，即使G随后不用其值；这不是错误梯度来源，但耗费不必要的计算/图构建。

## 2. 确定发现：C门控不改变设计中的KD权重

S L175–184只做teacher_feats[i] *= a_i，a_i=sigmoid(-2+4*darkness_i)>0；boxes/scores没有变化，也没有把每图loss乘a_i。

D L68–77对每张图展开feature做Pearson；若a>0且不触发epsilon/degenerate分支：

`corr(s,a*t)=<s-mean(s),a(t-mean(t))> / (||s-mean(s)||*a||t-mean(t)||)=corr(s,t)`。

D L93–111对每个空间token做L2 normalize再构造cosine affinity，因此 `A(a*t)=A(t)`。D L247–248的box decode只从feats取shape生成anchor；boxes/logits本身不变，IoU与cls也不变。

**结论：在非退化输入、忽略有限精度/epsilon分支时，C总KD损失及对学生的梯度与未门控L完全相同。** 若跑出差异，不能先归因于条件门控，应检查浮点误差、seed、输入顺序等。此结论由代码代数得到，无需GPU效果实验。极小向量触发clamp时可能不完全抵消，但那不是预注册的可靠性剂量机制。

另有两个确定偏差：

- P L25写可学习 λ(x)=σ(w·ĝ(x))；S L177是固定常数−2和4，没有Parameter/optimizer中的可学习w，且只有图像均值，没有噪声/ISP模型。
- S L121–123由batch最大亮度归一化；同一图的gate依赖batch里其它图，不是固定逐样本条件。batch=1且非黑图时darkness约0，所有图gate约sigmoid(−2)=0.119。不是按冻结亮度阈值78.283工作的条件函数。
- P L11写特征仅P4/P5；C将三尺度全部送入D L89，PCC包含P3。

## 3. 确定发现：I的对抗目标不等于g不变/h保留配对信息

P L9要求“h判别paired/shuffled、g不可判别”。S L151–160却：

1. h_p/h_s输入disc之前也经过GRL（L152–153）。梯度反转使h学习破坏paired判别，而不是保留这种信息。
2. g分支既用GRL，又把paired标签设0/shuffled设1（L157–160），与h分支标签相反，且共享同一个disc。g与h没有统一的判别目标，不能解释为对同一配对可识别性进行标准验证。
3. student_emb没有detach（L148、L152–158），因此学生还受到帮助该共享disc分类的梯度，并非只通过g的alignment获取KD（与L162注释矛盾）。

以上是确定的梯度路径/标签事实。**不应进一步断言某个最终collapse已经发生**，本次没有读取I训练实测。即使修复生命周期后，这些loss仍不能直接支撑预注册g/h机制。

## 4. 静态路径确认、尚无参数ID实测：I首次forward新增模块会漏optimizer

- S L107初始化aux=None；L110–116在首次真实teacher/student KD forward中才创建并host_model.add_module。
- S L227–233仅装criterion，不预建heads；全文件无build_optimizer覆盖或optimizer.add_param_group。
- 主审计补充了94当前安装包 `ultralytics_trainer.remote_snapshot.py`（本审计目录）：L320先set_model_attributes；L402调用_build_train_pipeline，其L300构造optimizer；L404创建EMA；L494才第一次真实training forward。L1124–1135采集构造时已有模块参数。

94当前安装包的顺序现已静态确认：正常训练路径下g/h/g_proj/disc在optimizer构造时尚不存在，首次forward注册以后也没有add_param_group，因此漏参；EMA同样在aux前创建。漏参会使step不更新aux，optimizer.zero_grad也不清理其梯度。**这是当前源码执行顺序支持的高置信实现风险，尚未运行真实参数ID覆盖/更新断言；现有canary已在P5维度处更早崩溃，不能声称已观察到“训练若干步而aux不更新”。** 不宜仅凭finite loss的canary声称可学习分解已正常训练。

所需最小CPU验收：所有requires_grad aux参数ID必须属于optimizer组；一次step后g/h/disc各自至少一项参数更新；optimizer.zero_grad后aux.grad应为None/0。该验收尚未运行，本地默认Python无torch（导入报ModuleNotFoundError），未为此安装环境。

## 5. 已发生的工程失败：同一P4 head复用于P5导致通道报错

S L113–115从P4 channels创建唯一InvariantHeads；g第一层Conv输入固定P4通道（L86），g_proj输出也固定P4学生通道（L88）。L164–169随后对P4/P5复用同一g/g_proj。

本地标准YOLO11n配置 `E:/SHARE/光sar/06_历史工程_只读/CoRe-LADD/shared/yolo/ultralytics/cfg/models/11/yolo11.yaml` L11宽度0.25，L44 P4基宽512、L48 P5基宽1024、L50 Detect三尺度，即P4=128/P5=256；该版本Detect `nn/modules/head.py` L144返回原始feats=x。

主审计补充的94真实日志 `canary2_invariant.remote_snapshot.log` 已证实：Detect输入通道为[64,128,256]（L29），trace定位trainer第166行（日志L65），最终报错 `expected input[32,256,20,20] to have 128 channels, but got 256 channels instead`（日志L81，按rg行号；ANSI进度回车可能影响查看器行显示）。这是已发生的首次forward工程失败，早于optimizer漏参的训练后果，不是I方法的科学负结果。即使修正g输入，g_proj固定输出128与P5学生256的问题也必须一起处理。

## 6. 科学设计风险（与确定代码错误分开）

### 配对不变不等于模态不变

训练pair discriminator区分(T(x_i),S(x_i))与(T(x_j),S(x_i))，依赖的常常正是共同场景、目标类别、几何和实例信息。将g的“正确配对不可判别”作为目标可能抹去要迁移的共同实例信号，而不是仅抹去RGB/IR私有外观。模态不变一般要求跨模态对应表示相容，不要求对应与不对应样本无法区分。当前H2需要重新明确对何种变换不变、保留何种任务信息；不能靠一个GRL宣称“保证”。

### 不足以识别分解

S L91–94、L148–169将空间图GAP成全图向量，丢弃目标级几何；h没有重建或任务约束，只有cosine正交不能保证其有意义，也不能排除分支收缩/任意旋转。“g/h代表不变/特异子空间”仍是假说。

### 归因矩阵不足

P L19–33没有same-modal KD主臂，只有native/full跨模态L、G/I/C和晋级后shuffle；这重复历史“paired>shuffle但输same-modal”的漏洞，也未覆盖用户AGENTS.md强制自模态增强基线。最终只要求完整方法>L不足以确认跨模态独特性。P L33“显著弱于”未冻结统计检验/效应阈值；L30–32的+0.3/+0.5需统一明确AP百分点评分还是0–1数值。

### 依据本身尚不支持强措辞

P L8“构造性排除”、L10“可消除”不是现有实验证据支持的结论；day/night是亮度中位数代理、P3对照recipe混杂。P2 native checkpoint可能误指VEDAI的交叉问题已提交主审计，应在依赖该probe排除P3层前完成修正。

## 7. 其它可复现性限制

- S L235–239把validate/final_eval置空，L299 val=False；训练receipt只有KD均值，不能直接裁决预注册full/day/night AP门。需要事前固定外部endpoint评估；不能把completed receipt当AP验收。
- S L312总写status=completed，即使max_steps只跑canary；没有max_steps/epochs_completed明确区分，容易把短测当formal完成。
- S L333–335仅绑定自身脚本为loss，未将实际依赖rgbt_cmdistill_kd.py计入losses hash；依赖漂移可能改变效果且receipt不反映。
- S没有shuffled-G/shuffled-I训练臂选择；invariant内部roll仅是辅助判别训练，不能替代完整训练轨迹的shuffled归因对照。

## 8. 裁决与最低下一步

**裁决：IMPLEMENTATION_NOT_MATCHED_TO_PREREGISTRATION，不能由这些快照运行读数验证H1/H2/H3。** 优先顺序为：先确认/修正已发现实现身份，再做CPU代数与梯度/optimizer/shape验收，再登记有理由的预注册偏差，最后才评估是否值得授权方法训练。当前审查未改代码、未替用户启动任何修复或训练。
