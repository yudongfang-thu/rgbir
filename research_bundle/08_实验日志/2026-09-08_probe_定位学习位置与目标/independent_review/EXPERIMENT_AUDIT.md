# LLVIP 定位位置与目标：独立证据审计

**PASS，范围仅为既有训练缓存的身份和代数诊断。** 实际审阅 agent 为 `/root/loc_target_review`，可观测模型身份 unavailable；没有声称跨模型审阅。未运行GPU、加载模型、重新前向或训练，未读取test，未计算hash。

核验了原校准8批、首32图的原始见证及桥接、实际L2/criterion/calibration源码，再对照最终 `anchor_join/output_attempt3` 与 `target_audit/output_attempt1`。三份源码与本轮94捕获副本逐字节相同；这是源码字节比较，不是hash证明。完整字段与输入stat清单见 [EXPERIMENT_AUDIT.json](EXPERIMENT_AUDIT.json)。

## 实际复算结果

- 原79条历史base记录的GT身份闭合，426次正native匹配的框、预测ID和anchor链接闭合；最终作者80条anchor记录与原值一致。
- 首批11个定位机会的native S/R匹配anchor都等于历史R候选anchor；其中4个真正selected，学生学在该R位置。4个教师目标anchor均不同于教师native匹配anchor。首批总量仍为80 GT、79 base、7 selected。
- 八批649 base、30 selected，256个不同图像路径，selected涉及26图。全八批没有稳定原生GT行追踪，不能将30称为去重物理对象。
- 全30条教师目标、120条边与独立SmoothL1公式一致：47条同向饱和且导数相同、63条同号不同量、10条反号。stack cosine为0.958663；按base分母为0.959681；换像素坐标并按Bλ缩放为0.980442。
- 原参数梯度记录的活跃批L2-box/native范数比中位1.261627%，native cosine为4正3负。参数范围只有原24个model.16/model.19张量，未扩称全模型梯度。

独立复算保存于 [CPU_RECOMPUTATION.json](CPU_RECOMPUTATION.json)，逐对象对照保存于 [AUTHOR_COMPARISON.json](AUTHOR_COMPARISON.json)。另核30个保存教师目标均在对应R anchor可表达的DFL期望坐标范围内（stride单位0.089382至8.940414）；这不代表原生TAL正样本归属或物理配准。

## 必须保留的语义边界

实际L2传递的是GT归一化框坐标，用SmoothL1 β=.1；它不是完整DFL分布蒸馏。分母取教师门之前的base，caller再乘Bλ；无效对象不能补成selected，未执行教师后门的null不能当失败目标。

保存R输出处的导数是代理。原student标量loss与R代理最大差约5.354e−9，这不足以证明student逐坐标、raw DFL或参数梯度相同。原缓存没有完整student输出和Jacobian，也没有box-vs-GT参数梯度cosine。相近导数不是“教师无新信息”的证明；10条反号边也不是实际负迁移的证明。

649条RGB/IR GT框相同只证明共享标签。坐标映射在数学上恒等，但32条已计算目标有2条浮点重构非exact，最大1.52587890625e−5像素。最初审阅者误用全exact断言，失败和修正保存在 [INITIAL_RECOMPUTATION_FAILURE.json](INITIAL_RECOMPUTATION_FAILURE.json)；最终作者已正确披露且使用原mapped值。

暂缓把本版L2仅靠放门或加剂量继续扩训，是现有证据下可接受的保守决定。**本诊断没有识别旧FT3负差的原因，没有证明放门/加剂量不可能有效，也没有否定所有定位蒸馏。** 617条base尚无教师目标；对其效果没有测量。

## 执行与来源限制

anchor首两次attempt在进入run前出现解析失败，原文件与失败记录保留，最终显示字符串Unicode转义前后AST独立验证相同；只验收完成的attempt3。目标侧验收attempt1。作者真值5+5项记录存在；审阅者另做3项有界代数真值与实际全量小表复算。

eval_type：标注GT相关计数/残差为real_gt，R输出导数是确定性模型输出代理；审阅者代数真值为synthetic_proxy。本审计不形成新AP、泛化或KD增益结论。

按本轮明确边界，`audited_input_hashes` 记为 **not computed**；没有因技能要求擅自补hash。完整输入范围、命令和不可得字段见JSON；原提示、响应与审阅时间线见 [REVIEW_PROMPT.md](REVIEW_PROMPT.md)、[REVIEW_RESPONSE.md](REVIEW_RESPONSE.md)、[REVIEW_TRACE.json](REVIEW_TRACE.json)。
