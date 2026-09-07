# C1/C1_y 与固定梯度观测的独立代码复核

结论：在本次明确审阅范围内没有剩余代码阻塞，接受 C1/C1_y 的分类损失、冻结选择适配器、criterion 分派和固定梯度观测实现。独立复跑 27 个分类测试、10 个梯度 helper 测试，加上两个独立 criterion 集成测试，共 **39/39 CPU 测试通过**。这不是实际模型/GPU训练通过回执，也不升级任何检测收益主张。

## 已核对的协议

- C1 直接复用冻结 C0 的独立 RGB/IR GT 匹配、区域有效性、冻结 R 粗候选、T 正确性、质量 q、rho 与稳定同分顺序。全类别载荷和当前学生输出没有回流改变选择集合。
- E_C 是教师质量筛选前的基础集合，分母为 max(1,|E_C|)。每个对象按有效 P3/P4 尺度平均；先算逐尺度、逐类别 Bernoulli KL，再平均尺度，没有先平均 logits。
- 教师原始相对差值先 clip 到 ±16，双方再除温度 2；教师与参考 detach，学生不 clip。学生前景和排除全部本模态 GT 的背景环均保留梯度。
- 目标类系数 1；非目标类先取平均再乘 eta=.25；没有再除 1+eta。C1_y 只将 eta 设为 0，保留目标项、集合、分母和全局 lambda。单类数据退化为目标类项，不虚构多类别知识。
- criterion 总损失仍为 native.sum()+实际 B×lambda×唯一 KD。C1 没有叠加 C0；N/C0 立即返回原父类路径，未进入新 observer。
- 固定 epoch 0/10/50/100/199 的首个批次被观测，即便 KD 为零也记录并不换批。只调用 autograd.grad，不调用优化器、GradScaler step 或 EMA update；不修改 Parameter.grad、参数或 buffer。返回记录仅为 JSON 数值，不保留计算图。
- 观测参数集合为检测头输入的前两组独立 P3/P4 特征模块，与当前校准器的明确参数集合一致。KD 使用真实 B×lambda 缩放一次，原生梯度使用其已经定义的 native.sum()。
- 非目标项的删除可能减少或增加合成梯度范数，取决于方向抵消；实现与测试没有把“删项一定降低总梯度”当定理。零梯度和非有限观测均单独记录，不用 epsilon 虚构比值。
- 配对选择调用稳定排序，不使用全局 RNG；observer 不新增 forward，不推进 Python/NumPy/PyTorch CPU RNG。CUDA RNG/实际 AMP 更新仍由真实 canary/轨迹回执确认。

## 独立集成验证

日志中的 `criterion_integration_review_test.py` 使用真实 criterion、真实选择/损失和真实 observer，仅把外部 Ultralytics runtime、模型输入和旧基类替换为明确标注的 CPU 合成夹具。

对 C1 和 C1_y，比较开启/关闭观测时的 total 和最终 backward 参数梯度，逐值相同；既有梯度观测不写入 .grad、不修改参数/buffer 或 CPU RNG，optimizer/EMA计数不变。另核对 N/C0 分派确实直接到父类且没有 observer。本测试不冒充 N/C0 六条真实训练等价轨迹。

## 范围外与仍需真实证据的项目

- 正式 readiness 仍需要 seed 0/42/123 × N/C0 的六条真实兼容轨迹、固定自然 64 批有效校准及真实至少 24 次成功 optimizer update 的 canary；这些正在运行或待运行，不列为本范围的“代码阻塞”。
- 本复核不证明 C1 优于 C0，不证明非目标知识有益，不证明负迁移消失；只能交给预注册实验判断。
- 过去 L 规格审阅与五对几何目视结论仍独立保留。五对 UNKNOWN 不是配准失败。L 的真实几何点集/D2/梯度覆盖没有被本 C1 代码接受替代。
- `localization_adapter.py`、`geometry_support.py`、`coverage_probe.py` 和 `resource_dispatch.py` 是本审阅者参与编写的模块；本回执**不对它们作独立自审接受**。resource 的接受由根执行者与另一审阅者负责。
- 本次只接受下列四个当前源码副本：classification_logit.py、selection_adapter.py、gradient_observation.py、independent_criterion.py。冻结 C0 oracle、当前参数集合校准代码和 runtime 仅作为上下文核对，不借此签署整个训练器。

机器可读回执与精确源码副本位于 `classification_gradient_independent_review_v1/`。该回执可与其他独立审阅按各自范围合并，不能单独冒充全工程 readiness。

