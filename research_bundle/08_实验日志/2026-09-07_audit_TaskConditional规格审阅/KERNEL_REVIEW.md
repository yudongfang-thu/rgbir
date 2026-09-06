# Task-Conditional 规格的内核与接口独立审阅

**结论：附录 A 的 KL 算子与总损失组合没有发现维度、方向或剂量错误；可以作为纯算子参考。实施前仍须冻结 GT-DFL 温度/边界合同，并通过同 anchor 的分布质量诊断。CPU 通过不代表真实几何、selector 或 trainer 已经实现。**

## 范围与实际执行

- 原文：`07_研究分析/RGBIR_Task_Conditional_KD_Codex_Spec_20260907.md`，重点 §6–9、§12.4、§13 与附录 A。
- 从 Markdown 的附录 A 代码块**原样提取**到 `verbatim_appendix_a.py`；未改参考内核、原文、现行训练代码或任何历史 run。
- 实际环境：`D:\Anaconda\envs\KGJ_proj\python.exe`，Python 3.8.0，PyTorch 1.8.0+cu111；本次设置 `CUDA_VISIBLE_DEVICES=''`，只用 CPU 合成小张量，线程数1，未加载权重/真实图像，未启动 GPU。
- `verify_reference_kernel.py` 是本次独立验证脚本；结果为 `kernel_validation_local.json`。14 项通过，1 项 CPU bfloat16 autocast 因此环境 API 不存在而未验证。未声称与文档附录 B 的 PyTorch 2.10.0+cpu 环境一致；未验证服务器或 GPU AMP。
- 框架接口核对使用已有现场源码镜像：`08_实验日志/2026-09-06_ops_GitHub完整审计包/remote_snapshot_20260906/framework_snapshot/ultralytics/utils/loss.py` 与 `tal.py`，版本目录证据为 Ultralytics 8.4.115。本次没有重新导入现行服务器环境，不能把镜像核对称作最新现场运行验收。

## 1. 内核正确的部分

1. `[B,4R,A] → [B,A,4,R]` 的 reshape/permute 对应边优先、bin 次之的原生布局；Softmax 在最后的 bins 维计算，未混入类别/空间维。
2. `F.kl_div(log p_S,p_T)` 为 `KL(p_T||p_S)`，四边均值和 `T²` 与正文 `T²/4 Σ_d KL` 一致。
3. teacher 与 anchor gate detach；student 保留梯度。独立解析梯度为 `w*T/(4*N)*(p_S-p_T)`，最大核验误差 `3.7253e-9`。
4. 每对象 anchor 权重由调用方提供，函数除以固定传入 normalizer，不除通过权重和。复制同一对象的 anchor 并均分权重后 loss 不变；将 gate 减半，loss 恰好减半。
5. 零 mask 返回张量标量和零学生梯度；非有限输入即使 gate 全零也被拒绝。
6. 输入 float16 时转换 float32 后计算，输出 float32，反向到 float16 叶子仍有限。此检查只证明 CPU 下混合 dtype 的内核行为，不能替代 CUDA AMP+GradScaler 的实际 batch 验收。
7. `combine_detection_loss` 先 `.sum()` native，再各加入一次 `B*λ*KD`。测试 native `[1,2,3]`、C=2、L=3、B=5、λ=.1/.2，得到10而非广播错误的18；native梯度全1、C梯度.5、L梯度1。

独立的 Python `math` 标量 KL oracle 得到 `0.99598337546`，参考实现 `0.99598342180`。测试同时覆盖不同 side 与 anchor，避免只用相同分布掩盖布局错误。

## 2. 原生接口与支撑范围

现场镜像 `loss.py` 的 `v8DetectionLoss` 使用 `preds['boxes'].permute(0,2,1)`，再 `view(B,A,4,R).softmax(3)` 解码；`make_anchors(...,0.5)` 和 per-level stride 建立预测坐标。`loss()` 返回 `loss * batch_size` 与 detached items；因此附录的组合式符合此镜像版本，不能再把 native 整体乘 B。

原生 `DFLoss.__call__` 会**就地**把目标裁剪到 `[0,R-1-0.01]` 后计算两邻近 bins 的加权交叉熵；`bbox2dist` 若传 reg_max 也会裁剪。规格要求 L 的超支撑对象拒绝、不静默 clamp，与保留原生检测 loss 不矛盾：这是新增 KD 合法集合的额外条件，不应据此修改原生 loss。

实施时应显式冻结：

- 几何可表示范围是 `[0,R-1]`，还是为了与原生 GT-DFL 完全一致采用 `[0,R-1-0.01]`；边界如何处理。
- 先以未 clamp 的真实距离检查 RGB 与 IR GT 支撑，再构造控制目标；不要调用会 clamp 的 helper 后才“检查”支撑。
- d 为整数（包括最末 bin）必须单独正确处理，不能照非整数的 floor/ceil 权重式得到0总质量或越界。
- 若复用原生 `DFLoss`，避免其就地 clamp 修改共享的 target；C/L/诊断不得互相改变 tensor。

这属于 §8.4/§12.4/§13 尚需实例化的合同，不是附录 A 没有自行检查 GT 的 bug：附录根本不接收 GT/几何，这些正是调用方责任。

## 3. GT-DFL 控制的温度必须明确

§12.4 正确要求明确温度，但尚未给唯一可执行语义。若固定 GT 两 bin 分布 q 直接用于 `KL(q || softmax(z/T))*T²`，最优点满足 `softmax(z/T)=q`，所以原生解码使用的 `softmax(z)=q^T/Σq^T`。

本次实际反例：GT距离 d=4.7，q=(.3,.7)。T=2 且 target 不变时，原生最优分布是 (.155172,.844828)，期望距离变为 **4.844828**。这已经不等于原生两 bin GT 的目标。

可采用以下两种清楚命名的方案之一，事前冻结：

- 作为严格原生 GT 重监督，使用 T=1 的 GT-DFL CE/KL；记录它与教师 KD 的梯度尺度差异。
- 如需使用同温度 T，则将 GT target 同步软化为 `q_T=q^(1/T)/Σq^(1/T)`，并保持 T² 缩放。这样原生最优仍是 q；实现需正确处理 q=0，不能用未说明的 epsilon 扩大支撑。

两种控制的优化干预不同，不能在看到 AP 后悄悄切换。此处是未冻结细节，不能描述成附录 A 的实现错误。

## 4. 期望框更好，不足以证明 DFL 分布有益

D1 与 D2 的区别、同物理 anchor 检查都很必要，但选择器目前按**期望框 IoU**比较 T/R，实际 L 迁移的是**完整距离分布**。即使几何已完美对齐，期望相同也能对应形状完全不同的分布。

本次合成反例（四边采用同样的分布，T=2）：

- GT 距离4；教师均值3.996、参考/学生均值4.189，教师点估计明显更近 GT。
- 教师主要把概率放在 bin2 与 bin6，参考/学生主要放在 bin4与bin5。
- 教师 KD 与原生 GT-DFL 梯度内积为 **−0.03100**。
- 沿 KD 梯度做0.01单步，原生 GT-DFL CE 从 **0.2231435 上升到0.2234537**。

这是“教师期望框优势不保证原生 GT-DFL 梯度有益”的可执行反例，不是数据集 AP 结果，也不说明分布KD普遍无效。它支持在 D2 中增加只读的：T/R 对 RGB GT 的 DFL CE、预测分布形状/熵、共同参数上的 KD/native 梯度夹角记录。不要把这些统计未经预注册直接加入首版 gate。

因此优先 DFL 是一个需要数据支持的设计选择。若已有几何只能支持 box 映射或 DFL 质量难以支持，规格 §8.5 的独立 box KD 路径仍然有价值；应另命名并冻结，不能同一 run 静默切换。

## 5. 尚未验收、不可由内核测试推出的事项

- 几何证据、同物理 anchor/stride/bins、唯一 owner 和 GT 持久 ID。
- E_L 是教师质量筛选前集合、A_i^0 先于教师筛选，且 selector 不读取当前 S。附录 docstring 的 “eligible anchors” 应在实现说明中统一解释成**预选 anchors**，否则多anchor扩展容易错除为过滤后的数。
- N/C 与现行 OEv1 的初始化、数据流、RNG、BN、EMA、梯度和预算等价。
- `combine_detection_loss` 的输入有限性与 device 等仍需外层 criterion 检查。现行 `EvidenceCriterion` 已在组合后执行 finite 检查；仅给 helper 增加几条 assert 不替代完整验收。
- 当前 raw 阶段是否真正存在可靠定位优势；已保存的 NMS 框不能重建 DFL logits。
- GT-DFL 控制继承 IR mask/选择信息，不能叫完全 RGB-only；强同模态 teacher=R 会归零，需按正文使用有效独立教师。

首批可以安全完成独立算子与只读诊断；当前审阅没有提供开启直接 DFL GPU 长训的科学验收结论。
