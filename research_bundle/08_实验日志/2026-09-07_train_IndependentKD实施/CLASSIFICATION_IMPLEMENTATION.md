# 独立类别蒸馏实现回执

**结论：C1/C1_y 纯核及 C0 选择 adapter 已实现，27 项本地 CPU 合成/回归测试通过；旧 C0 的 scalar、完整 stats 和 raw-score 梯度在测试输入上精确一致。尚未进行真实 GPU batch、完整训练轨迹、梯度校准或 canary 验收，不能据此准入 E200。**

日期：2026-09-07。范围：仅新增 `rgbir_independent_kd_v2` 下三个分类文件；未修改原 OEv1、定位、trainer、calibrator，未 SSH、未使用 GPU、未启动长训。不写文件哈希。

## 实现和接口

源目录：`03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_independent_kd_v2/`。

- `selection_adapter.py`：优先使用同目录 `task_conditional_reference/legacy_oev1/object_evidence_loss.py` 的冻结源码；该目录不存在时才回退相邻旧 v1。本地纯 torch 模块不 import runtime/trainer/Ultralytics。
- `classification_logit.py`：逐对象/尺度/类别 raw 相对 Delta 的 Bernoulli KL，C1 η=.25、C1_y η=0，教师 clip ±16 后除 T=2，学生同温度且不 clip，tau²，分母 pre-teacher E_C，前景/背景保留学生梯度，教师/参考 detach。
- `test_classification_logit.py`：unittest，无 pytest 依赖；本地 PyTorch 1.8.0+cu111 仅 CPU 执行，SciPy 1.10.1。服务器 pinned CUDA 环境复验由统一集成流程负责。

```python
selection = build_classification_selection(
    student_raw, teacher_raw, reference_raw, batch,
    strides=(8, 16, 32), config=evidence_config, selection_seed=0)

# No actual_B or lambda multiplication inside either function.
loss, stats = classification_loss_from_selection(
    selection, temperature=2., raw_teacher_clip=16.,
    off_target_weight=.25, return_records=False)

# Convenience wrapper; a supplied selection is checked against this exact
# source prediction, label tensors, batch identity and tensor mutation versions.
loss, stats = classification_logit_loss(
    student_raw, teacher_raw, reference_raw, batch,
    config=evidence_config, selection=selection, off_target_weight=.25)

# All returned loss components are differentiable, unweighted by global lambda.
parts = classification_loss_components(selection, off_target_weight=.25)
# parts: loss, target_loss, off_target_loss_unit, off_target_loss, c0_loss
```

`ClassificationSelection` 返回 `student_delta/teacher_delta/reference_delta [M,L,C]`、`valid_levels`、`selected`、`labels`、`base_to_matched`、`matched_to_base`、`selected_matched_indices`、`base_object_ids`、q/eligible、各模态 region masks、可导 `c0_loss` 和原 C0 完整 `c0_stats`。

原选择调用 frozen `_match_objects/_evidence/_has_candidate/_choose`，没有以 C1 内容重新排序。C0 scalar 按旧 helper 的输出及原运算顺序复建，供校准和回归。全 matched → E_C 紧凑序号 → selected 三者显式映射；teacher-correct 失败但已属 E_C 的对象继续留在分母。

## 本地验收

最终通过日志：`classification_cpu_tests_attempt6.log`，27 tests、0 failures、0 errors；25 项阶段回执保留，不覆盖。

最后补充：C1_y 的 clipping/entropy 统计只计实际监督的 GT 通道；前景重叠统计只计 selected 且 valid 的尺度；传入不同 evidence config 的缓存复用被拒绝。新增两项统计边界测试，attempt5/6 日志保留。

覆盖：旧 C0 完整 stats/scalar/score-gradient 精确一致；matched/base/selected 压缩映射；teacher rejected 仍留分母；稳定 quality tie；单有效尺度；各模态自身前景和排除全部 GT 的背景；无匹配与无选择可导零；学生值不改选择；非目标通道干预仅改变 C1；单类 C1=C1_y；双方温度和 KL(T||S) 解析梯度；clipping 顺序、极值稳定性；有效尺度 divergence 先算再平均；目标系数及非目标平均；不同 batch/原地变更缓存拒绝；最后小 batch 不在内核重复乘 B；stats 可 JSON 序列化。

专门增加数学反例测试，证实在共享参数上删除非目标项可能增大合成梯度范数，因此日志不宣称 C1_y 剂量必然降低。

保留技术历史：第一次 23 项中一个 fixture 的 4px 位移未跨过 P3 网格，导致“区域必须不同”的测试断言失败；改为合法匹配且跨网格的 6px 位移后 23 项全通过（attempt2）。新增两项后，8px 小框的外环只有三个背景点，不满足原 minimum_background=4，单有效尺度 fixture 失败（attempt3）；调整为真实满足一层有效、另一层背景不足的 10px 框后 25 项全通过（attempt4）。两次均为测试输入假设错误，不改任何选样规则。第一次控制台原文未独立保存，原因在此登记；attempt2/3/4 日志均保留。

## 统计语义与未完成边界

- 输出明确标记 relative evidence 不是 detector confidence；原 C0 scalar stats 仍存在，新增 C1 分尺度、分目标/非目标损失与 per-class 组成。
- 对 GT 前景中其它 GT 覆盖比例做只读统计，未增加门控。
- 校准可用同一 Selection 的 C0、C1 及目标/非目标分量；共享梯度方向和合成范数应由父流程的固定参数集合测量。
- 尚未证明真实 E200 trajectory / RNG / optimizer / EMA 等价，未测实际显存和进程 RSS，未完成 64 批 C1 校准及 C1/C1_y 各 24 次成功 optimizer update。
- 该实现本身没有形式训练放行权限，也不写 λ 或更改旧配置。
