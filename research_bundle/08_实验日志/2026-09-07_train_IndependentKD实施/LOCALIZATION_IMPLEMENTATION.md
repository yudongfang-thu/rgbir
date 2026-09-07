# L1/L_GT adapter、几何支持域及自然训练覆盖工具

> **已实现独立 L1/L_GT、双模态 GT 边界支持检查和真正使用 seed=20260907 的自然训练覆盖 loader；17 项 CPU 合成测试通过。尚未执行真实 64 批覆盖、几何补证、λ 校准或定位 GPU canary，不能据代码通过宣称 L READY。**

## 实施范围

所有新增代码位于 `03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_independent_kd_v2/`：

- `localization_adapter.py`
- `geometry_support.py`
- `coverage_probe.py`
- `test_localization_adapter.py`
- `test_coverage_probe.py`

优先读取本目录冻结的 `task_conditional_reference/`，该目录不存在时才回退到邻接的旧 `rgbir_task_conditional_v1/`。没有改旧定位/几何/数据流源码、阈值、既有结果或在跑任务；没有执行 SSH、GPU 或计算文件哈希。

## 与 trainer/calibrator 的接口

```python
adapter = LocalizationAdapter(contract_path, localization_config, require_verified=True)
kd_scalar, stats = adapter.compute(
    student_raw, teacher_raw, reference_raw, batch,
    strides=(8,16,32), task="L1", return_records=False,
)
# task 仅 L1 或 L_GT；不含 C；标量没有乘 actual_B 或 lambda。
total = native_total.sum() + actual_B * lambda_L * kd_scalar
```

`adapter.select(...)` 可单独获取旧 `LocalizationSelection`，其中 geometry mask 在全部基础集合 E 形成之前应用。随后仍是原 frozen R 选 anchor 和原 T/R 质量门，L1/L_GT 共用完全相同 selection/E/normalizer。当前 R IoU 上限仍是0.70，不在本实施中修改方法假设。

```python
loader = build_natural_loader(cfg, seed=20260907)
for batch in loader:
    # img / strong_img: CPU uint8，调用方仅转换一次。
    # teacher_batch 已存在，包含CPU normalized bboxes、cls、batch_idx。
    # 将tensor移入设备由calibrator统一处理。
    ...
```

返回值只有 loader，不是 tuple。`loader.coverage_metadata` 记录私有 generator 种子、实际 workers、shuffle/replacement/prefetch、数据集/recipe 来源。这个 loader **只供固定校准/诊断，不替换正式 N/C0 历史训练数据流**。

## 新几何支持的准确身份

版本为 `paired_gt_boundary_empirical_error_v2`：

1. 只读取原先独立接受的 exact-image contract；没有合同或非 verified 时，formal adapter 拒绝。显式诊断模式也只产生 false mask，不使用全true几何假设。
2. 对同类匹配 RGB/IR GT，分别调用原合同，要求两侧完整框通过同网格、实测误差、真实原图/输入矩阵及共同凸包检查。
3. 额外要求两侧框的边界邻域都被支持。邻域半径不是新增可调像素常量，而是既有 `max_error_raw_px + uncertainty_raw_px` 经增强矩阵最大奇异值传播的误差余量。邻域移出输入图像或共同点凸包即拒绝。
4. anchor 另外检查处于两侧 GT 和两套独立点凸包内；实际 P3/P4 stride 分别检查，之后旧 selection 再执行 DFL 未 clamp 支撑及全部 RGB GT 的 native 唯一归属。
5. 原六点、整图三象限、P95/maximum 容差和不外推规则保持原样；本次没有实现更宽松的局部象限合同。

边界范围加严属于新版本 geometry mask，可能减少 E。它没有在教师质量筛选后删除分母对象；L_GT 使用同样的新 mask。实现明确保存该支持域版本和分原因计数，便于区分旧/新诊断。

该算法仍是稀疏独立点支持下的经验近似，不证明凸包内部逐像素物理真值；它不从 GT IoU、模型相似度、同路径名或人工 `verified=true` 自动产生几何证据。

## 自然 64 批覆盖检查

部署后的 CPU CLI：

```bash
python coverage_probe.py --config /path/to/frozen_config.yaml --roster /path/to/frozen_roster.json --output /path/to/new_attempt
```

输出目录必须全新。CLI 固定64批、seed20260907，不提供换批/抽样候选补齐接口。读取真实 train YAML、实际 paired mapping 与原生增强；不加载教师/学生/评估器，不访问 test。

与旧 `build_dataloader` 的常量 generator 不同，本工具显式创建独立 `torch.Generator().manual_seed(20260907)`；随机 shuffle 为全训练集无放回排列，workers 的 Python/NumPy/Torch CPU 种子由该私有 generator 导出。workers=0 时变换拥有私有 RNG 状态，避免改变调用方 CPU RNG。构建阶段也保存/恢复 CPU RNG，没有用 `torch.manual_seed` 影响已存在 CUDA RNG。

每图记录：dataset index、worker ID/seed、RGB/IR 源文件、实际 RGB/IR augmentation matrices/trace、输出shape、两套增强标签、原 roster 精确源路径命中。路径匹配仅使用真实 canonical path/roster 提供别名；不靠 basename-only 产生授权。

`coverage_receipt.json` 中 `roster_hit_batch_upper_bound` / `roster_both_gt_batch_upper_bound` 是**这次自然流的上界**：即便名单中所有区域都通过几何，且每图有合格对象，也不可能在名单未出现的 batch 蒸馏。低于16时返回 `ROSTER_CANNOT_REACH_16`；达到16仅返回 `GEOMETRY_AND_SIGNAL_STILL_REQUIRED`，不能当作 D2/梯度已通过。

为了保持与真实多worker流程一致，DataLoader仍预取，可能在进程中变换64批以外的样本；正式统计严格只消费/保存前64批，既不替换也不重抽。主训练正式 data seed 仍由既有协议管理。

## 验证

实际本地环境：Python 3.8.0，PyTorch 1.8.0+cu111，测试仅CPU。最终12项定位/几何测试和5项覆盖测试均通过。

定位测试覆盖：固定anchor梯度仅落到DFL；T/R detach；L/L_GT同E/anchor/mask；新额外几何不拒绝时与旧损失逐数相等；学生值不参与选择；空集合可导零；联合任务拒绝；IR框独立越界拒绝；边界误差邻域；anchor双框约束；未知图/不同矩阵/NaN/负人工不确定度拒绝；共同翻转网格接受。

覆盖测试覆盖：同seed样本与增强一致；不同seed真实改变样本排列；前16张无重复抽样；workers0不消耗调用方CPU RNG；双worker分别播种且图像增强可复现；实际命中计数为上界且不伪报verified；已有目录拒绝覆盖。

初次两项测试因旧本地Torch没有 `torch.testing.assert_close` 报测试API错误，改用 `torch.equal` / `torch.allclose` 后通过；不是方法损失错误。真实94 pinned环境仍须复跑这17项及全链路检查。

## 待根任务执行

- 统一部署并在94 CPU运行两个数据集自然64批覆盖；不要由本子任务另开SSH/作业。
- 据确切上界决定几何审核量与范围；没有通过不启L长训。
- calibrator复用导出的自然loader并保持每batch训练模式R副本/BN重置规则。
- 真实T/R、增强后D2、非零剂量、24次成功更新canary全部通过后才形成正式定位READY。
