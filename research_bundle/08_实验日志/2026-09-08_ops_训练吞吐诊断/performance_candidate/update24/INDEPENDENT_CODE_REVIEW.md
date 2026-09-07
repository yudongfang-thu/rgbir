# 独立代码审阅：24 次成功更新诊断

结论：**PASS_FOR_BOUNDED_24_UPDATE_DIAGNOSTIC**。2026-09-08，审阅者 baseline_feature_analysis；作者 ap_error。仅准许在原全局 lease 下执行此受限旧/新 C1 比较，不是生产替换或 E200 长训准入。

只读对照了 `compare_24_updates.py` 与本地现行 `train_independent.py`、`runtime.py`、`independent_criterion.py`、`selection_adapter.py`、`verify_compatibility.py` 和冻结 `legacy_oev1/train_object_evidence.py`。未改 release，未使用 SSH/GPU，没有计算 hash。

- 两个独立解释器均走原 `historical=False`、Tracked 双标签训练集和原 IndependentCriterion 的私有子类；函数 globals 副本只更换池化绑定及审计包装。原模块绑定不变，原初始化、教师/参考前向、GT、资格、排序、分母、AMP、优化器和 EMA 路径保留。
- 固定 C1 seed42、B32/workers4、E200 配方及 λ=`0.09227393550836771`。24 次是实际 `optimizer.step` 调用，另核真实成功/跳步计数，最大 96 次尝试；不是把 AMP 跳步算更新。
- 初始模型/优化器/EMA/scaler/RNG、逐批元数据与双标签及 worker RNG、首批真实图像、逐次完整参数/梯度/优化器/EMA 状态均有独立比较。辅助参数期初/期末不变，并要求实际 KD 分数梯度有限非零。全 24 批像素不保存，边界已声明。
- 选择身份/控制精确性、中间浮点、loss/gradient/update 的容差与字节精确性分开汇总；原 teacher_delta 超容差记录不会被 exact 最终轨迹掩盖。
- CUDA 同步区间内全状态复制/保存/审计耗时单列并扣除；去掉前 6 批热身。该批次时间不含 loader fetch，且审计会改变重叠，因此不是未插桩整 epoch 吞吐。

独立在 CPU 再执行作者的 16 项真值测试，全部通过；包含零参数 super 的 closure 与原 globals 保留、固定容差方向、NaN/有符号零、缺失 attempt、选择身份与参数差异拒绝、中间误差与轨迹分层。实际回执：`independent_cpu_review_attempt1.json`（torch 1.8.0+cu111，CUDA 未初始化）。未在本地假装执行真实 trainer。

仍须由真实受限运行确认 pinned 环境导入、资源峰值及 24 更新结果；所有数值差异原值保留，不改容差。此审阅不保证未来 selected-only 候选，因为其尚是另一实现。
