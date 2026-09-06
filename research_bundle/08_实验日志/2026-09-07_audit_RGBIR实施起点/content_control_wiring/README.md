# C 内容对照接线独立审查（2026-09-07）

> **结论：最新版 `c_shuffled` / `c_same_modal` 接线未发现阻断各24次成功更新 canary 的问题。两臂通过实际 TaskCriterion 的CPU接口合成检查；此结果不是实际模型、AMP或GPU canary通过，也不构成正式长训放行。**

## 目的与范围

独立复读本地 `task_criterion.py`、`train_task_conditional.py`、内容控制内核、shuffled loader以及vendor训练器；通过SSH只读核对94实际 `DetectionTrainer.preprocess_batch`。未编辑两个接线文件、vendor或现存run，未启动GPU。

本轮只审核原OEv1的C归因路径。CL的shuffled定位内容内核已实现，但尚未接到正式训练，本报告不替其验收。

## 数据与梯度判断

| 核验点 | 结果与理由 |
|---|---|
| Shuffled输入归一化 | 94实际原生preprocess将所有Tensor搬到device，仅对`img`除255；旧wrapper对`strong_img`除255，新wrapper对`content_img`除255。因此三种图各归一化一次。 |
| Shuffled内容与选择 | paired T决定对象匹配、原GT区域、E、q、K、分母；wrong T只替换最终证据目标。错误图的类别或标签不重新过滤对象。 |
| 同模态是否访问IR | 在privileged YAML=RGB、identity mapping、独立RGB seed0教师的设置下，teacher dataset只构造RGB数据；`same_modal=True`直接复用学生图像与RGB标签；损失入口使用RGB标签allowlist。未发现IR读取路径。 |
| 教师与参考梯度 | 原加载器冻结参数；所有教师/参考前向处于no_grad；内容目标再次detach。教师/参考仍不在学生optimizer/EMA导出中。 |
| C-only的L权重 | 两条新C控制均显式`lambda_L=0`；total直接取旧`combine_loss(native,C,B,lambda_C)`，不把L标量加入总损失。L诊断仍运行。 |
| 接口 | loader返回的content_img与criterion使用一致；C内容函数和same-modal函数返回的stats均符合criterion计数/梯度检查所需字段。 |

真实CPU loader先前验收在Drone/LLVIP各首、中、末3张train图及collate上证明：全部原paired字段、Python/NumPy/Torch CPU RNG连续状态完全相同，wrongIR与pairedIR几何矩阵一致。原图尺寸每数据集均只有一组，shape分层不增加实际子组。

## 本轮发现且已经修复的事项

1. vendor的`runtime_ready.json`原本硬写`teacher_labels_used_by_kd=True`。外层现在在写入前修正same-modal字段，并额外记录`privileged_ir_labels_used_by_kd=False`；未改vendor文件。
2. run_evidence原先遗漏实际内容模块与donor roster。现在trainer列表包括`shuffled_pair_data.py`，loss列表包括`content_controls.py`，shuffled split_rosters包括冻结derangement。
3. `C_ONLY`最初让正式C内容对照绕过canary验收。现在正式`c_shuffled`/`c_same_modal`必须有本臂ACCEPTED记录、至少24次成功更新、非零C梯度、相同dataset/teacher/reference、旧C/N等价；shuffled还绑定相同donor roster。

## CPU接口检查

使用实际本地TaskCriterion，注入CPU合成native/frozen模型接口，隔离日志写入。Python3.8 / PyTorch1.8，`CUDA_VISIBLE_DEVICES=''`。两臂均满足：

- 与独立内容内核调用相比，总loss与C stats逐项精确一致；
- C梯度非零，L最终权重为0；
- paired teacher前向次数为shuffled两次、same-modal一次；
- native/C/L梯度组合与weight0检查通过。

脚本为`verify_wiring_cpu.py`，实际结果为`wiring_cpu.json`。合成接口不验证真实checkpoint数据身份、AMP、optimizer更新、GPU显存或最终检测指标；这些由随后的独立24更新canary完成。

## 产物与后续

远端已有数据管线证据：

`/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/task_conditional_shuffled_loader_cpu_20260907/real_dataset_cpu.json`

同目录包含固定seed20260907的`dronevehicle_derangement.json`（17990、0自配）和`llvip_derangement.json`（9619、0自配）。原生框架CPU fixture共8项通过。

下一步仅放行资源guard下的两个24更新canary。检查实际非零C梯度、成功更新计数、模型与donor身份、学生输入一致性及完整进程树资源峰值后，再形成正式acceptance；本报告不提前填写GPU通过或收益结论。
