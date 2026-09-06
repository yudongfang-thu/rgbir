# Task-Conditional接线独立静态审阅（02:34版本）

> **整体接线满足冻结T/R、C原函数复用、独立L和只读native归属方向；正式训练前需落实C浮点加法等价、正式receipt身份绑定及C非零canary梯度检查。此为只读审阅，不是实际训练通过证明。**

审阅文件：新模块`task_criterion.py`、`train_task_conditional.py`、`tracked_pair_data.py`；辅读legacy trainer/paired loader、localization loss/geometry/calibration。实际94仅导入并读取pinned8.4.115原生增强源码，没有GPU或远端写入。

## 已确认的合理部分

- TaskCriterion继承legacy训练态criterion，teacher/reference均eval+no_grad，native先调用；现有legacy setup把辅助模型排除于optimizer和EMA。
- C调用原`object_evidence_loss(...arm='paired')`，相同证据cfg/seed。L仅在冻结R-selected anchor上构造，native fg不作选样门。
- native归属hook只保存assigner输出fg_mask/local_gt_index；结合batch_idx映射回全局GT。unique规则使用全RGB GT并复制小GT扩展至stride[1]的native规则，不只看paired GT。
- `ParameterTap`只包装原get_params一次，不另抽随机数。已读取94的RandomPerspective/LetterBox/RandomFlip get_params，键名M/ratio/left/top/flip/direction与实现一致，且在apply前获取；固定增强无旧版pre_transform遗漏。Drone实际样本路径无symlink，canonical来源一致。
- loader沿用原before/after RNG回放；几何metadata在同一次处理后加字段。仍需真实多batch初始化/输入/RNG等价canary，不能凭静态结构宣称已通过。

## 已向root报告的修正点

1. **C总损失结合顺序。** 原legacy先在Python形成`float(B)*float(weight)`再乘kd；新实现先wc*c再乘B。实际batch32是2的幂时通常一致，但Drone末batch6不可预设bitwise一致。建议n/c直接用legacy.combine_loss；CL/CGT从同C base再加B*λL*L，真实短测比较包括末尾小batch。
2. **正式启动证据身份。** 当前validate_config只确认geometry文件存在、calibration状态和λ、D2 geometry_verified/count、canary status/arms；尚未加载当前GeometryContract确认verified及非空entries，也未把dataset、geometry路径、recipe/λ、24次成功update与当前run绑定。应核对这些具体字段，避免错用其他数据集或旧geometry的receipt，不需要添加哈希。
3. **C canary非零信号。** 当前只要求gradient_checks非空；即便C score梯度始终0也可能通过。建议c/cl/cgt要求至少一次非零C梯度，L类另要求非零L梯度；n仍验证0系数与native等价。原legacy已有非零C检查，应保留。
4. **真实配置记录。** --seed改写cfg后，protocol_config.yaml仍复制原文件；虽然launch_manifest/args记录实际seed，建议另写resolved config并在receipt明确引用，减少新seed复用旧配置造成的阅读歧义。

这些点不要求停止现有旧实验，也不否定新算子CPU测试。后续root修改后应以实际修复及canary回执为准，此文保留审阅时状态，不追认未来运行。
