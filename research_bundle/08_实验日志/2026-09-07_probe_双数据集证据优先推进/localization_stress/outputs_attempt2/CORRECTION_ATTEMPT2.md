# attempt2 汇总字段修正

独立审阅指出：attempt1 在 worst_case 行先写 `all_directions_gate_n` 成功数，随后通用 `statistics(...).n` 用同名键覆盖为总分母。旧 CSV 此字段与旧 README gate 存活表不可引用；逐对象布尔数组、`all_directions_gate_fraction`、连续指标及正文4px的34/61、9/47不受影响。

原 `outputs_attempt1/` 不修改；在修复前将原脚本、README、协议、自检存入 `attempt1_source_snapshot/`。新实现对布尔事件只写 `*_success_n`、`*_denominator_n`、`*_fraction`，不进入通用连续统计器，消除命名碰撞。加入“5对象分母但只有2成功，掩码外成功不计”的已知真值检查；完整CPU重跑写 `outputs_attempt2/`，自检将直接比对worst计数与保存布尔数组，并核两attempt全部逐对象数组exact。

本修正不改变冻结协议、对象、anchor、质量门、扰动、CE/KL/梯度定义；不是看结果重选对象。accepted状态由独立审阅者决定。
