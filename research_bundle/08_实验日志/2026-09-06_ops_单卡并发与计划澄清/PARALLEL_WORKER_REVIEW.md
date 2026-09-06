# parallel_seed_worker.py 独立审阅

> **修订复核：通过。** 每阶段已补用户四卡放宽条件；seed123改GPU4，先canary，再等待N42的metric与完整eval receipt后才full。CLI、实际源目录和回执文件名已核验。当前审阅不运行worker、不写远端或修改进程。下文初审问题保留其历史含义。

## 修订版复核

1. 新`guarded()`在每个stage及每次重试前调用同一guard的`inspect`，读取`usage.active_gpus`，并用nvidia-smi确认全空卡集合；正确执行至多3卡，或至多4卡且入场后至少2张全空卡。**R1已解决**。
2. 新seed/GPU映射固定为0→2、123→4。R123仅在GPU4做canary并验证通过，随后等待`runs/rgbir_object_evidence_v1_20260906/full_weight0_s42_attempt1/evaluation_val.json`与`eval_evidence/run_receipt.json`两者存在，才进入full。该N42实际run目录已通过远端只读`ls`确认；已有P42同评估器确实输出该receipt路径，receipt工具源码也使用`run_dir / run_receipt.json`。
3. GPU4 canary预约6500MiB，依据已完成6304MiB短测；full/eval预约8300MiB。N42原预约10000MiB与短测6500MiB合计16500MiB，低于70%准入线，且每次另检查实际空闲2048MiB余量。正式R123在N42完整eval后启动，不挤占其终点评估预约。
4. canonical seed123 paired canary回执路径已远端只读确认；validator的`--gpu`收到实际GPU4，没有硬编码只准GPU2的问题。
5. legacy核对现在要求完整eval_evidence receipt，且备注明确通用failed不能证明退休、仍要核对canonical seed0断言traceback；不再无条件把queue失败解释成调度退休。
6. 脚本AST解析通过。原canary→validator→full→固定eval顺序、输出原子创建、只重试未启动拒绝、保留R42/P/N原则没有变化。

剩余非阻断事项：采样线程异常收集可增强；N42如果真实失败，R123保持waiting不会错误进入full，需要后续状态审计处置；R42自己的旧eval仍可能等待R0。当前没有发现应阻止该修订版启动的API、路径、方法或重复运行问题。

## 已核实

- `--seed`只允许0/123，`--gpu`只允许2/5，并强制0→2、123→5。
- `--max-steps 24`只用于canary；canonical canary完成后调用已审阅validator，返回码正常且输出status=passed才允许full。
- full复用固定release_v1/config_drone.yaml和seed；不更改batch32/nbs64、E200、λ/ρ/T、数据或模型初始化。
- full执行guard的`--formal-train --profiled-second-train`；该CLI确实存在于当前guard。
- canonical canary与full路径启动前检查不存在；trainer使用原子`mkdir(exist_ok=False)`，重复进程不能成功共享同一run。ownership JSON使用独占创建，新worker相互重复会失败。
- evaluator参数`--config --run`与实际实现一致；实际评估器先检查training_completed，固定last/EMA、1469图val，不访问test。
- 只有guard返回rc=2且日志为QUEUED、从未LAUNCHED才重试；已启动故障不会自动换attempt重跑。
- 没有信号、kill、原源码改写或原日志覆盖操作，不干扰既有R42或P/N。

## 部署前需要修复

### R1：每阶段入场补用户四卡放宽条件

初版`guarded`直接调用全局guard，没有旧random_worker中`admission_policy()`。当前guard仅设MAX_ACTIVE_GPUS=4，并不自行判断仍有至少两张全空卡。

应在每个stage及每次资源重试前读取guard inspect和nvidia-smi，用相同条件`len(active)<=3 or (len(active)<=4 and len(empty-active)>=2)`核验；不足时等待而非启动。当前不增卡也不应省略原队列已有的入场政策。

## 非阻断但需要明确的影响

1. 旧R42驻留worker将来eval仍预约10000MiB。新R0 full预约8300MiB，共18300MiB超过当前guard 70%线（24084×0.7=16858.8MiB）。因此R42训练完成后eval可能等待R0 full退出；checkpoint安全，不是训练失败或死锁。无需为此中断42，但不能承诺旧42一结束训练立刻得到评估。
2. 旧队列在42完成train+eval后试图启动canonical seed0 canary，目录存在断言会failclosed退出，机制清楚；该异常应在交接文档标为预期调度退役，保留原raw status。
3. `legacy_queue_seed42_completed_handoff.json`只在新R0自身完成时检查一次，若此时42 eval尚未结束就不会写，之后也不会补写。不能将其缺失理解为handoff失败；可在后续状态审计补充。
4. 不能只凭42 `completion_receipt.json`和`evaluation_val.json`存在，把任何旧queue失败解释成退役。评估器先写metric JSON再写eval_evidence receipt，若后一步失败也可有这两个JSON。若要写成“确认预期退役”，还需核验旧queue确实在seed0存在性检查处退出，或完整评估回执与对应traceback/阶段日志。
5. canary外部采样线程失败可能只导致样本停止；建议捕获线程异常并在允许full前失败，避免在测量不完整时仍声称profile完备。已有profile要求至少3个双PID样本可部分约束，但并非证明整个canary采样完整。

## 已有独立profile结论

GPU2 R42+seed0短测30个双CUDA PID样本，全卡峰值13973MiB、最小空闲10111MiB；短测24次真实更新、有非零KD梯度，原R42持续更新。详见`independent_concurrency_audit/README.md`。这里只支持同卡共存工程检查；全程full显存和吞吐仍需正式训练资源回执。
