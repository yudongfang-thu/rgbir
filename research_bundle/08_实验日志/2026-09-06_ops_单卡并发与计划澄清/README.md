# 单卡并发与计划澄清（2026-09-06）

> 当前主方法仍为OEv1对象判别证据蒸馏，random是归因对照。用户规则允许每卡最多3任务；此前每卡1个正式训练是保守排程，不是用户限制。23:59已实测并启用GPU2双正式训练R42+R0；R123短测通过、等待N42完整评估后接用GPU4。方法和原P/N运行不变。

## 目的
回答当前方法、原计划沿革及单卡多任务规则，避免把方法、对照、数据集扩展和调度安排混为一谈。

## 设置
原P/N×3保持冻结；OS-SSL暂停、VEDAI暂缓。先在已有GPU2的random42旁运行同源码seed0的24更新canary，输出独立profile目录。入场通过同一个全局guard，短测预约6500MiB显存/32768MiB RSS，依据既有P0与R42同配置canary峰值6304MiB；完整训练已观察峰值7632MiB，不能用短测低峰代替。

短测通过后，在原guard同锁内只修订R42和P0两个精确live lease的显存/RSS预约（10000→8300、49152→32768MiB），保留before/intent/after/receipt，不改guard源码、PID、绑定或模型。迁移待运行R0/123，各先做canonical canary和直接张量/源码/选择剂量校验；R0在GPU2正式双开，R123在GPU4短测后等待N42完整评估回执。取消原拟GPU5双开，避免P0的终点评估资源预约冲突。详见[冻结调度交接](PARALLEL_HANDOFF_PLAN.md)。

## 结果
独立并发profile通过：30个双CUDA PID样本，全卡峰值13973MiB、最低空闲10111MiB；24次真实更新/6次AMP跳步，KD梯度非零，teacher/reference无梯度；原R42更新2313→2373持续推进。详见[独立短测审计](independent_concurrency_audit/README.md)。

R0/123 canonical canary均通过：各24次真实更新，与原P同seed的初始化和首batch直接张量相等、30个共同batch的K/分母/名义剂量一致且选择对象不同。loss/loader相同，trainer只有原先已审核的arm路由两行差异。预约修订13项CPU验证及worker独立审查通过。

23:59快照（epoch为正在进入的轮次，不是已完成轮数）：

| GPU | 实际正式任务 | 状态 |
|---|---|---|
| 2 | R42 + R0 | R42第11/200轮；R0正式进程已入场，启动初始化；guard确认2个formal训练及2个CUDA PID |
| 4 | N42 | 第175/200轮；R123已通过短测，CPU等待N42训练和独立评估全部完成 |
| 5 | P0 | 第65/200轮 |
| 6 | N123 | 第58/200轮 |

共4张物理卡、5项正式训练，GPU1/3/7全空，符合4卡放宽条件；实际RSS约139.2GiB、预约192GiB。原P/N仍3/6独立端点、0/3完整同seed对，没有本轮新增方法效果结论。原始状态见[parallel_live_snapshot.json](parallel_evidence/parallel_live_snapshot.json)。

## 结论
用户AGENTS每卡最多3个任务且保留≥2GB、全局RSS≤300GB；正常最多3卡，空闲条件允许4卡。按当前完整训练峰值7632MiB计算，双开约15264MiB；三开约22896MiB。实测memory.total为24564MiB，其中480MiB保留，可分配used+free共24084MiB，三开只剩约1188MiB（另计上下文），不满足余量。故本轮实际采用双开。旧guard还设2个CUDA PID、70%显存及第二正式训练profile条件，本次保留这些更严限制，并基于实测修正了过高的预约值。

OEv1的科学验证链不变：先P对同代码weight0检验净收益，再R检验整体可靠性/质量选择；same-modal、有效shuffled、GT-only及FGD尚未正式实现/启动，不能算作完成。详见[方法与计划沿革](plan_lineage.md)。

23:59:40追加独立只读核验：R0已进入第1轮、完成64次真实更新，GPU2两项正式训练均继续；R123仍等待N42完整评估。项目实际RSS约140.33GiB、预约192GiB。证据见[独立启动审计](parallel_evidence/INDEPENDENT_LAUNCH_AUDIT.md)，本次没有新的AP端点。

## 产物路径
`plan_lineage.md`记录原计划；`concurrency/`保存只读94状态和旧队列/guard源码；并发短测脚本和回执保存本目录及94 `artifacts/oev1_concurrency_20260906/`。

## 局限与下一步
不根据短测修改λ/T/ρ、batch、seed或检测端点，不把并发profile当方法效果或全程吞吐验证。正式双训练会共享计算资源，不能承诺2倍加速。

既有random42父队列没有热更新停排接口，保留其日志管道和完整评估。转交的canonical seed0目录会使旧队列在42评估结束后停止，原status可能记failed；必须核验42完整端点和确切seed0已存在断言，不能将任意失败解释为计划交接。R42原终点评估的10000MiB预约可能等待R0释放；P/N主实验不受此预约冲突影响。所有原始attempt、checkpoint及失败记录保留。
