# RGBIR 对象判别蒸馏三 seed 扩展（2026-09-06）

> 结论：新增四个完整run已安排，其中weight0 seed0与paired seed123已开始实际更新，其配对臂分别同卡排队；原paired42约89/200轮继续。全项目仅GPU4/5/6，每卡一个训练，尚无最终AP。

## 目的
检验已冻结对象判别蒸馏干预的seed稳定性。当前训练健康检查不提供性能增益结论；不据中间loss改方法或阈值。

## 设置
原GPU4的paired42→weight0 42队列继续；GPU5新队列执行weight0 0→paired0，GPU6执行paired123→weight0 123。实际卡号启动前动态选择并记allocation.json。每张卡一个本轮正式训练，同卡串行训练和终点评估。所有run由项目resource guard覆盖，显存预约10000MiB、RSS49152MiB，空闲余量至少2048MiB。

学生seed扩展为0/42/123；IR教师和RGB参考保持同一对seed42 checkpoint。本实验测量学生训练随机性，不声称评估教师seed方差。trainer/loss/loader/config从原release_v2复用不修改，通过既有--seed参数改变学生seed。E200、batch32/nbs64、workers4、λ0.1、ρ0.5、T2与候选阈值全部不变。所有端点为固定E200 last/EMA、统一1469图val。

## 结果
旧run健康审计：health_audit_snapshot1/health_audit.md。该快照完成86轮、当前87，首5→末5轮训练cls均值1.115328→0.561942；487个已记录batch无非有限值、零选择或零KD；末期入选约26.7%。这些是工程现象，不证明检测收益或后期梯度质量。

扩展8项CPU控制流检查通过；seed0/123各两臂24实际更新canary通过，初始权重及首批输入同seed成对相等，KD梯度非零、weight0与native等价。四个canary显存峰值均6304MiB，RSS峰值最高28766MiB，资源门通过。

2026-09-06 07:42:39 +08:00启动核验：paired42为89/200轮；weight0 0已234次实际更新，paired123已223次。全部三个GPU每卡1个CUDA PID，实际显存分别7632/7606/7606MiB，全项目RSS86088MiB（约84.1GiB），预约144GiB，满足限制。完整证据见launch_verification_snapshot1/summary.json。

跨seed身份复核：seed0/123的真实args/launch/completion一致，两臂初始student均499个张量中12个不同；首批保存张量及前30batch样本顺序相同，符合原生DataLoader固定generator的实现。当前重复覆盖初始化随机性，未覆盖所测数据顺序/增强随机性的变化；不因这个发现改动既有recipe。见cross_seed_realization.md。

固定端点汇总器经10项CPU fixture与独立审查通过。endpoint_snapshot2当前0/6完整终点，三seed统计为空，不读取CSV的val占位0。汇总器当前源为本目录根下analyze_three_seed_endpoints.py；历史code/首次副本保留，使用时遵循ENDPOINT_ANALYZER_NOTES.md。

## 结论
本轮决策基于授权、已通过工程验证及三seed研究规范，未基于方法AP作选择。3seed P/N完成后仍只支持“该干预相对weight0的净结果”，四臂归因及同剂量/同模态证据仍未齐全。

## 产物路径
- 本地：本目录。
- 94新增小产物：`/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_object_evidence_expand_20260906/`。
- 94新增run：`/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbir_object_evidence_expand_20260906/`。
- 复用源码：`/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_object_evidence_v1_20260906/release_v2/`。
- 原seed42队列仍归首轮日志管理；原results/checkpoint/receipt不覆盖、不移动。
- 运行状态：EXPERIMENT_TRACKER.md；新screen为rgbir_oev1expand_queue_s0和rgbir_oev1expand_queue_s123。
- 新launcher：expansion_worker.py；94只读发布副本在扩展artifacts/release_v1。旧训练源码仍复用原release_v2。
- 各worker实际seed覆盖：94扩展artifacts/workers/full_s{0,123}_attempt1/effective_config.yaml与每臂seed_override.json。

## 局限与下一步
最终AP、逐seed方向与mean±SD待两臂各三seed完成后计算。不访问test、不跑不同阈值的试探性方法。后续same-modal/shuffled、random同K及GT-only的精确定义须独立冻结，不能用三seed P/N代替跨模态归因。
