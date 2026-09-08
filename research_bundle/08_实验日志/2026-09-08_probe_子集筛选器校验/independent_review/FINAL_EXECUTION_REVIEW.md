**PASS_EXECUTED_SUBSET_SCREEN。** 两臂实际完成通用预训练初始化、正常BN、固定2048子集的E8/512batch及各一次完整dev1469图/22462GT固定last/EMA评价。当前筛选器未保留已知C0正向对照；不能据此判断C0无效。

| 臂 | mAP50–95（百分数） | 成功更新 / 尝试 | AMP skips / EMA调用 | 训练秒数 | 评价秒数 |
|---|---:|---:|---:|---:|---:|
| N | 27.588046 | 298 / 304 | 6 / 304 | 224.976281 | 18.347238 |
| C0 | 26.459942 | 297 / 304 | 7 / 304 | 214.619190 | 19.024567 |

C0−N = **−1.1281045876 pp**。已核原fraction、五类宏均值及已有分析器单位/差值；无新增指标或AP重推理。两臂并非成功更新数完全相同。

共同499tensor五类初始化、fresh optimizer/EMA、前30批服务器完整双模态uint8像素/双标签等值回执闭合；正常BN有243buffer改变。该像素证据只限前缀，原张量未下载。全训关闭梯度探针，因此gradient_checks为空及nonzero_kd_gradient=false不表示KD为零：真实canary已证明非零梯度，保存的8条C0训练记录均有正KD损失。两臂62774为累计选择次数。

6个实际阶段均clean exit0、monitor_errors为空，无failure/timeout冲突。训练NVML峰6704MiB、RSS17727/17964MiB，处于7168/20480MiB预约内；完整评价NVML1370MiB，RSS4113/4159MiB。原lease显存余量/项目RSS/第四卡条件通过。源码仍与10输入快照逐字节相同，实际source manifests、配置副本、初始化来源及训练/评价checkpoint stat相互闭合；两份actual评价合同、完整roster和capture GT相同。

累计执行 **571.548243秒**，排队 **10.687456秒**，总墙钟 **582.235702秒**，均由实际6阶段回执闭合；执行低于2700秒硬限时。排队不包装成计算时间。

这是固定子集/seed42/独立E8的已知对照校验，未通过方向保留检验；不证明一般方法排序、E200预测性或C0整体无效。未重复CPU测试/流测试，未启动GPU、读取权重、修改源码或计算新hash。详细检查与来源见FINAL_EXECUTION_REVIEW.json和FINAL_EXECUTION_INPUTS.json。
