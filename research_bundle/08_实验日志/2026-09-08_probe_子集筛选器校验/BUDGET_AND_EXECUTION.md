# 执行前固定预算与比较合同

**本次只验证廉价筛选器能否保留已知N/C0对照的方向；原全量E8与新子集E8不是等曝光训练预算。没有新结果前固定本合同，不据AP修改。**

方法、数据、初始化、八轮日程按上条目的NEXT_FAST_SCREEN_TASK。只用原global lease顺序运行N canary、C0 canary、N训练/评价、C0训练/评价；先通过两canary及预算，再训练，不先单独跑N完整训练。所有阶段独立fresh初始化，正常BN；不修改原C0算式，不迁移C1加速候选。

每canary至少24次成功optimizer更新、至少30个batch，最多48次optimizer尝试。记录尝试、AMP skip、成功更新和EMA调用，不能互相替代。N canary在服务器数据盘保存共同五类学生初始state及前30批增强后、归一化前的完整双模态uint8输入和双标签；另三次训练运行逐批读同参考并做torch.equal。只允许本新attempt读取，源stat/对应序号记录。约2.4GB输入tensor只留数据盘，**不下载、不发布、不以hash代替逐元素比较**；像素证据范围明确限于前30批。

45分钟指本attempt的执行预算；资源准入/排队时间单列。原dispatcher写RUNNING时开始计入该stage，直到终态，包含worker导入、初始化、训练、像素读写、评价和保存；阶段间本driver的校验/交接也计入。只在私有调用侧观察原dispatcher状态，不修改原文件、不创建第二套lease。每stage拿到剩余预算后，worker检查超时；外层在本stage实际RUNNING后监测deadline，只能终止该job自己启动的进程树。排队不占执行预算。超时保存incomplete/failure，不取中途checkpoint算结果。

两canary完成后，使用各自所有实测完整batch时长的算术平均（扣除单列的像素审计I/O），不选择最快批：

`train_estimate_arm = 1.2 × (512 × mean(normal_batch_seconds) + observed_non_batch_overhead + observed_flow_audit_seconds)`。

`observed_non_batch_overhead=max(0, canary.seconds − sum(normal_batch_seconds) − flow_audit_seconds)`，含初始化、最终处理及测量未归入batch的开销；N写入参考与后续读取不同，计入其实际写入成本作为保守参考。完整训练仍保留前30批核对，因此I/O不省略。另为两次完整dev独立评价各预留60秒；这一时间来自既有完整评价量级的保守额度，评价显存另须已有同路径的实测依据，仍受本次guard约束。

两canary已花执行时间、上述两臂训练估计、两次评价和已花driver开销合计超过2700秒时，预算BLOCKED，不启动两臂完整训练。恰好2700秒可准入；估计不代替实际硬限时。没有调整轮数、LR、batch/workers、子集或λ的补救分支。

新canary初始测量预约8192MiB显存/32768MiB RSS；正式训练使用各臂真实canary峰加现有余量（GPU max(256MiB,5%)向上256取整；RSS max(2048MiB,5%)向上1024取整）。整卡2GiB空余、项目显存<70%、最多实测双开、项目总RSS≤300GB及240GiB预约阈值继续由原lease/guard检查。通常三张物理GPU，第四张仅在加入后仍至少两张全空卡时按日志援引规则。

两臂完整末端及评价/来源检查有效后，只报告C0−N与原值。一seed不报SD；正向仅表示这个已知对照在本子集/日程保留，非正向只结束这个筛选器版本。失败、missing或预算中断不记成负AP，不重复全量E8，也不进入E200。
