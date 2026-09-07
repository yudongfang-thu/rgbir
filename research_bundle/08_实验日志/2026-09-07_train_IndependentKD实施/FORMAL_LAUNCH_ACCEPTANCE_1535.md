# C1 三 seed 正式启动与技术验收（2026-09-07 15:35）

**C1 seed42/0/123 已按预定顺序正式启动，均已达到至少24次成功更新；没有新的方法 AP 结果。L1继续因几何证据不足阻塞。**

## 实际技术验收

- 冻结训练版本：94 `release_gpu5`。173项工程CPU测试及18项参考包测试在指定 torch2.10.0+cu128 / Ultralytics8.4.115 环境通过。
- N/C0 × 0/42/123 六条真实轨迹全部 ACCEPTED：每条30完整数据batch、24成功update，初始化、完整梯度、更新、EMA及随机状态严格相等。只支持已检查范围，不宣称验证了整个E200的逐位一致。
- C1 固定64自然batch校准：64/64有限非零，λ=0.09227393550836771，无裁剪；参数模块为16/19，R副本每批重载参数及buffers并train。C1_y共用该系数。校准回执里的unique_images字段是定位计数器，在分类分支为0；分类自然流身份见64批实际清单，不能误读为没有训练图像。
- C1/C1_y：各24成功update、30尝试、6次AMP跳步、3767次入选对象计数（批次累计计数，非唯一物理对象数），KD梯度非零；各峰值NVML6510MiB，进程树RSS28699/28763MiB。
- C1双开：与原same-modal任务共用GPU5，28个实际工作采样区间均有两个项目CUDA任务。新增C1 NVML6510MiB、RSS28744MiB，整卡最小空闲9802MiB。未使用第四卡条款。
- 完整dev1469的原生/对象记录评估器：五汇总指标及逐类AP严格相等；评价资源NVML1370MiB、RSS6260MiB。该探针是N42评价接口验收，不是新C1效果。

## 三 seed 与资源

15:32:11 实际快照：

|seed|物理GPU|训练PID|成功更新|阶段|
|---|---:|---:|---:|---|
|42|2|2753901|64|E1/200，持续训练|
|0|2|2757514|44|E1/200，持续训练|
|123|5|2760047|24|E1/200，持续训练|

实际启动时间由 `formal_live_20260907_153211.json` 的 LAUNCHED 事件记录。各seed的progress.json沿用legacy的arm=paired标签，正式身份应读protocol_config.yaml与launch_manifest；scheduler state.json中的TRAIN_WAITING是进入阻塞run_job前的状态，真实运行以LAUNCHED、训练progress和资源采样交叉判断。

实际输出位于94数据盘：`/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_independent_kd_v2_20260907/formal_C1_gpu5_attempt2/runs/C1_seed{42,0,123}/`。最初计划的独立runs根尚未使用；没有移动任何原始目录。

每训练任务按本次C1实测峰值另加2048MiB预约显存，按实测进程树RSS加4096MiB预约内存；评估显存仅加256MiB。仍同时满足每卡最多2个项目CUDA任务、项目VRAM<70%、整卡至少2GiB空闲、240GiB提前排队与300GB硬上限。15:35项目五个正式训练共用GPU2/4/5，总RSS约141.19GiB。

## 失败与修复

科学验收的首次RNG不对称、诊断RSS预约不足及验证器回调接口失败均见相应原始attempt与修复文档，未删改。

第一个formal campaign在导入Ultralytics前失败，未创建run目录、未执行训练batch。原因是调度器resolve解析了虚拟环境解释器链接。只修LOG队列脚本两处为absolute；指定venv入口已在94实际CPU确认，根独立重跑16项队列测试并核对两处差异通过。NEW训练代码、系数、数据、六兼容及canary证据保持不变。新campaign使用attempt2；旧暂停协调器在确认零训练后退役，旧日志/manifest/failure均保留。

## 后续责任

各worker在E200后顺序运行完整dev独立评估；不按中途AP停训。C1_y三长训、最终四臂等仍按冻结阈值触发，尚未入队。旧C0 shuffled42/same-modal42截至15:35在E52/E36，继续原训练及评价。

旧N/C0汇总指标桥接候选保留DRAFT。必须补采旧六last的逐类AP与objects，才能完成固定阈值修复/损伤和背景误检比较；缺失证据不能跳过升级前的harm检查。几何UNKNOWN不被转换成定位无效结论，L门控不放宽。

复核入口：`remote_admission_1532/source_manifest.json`、`formal_live_20260907_153211.json`、`stage_status_20260907_152201.json`及本目录各独立review。GitHub发布状态另见实际发布回执。
