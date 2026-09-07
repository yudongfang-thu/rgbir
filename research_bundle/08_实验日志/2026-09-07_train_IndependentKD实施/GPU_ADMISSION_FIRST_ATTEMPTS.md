# GPU准入阶段记录（截至首次兼容失败）

**C1两批真实FP32资源probe通过；N42首次兼容因验证器首次导入随机状态不对称而失败，未放行正式训练。**

- release_gpu1：pinned18包测试通过；新143附近测试中的resource fixture错误引用部署父路径，GPU未启动，原输出保留。
- release_gpu2：153项工程测试+18项参考包测试通过；GPU排队时错误地要求首次probe完全空卡。发现GPU2虽有其他用户占用3805MiB但空闲20279MiB，16,000MiB cap与2GiB余量可安全相容。只取消尚无LAUNCHED事件的本项目调度器，未停任何训练/他人任务。
- release_gpu3：保留真实共享规则，第四卡仍需另两张完全空卡。CPU全部通过，C1两批profile实际在GPU2执行：Torch allocated7955.83MiB/reserved8732MiB，NVML9232MiB，完整进程树RSS24041MiB，24.39秒，两批KD非零。仅PROFILED，不把2批λ当正式系数。
- 第一个N42兼容attempt1：通过真实loader阶段，初始化trace的PythonRNG不等；同时被32GiB诊断预约拦下（实测38679MiB，项目总量未达到300GB）。旧trace/日志不动。

## 最小修复

94 pinned Ultralytics在callbacks/platform首次导入的Events.__init__中调用一次random.random。同一进程顺序构建旧新trainer，仅旧路径发生该副作用；CPU对旧trace重构完整RNG已验证。改为loader、旧轨迹、新轨迹分别使用新Python子进程，顺序执行、共用一个lease，父进程不初始化CUDA。不忽略RNG字段，不调整学生/教师/参考/数据/增强/优化器。

新诊断将按64GiB RSS预约，保留240GiB提前排队与300GB全项目上限。生产学生仍为workers4，未为解决诊断内存改变正式recipe。

证据：remote_admission_1430/保存真实小产物，COMPATIBILITY_FRESH_PROCESS_FIX.md与同目录CPU复现记录解释失败。GPU兼容最终状态以新attempt实际回执为准。
