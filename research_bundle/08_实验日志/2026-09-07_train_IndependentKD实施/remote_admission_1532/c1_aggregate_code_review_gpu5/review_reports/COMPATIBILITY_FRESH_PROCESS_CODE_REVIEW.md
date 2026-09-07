# Fresh-process 兼容验证独立代码审阅

**结论：当前六次单 seed 运行路径 CODE_SCOPE_ACCEPTED，可以部署新不可变 release，在同一资源 lease 下重做真实 GPU 兼容验收。此次不接受尚未执行的真实轨迹等价，也不允许复用失败 attempt。**

日期：2026-09-07。审阅者 `/root/review_matrix_spec` 未编写或修改 `verify_compatibility.py`、`test_verifier_process_isolation.py`、trainer 或 runtime。只读源码及本地 CPU 测试，无 SSH/GPU。审阅源码副本在 `compatibility_fresh_process_code_review/`。

## 核查结论

1. **修复不改变被比较的训练路径。** `run_isolated_phase` 用同一 Python 可执行文件和当前 verifier 绝对路径，分别启动 loader、historical、new；`subprocess.run(check=True)` 阻塞等待阶段退出。不是删除 RNG 比较、手动补一个随机数或重设某个训练中间状态。旧/新训练仍从固定 seed 开始，各经历相同首次导入环境。
2. **父进程不执行 CUDA 阶段。** 原生 loader 的 pin-memory 可能初始化 CUDA，现也放入独立子进程。父派发前后以及主 run 检查 `torch.cuda.is_initialized()`；父没有 reset/查询 CUDA 峰值调用。全进程树由现有 lease 覆盖，每个子阶段先验证自身属于继承的单 GPU lease；不新申请 lease、不重选卡、不 detach 新会话。实际 guard 的后代 PID 计算支持此结构。GPU 峰值由依次退出的 worker receipt 取最大，不把父 CPU 状态冒充测量。
3. **每阶段身份绑定。** Worker 校验 CLI seed/arm 与实际 YAML 完全一致、source=paired、E200/640/B32/nbs64/workers4，且经过 `validate_execution`（含 N=0/C0=.1、数据无 test、pinned 版本、模型路径等）。父核返回阶段、完整 cfg、完成状态，再核真实 execution binding；绑定实际加载源码（含 __main__/入口、runtime、trainer、criterion、冻结 loader / OEv1）、数据 YAML、mapping 和已引用 roster 文件。
4. **精确范围保持。** Loader 仍30个完整 batch，比较原 RGB/IR 像素、标签、原字段、worker RNG 与父 RNG 延续；只容许 pair_info 内新增追踪字段。训练两条轨迹各24次真实 optimizer.step，比较初始参数/缓冲、所有具名梯度、optimizer/scaler/EMA、实际 skip/update、样本字段及完整 RNG。C0 必须选中对象且有非零 KD 梯度。失败保存少量 RNG 差异后仍抛原断言，没有把失败放宽成成功。
5. **正式六份 scope 不放宽。** 当前调度方式应是 N/C0 × seed0/42/123 六次单 seed 入口；每次根 receipt 绑定该 seed 的 cfg。现 `admission.check_readiness` 再核 receipt seed/arm 与 binding 实际 cfg，最终集合必须恰好等于六对身份；重复同一份不能补足缺失。单个 N42 通过不升级全六份。
6. **失败与不可变产物。** 每次输出 / 各阶段目录 `exist_ok=False`，原 trace 和失败 attempt 留存。子失败由非零返回码传播，父只生成 failure receipt，后续阶段不运行。新增 `initialization_rng_stages.pt` 和 `mismatch_rng_*` 有助于区分首次导入与数据增强；不复制训练原图全集。

## CPU 验证

- `D:/Anaconda/envs/KGJ_proj/python.exe -m unittest test_verifier_process_isolation -v`：独立 **6/6 通过**，0.124s。真实 Python 子进程测试重放一次性导入 RNG；其余测试检查顺序阻塞派发、父已有 CUDA 时拒绝、子失败传播、错误阶段回执拒绝和失败 RNG 留存。
- `D:/Anaconda/envs/KGJ_proj/python.exe verify_compatibility.py --self-test`：独立 **4/4 通过**，0.003s，精确 tensor / tree 与 RNG observer 逻辑未退化。

测试包含明确合成模块和 mocked dispatcher/binding；其作用是检查隔离与状态流，不冒充真实 Ultralytics、GPU、24步数值等价或服务器资源峰值。

## 一个接口边界与真实验收边界

CLI 仍允许 `--seed all`，但其顶层 receipt 的 seed 为字符串 `all`，各 per-seed receipt 当前也未带可供 admission 使用的独立 execution_binding。**all 模式结果不能直接进入当前正式 ready。** 这是安全拒绝而非错误接受；不阻塞根当前已明确的六次单 seed 调度。本次代码接受只覆盖单 seed 入口；若以后要支持 all 模式准入，应补逐 seed binding 或显式关闭该入口，不能在收集时随意重标顶层 seed。

首次 Events 导入消耗随机数的因果链有源码和旧 trace 模拟支持，修复是否充分仍由本次 fresh-process 真实六份轨迹决定。旧失败不能改写为成功；新测量后再评估 GPU/RSS 预约、精确等价及正式 N/C0 复用。未见其他阻塞当前重验的源码问题。
