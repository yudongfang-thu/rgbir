# 兼容验证首次导入 RNG 差异与最小修复

**结论：首个 N42 验证在 30 个 loader batch 已通过、历史 24 次更新完成后，于新 trainer 初始化的 Python RNG 游标处失败。实际源码和旧 trace 的完整 Python RNG 与“首次导入额外调用一次 random.random”严格吻合。已将各阶段改为顺序全新 Python 子进程，保留原生首次导入行为和全部精确比较；未启动新的 GPU 任务。**

日期：2026-09-07。根明确授权只读 SSH 94 诊断并修改 `verify_compatibility.py`。原始 attempt 和 release_gpu3 没有被修改或删除。

## 已读取的实际证据

服务器基础目录：`/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_independent_kd_v2_20260907`。

- `compat_N_s42_attempt1/failure_receipt.json` 仅报 `initial_trace.pt.rng.python[1][624]` 的精确比较失败；错误发生于新 trainer 的 on_train_start，未达到任何新学生更新。
- `compat_N_s42_attempt1/seed42_N/loader_receipt.json` 为 PASSED，前 30 完整 batch 的原字段/像素/双标签/worker RNG 比较已过。
- 旧 trainer 已完成真实 24-success trajectory；旧/新初始化比较仅报告 Python RNG state 的第 624 索引游标不同，其余已比较模型/EMA/optimizer/scaler 字段未报差异。
- 原始大 trace 留在服务器。只用 pinned Python 在 CPU 通过 `torch.load(... map_location='cpu', mmap=True, weights_only=False)` 读取旧 `historical/initial_trace.pt` 中的小 RNG 字段；两次诊断均确认 `torch.cuda.is_initialized()==False`。

## 原因链与可复核模拟

实际 pinned `ultralytics/utils/callbacks/platform.py` 导入 `ultralytics.utils.events.events`。该模块第一次构造 Events 时，`Events.__init__` 为 session_id 调用一次全局 `random.random()`；这是模块首次导入副作用，不属于训练图像增强。旧 driver 在同一个 Python 进程顺序构建旧 trainer、新 trainer，第二次构建复用该模块缓存，因此重新 seed 并不能重放首次导入副作用。

原生 `BaseDataset.get_img_files` 每次调用 `check_file_speeds`，后者固定使用 `random.sample(files, 5)`。单次 trainer 初始化依次处理 17,990 RGB train、17,990 IR train、1,469 RGB val。

远端 CPU 模拟直接比较完整 Python RNG state，结果为：

| 从 seed42 开始的模拟 | 末尾游标 | 与旧 trace 完整 state 相等 |
|---|---:|---|
| 三次上述规模 random.sample | 22 | false |
| 先一次 random.random，再三次 random.sample | 23 | **true** |

旧 trace 实际游标为 23。旧/新日志中首次 RGB train 测速抽中图像的平均大小也不同，说明差异在新增 tracked wrapper 接管样本前已经进入原生 dataset 构造。新失败实际 RNG 尚未被旧 driver 单独保存，不能把模拟出的 22 冒称已直接读取的新 trace 值；后续 fresh-process 验证仍是修复是否充分的实际判据。

## 已实施修复

`verify_compatibility.py` 保留用户 CLI 和固定范围：单 seed/arm 30 完整 batch、旧/新各 24 个真实成功更新。

1. loader、historical、new 三段分别用 `subprocess.run(..., check=True)` 执行同一 verifier 的内部 worker 入口，每段等待进程退出才运行下一段。继承唯一现有 lease；不申请新 GPU，不开并发训练。
2. 旧/新都在全新解释器中经历相同的真实原生首次导入，而不是手动消耗一颗随机数、重置某个中间 RNG 或删除比较字段。所有模型、全梯度、优化器、EMA、scaler、Python/NumPy/Torch RNG 仍严格逐项/逐位比较。
3. 父协调器完全不初始化 CUDA。移除其 `reset_peak_memory_stats` 和峰值读取，在子进程派发前后断言父 CUDA 未初始化。loader 也隔离，因为原生 pin_memory 路径可能在后台线程初始化 CUDA；不能以“loader 张量是 CPU”推断协调器永远没有 CUDA context。
4. 子阶段各写实际执行配置、PID、完整结果、GPU 峰值和 execution binding；父逐份核配置/阶段/binding，最终峰值取顺序子阶段最大值。父不与子共占一个额外 CUDA PID。
5. 初始化额外保存极小 `initialization_rng_stages.pt`，记录 trainer 构造前后 CPU RNG 与 events 模块是否已导入。今后的精确失败会另存 `mismatch_rng_<trace>.pt`，只含 expected/actual RNG，不重复存模型或训练原图；原断言仍会抛出。

资源方面，根已指出该 attempt 进程树 RSS 超过其 32,768 MiB 预约，计划将诊断预约改为 65,536 MiB；旧同进程保留分配缓存的问题由阶段进程退出改善，但不能代替新的实测。该预约调整仍由统一调度器完成，我没有启动或停止任何服务器 GPU 作业。

## 本地验证与边界

- 新 `test_verifier_process_isolation.py`：6/6 CPU 测试通过。包含真正 fresh Python 子进程重放一次性导入 RNG、顺序阻塞派发、父 CUDA 已初始化时拒绝、子失败不升格成功、错误 phase/config 回执拒绝、小 RNG 失败证据保留且原精确断言仍失败。
- 原 verifier `--self-test`：4/4 通过，精确 tensor/tree/NaN 字节和 RNG observer 行为保持。
- 新原始测试输出保存在 `compatibility_process_isolation_cpu_attempt1.log`。

没有新真实 GPU PASS、没有放宽任何精确比较，没有认定 N/C0 整段 E200 等价。根应部署新 release，并用新 attempt 和实际 RSS 预约继续验证；先前失败 trace 永久保留。
