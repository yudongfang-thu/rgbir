# 当前 C1 / 旧 C0 内容控制的暂停与恢复边界

**现有入口不支持把在跑任务从 `last.pt` 无损恢复后迁移 GPU。`last.pt` 不是完整训练状态快照；中途改损失代码、结束进程再换卡，不能继续宣称是原冻结轨迹。保留进程的原地短暂停顿与 checkpoint 重启是不同操作。**

2026-09-08，审阅者 `/root/loc_stress`。本轮只读已保存源码、运行参数和小产物，未加载 `.pt`、未创建 CUDA 上下文、未停止/暂停任务、未修改训练代码或配置，也未计算新 hash。本文列出操作选择，不代表已执行或发放新的实验准入。root 已统一采集当前 installed 源码和五任务 checkpoint stat，入口为 [source_and_checkpoint_stat.json]（服务器/本地保留，未包含于本阶段发布：runtime_sources/source_and_checkpoint_stat.json）。

## 1. 现有两条入口与协调器

- C1 实际部署快照：`../2026-09-07_train_IndependentKD实施/remote_admission_1532/formal_C1_gpu5_attempt2/runs/C1_seed42/implementation_snapshot/train_independent.py`。只接受 config/output/arm/source/seed/max-steps，没有 resume；`run()` 要求输出目录不存在，随后从原 cfg.model 构造 trainer。把已训练权重改成 cfg.model 只是更换初始化，不会恢复 epoch、optimizer、EMA 或 loader 状态，也改变冻结模型身份。
- C1 coordinator：同目录上层 `remote_admission_1532/formal_campaign_gpu5_attempt2.py`。`worker()` 发现未完成 run 已存在即报 `Incomplete prior training attempt requires explicit technical review`；没有从 last 自动重启功能。`state.paused` 只阻止后续入队，不冻结已经启动的 CUDA 子任务；改变这个值不能给现有任务腾出资源。
- 旧 `c_shuffled` / `c_same_modal` 由 `release_v8/train_task_conditional.py` 启动，随后由各自顺序 dispatcher 负责评价。其本地源码 `03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_task_conditional_v1/train_task_conditional.py` 同样无 resume，输出目录 `exist_ok=False`。dispatcher 的既有日志也用独占创建；不能通过重新运行同一个 job 覆盖/续写已有 attempt。
- 两条路径继承 `legacy_oev1/train_object_evidence.py` 的 `ObjectEvidenceTrainer`，其保存覆盖仅做 live model/EMA 的有限性检查，然后调用 pinned 父类保存。NaN 自动恢复被该子类显式关闭。

## 2. checkpoint 到底保存什么

已读本轮现场 [engine_trainer.py]（服务器/本地保留，未包含于本阶段发布：runtime_sources/engine_trainer.py），`save_model()` 第 709 行起，`check_resume/_load_checkpoint_state/resume_training` 第 957 行起。它来自真实 `environments/sn6-int8-kd/lib/python3.10/site-packages/ultralytics/engine/trainer.py`。与先前 `../2026-09-07_audit_native与蒸馏载体/native_raw_evidence/pinned_engine_trainer.py` 统一 CRLF/LF 后全文相等。采集脚本采用 read_text/write_text，在 Windows 写入了 CRLF，因此本地副本不是字节相同拷贝；服务器原始 size/mtime 单独保留。下表是当前源码契约，不冒充本轮对 `.pt` 内部字段逐个实测。

|状态|当前保存/恢复行为|对恢复的含义|
|---|---|---|
|当前优化中的 student 参数|checkpoint 字段 `model=None`；保存的是 half 精度 EMA|无法从该文件找回保存瞬间的 FP32 live student；底层恢复模型来自 EMA|
|EMA|half 模型及 `updates` 保存；恢复后转 float|保留 EMA 端点与计数，但 half 序列化已经丢失部分精度|
|optimizer|保存 `convert_optimizer_state_dict_to_fp16(deepcopy(...))`；恢复 load_state_dict|有优化器状态，但不是 FP32 状态的逐位保存|
|AMP scaler|保存/恢复 `scaler.state_dict()`|该部分具备恢复入口；不等于完整 AMP 训练轨迹已保存|
|epoch / best_fitness / args / CSV|保存；start_epoch=checkpoint epoch+1|支持框架的 epoch 级继续训练，不支持从当前 batch 精确接续|
|scheduler|没有独立 scheduler state_dict；重新创建并设置 last_epoch|本冻结 LambdaLR 可按 epoch 重建日程，但不是全状态快照|
|Python / NumPy / Torch CPU / CUDA RNG|checkpoint 未保存；构造 trainer 时重新 init_seeds|相同 seed 不是恢复中断前 RNG 的位置|
|sampler、DataLoader generator、iterator、worker RNG、prefetch|没有保存/恢复|单卡 InfiniteDataLoader 重建，会重置排序/worker 增强流，不能声称原批次轨迹相同|
|未提交梯度与 last_opt_step|没有保存；训练开始 zero_grad，last_opt_step=-1|梯度累积边界不能精确接续；即使在 epoch 边界也不承诺没有跨边界未提交状态|
|KD 自定义状态|criterion.calls/selected_total、real_updates/update_attempts/AMP skips/batch_visits 等未写进父类 checkpoint；criterion 被去除|日志和判据累计量会重置；部分路径把 seed+calls 传给选择/诊断，不能假装调用序号连续|

训练加载器来源：[本轮 data_build.py]（服务器/本地保留，未包含于本阶段发布：runtime_sources/data_build.py），统一换行后与此前已执行 evaluator profile 的 `002_build.py` 全文相等。它创建持久 InfiniteDataLoader，预取系数 4，generator 固定初始化，worker 从 torch.initial_seed 派生 NumPy/Python seed；没有用于恢复 worker/prefetch 的序列化接口。当前现场旧控制 [resource_dispatch.py]（服务器/本地保留，未包含于本阶段发布：runtime_sources/resource_dispatch.py） 也与本地审阅源码统一换行后全文相等。

因此，直接调用底层 Ultralytics resume 最多得到“从一个 EMA checkpoint 继续优化”的技术行为，不能证明等同从 live student、同 RNG 和同 minibatch 状态继续。项目当前入口没有接入它，也没有已接受的 resume 等价测试。不能为了迁卡临时加一个 `resume=True` 就把它记成原 run 连续完成。

## 3. save_period 与 last 的含义

实际 C1 seed42 的 `args.yaml` 明确 `save: true`、`save_period: -1`、`resume: false`、`time: null`、batch32/nbs64/workers4。旧控制的已执行 canary args 和共用 trainer 也采用同值；本轮没有重新采集旧控制正式 args，不把 canary 文件说成现场正式 args。

`save_period=-1` **只关闭额外的 `epochN.pt` 周期副本，不是关闭保存**。父类在每个完成 epoch 写 `last.pt`，fitness 与 best 相等时也写 `best.pt`。本项目训练内 validate() 返回常量 fitness=0，因此 best 不构成独立的 dev 最优端点，不能用它绕过固定 E200 last/EMA 规则。

父类用 `Path.write_bytes` 更新 last/best，没有临时文件加 atomic rename。一次 stat 只能证明该时刻存在一定大小的文件，不能证明内容可恢复；更不能在正在写 checkpoint 时杀进程。中断发生在 epoch 内会损失本 epoch 尚未保存的进度，并遇到上表的状态缺失。当前训练子类 `final_eval()` 是空实现，不能机械套用父类最终 strip_optimizer 行为来推断这些文件已经去除了优化器；本文没有读取 `.pt` 确认字段。

本轮 root 实际 checkpoint stat 如下，时间由原 `mtime_ns` 转为北京时间；best 与 last 大小相同，best 的 mtime 晚 8–12ms，符合常量 fitness 时同一次 epoch 保存两个文件的路径。

|任务|last.pt 大小，bytes|last 最后修改时间，2026-09-08 北京时间|best 比 last 晚|
|---|---:|---|---:|
|C1 seed42|10,755,731|01:57:46.529|12ms|
|C1 seed0|10,755,539|01:42:20.357|12ms|
|C1 seed123|10,755,667|01:44:02.191|8ms|
|c_shuffled seed42|10,759,187|01:47:52.591|12ms|
|c_same_modal seed42|10,758,803|01:56:32.744|12ms|

这些时间说明当前确有最近的 epoch 级文件，不是训练开始后一直不保存。每完成 epoch 都执行保存由当前 trainer 源码确认；这份单时点 stat 没有每个历史 epoch 的 mtime 序列，不能独立证明历史每次写入成功或断言 checkpoint 内最后 epoch 数。本轮既未反序列化文件，也未重复 SSH 取证。

## 4. 可执行选择及限制

|选择|是否保持当前进程/协议|操作边界|
|---|---|---|
|保持五个训练进程，先处理外部 CPU/IO 竞争、未来任务排队和资源分配|训练代码、数据流不动|这是当前风险最小路径。具体是否有效以本次利用率诊断为准；不能凭感觉得出 GPU 一定是瓶颈|
|原 GPU 原进程原地暂停后继续|进程不退出时，模型/optimizer/RNG/loader 可留在内存；没有 checkpoint 重启|若 root 决定短时试验，应只识别并控制所选训练 PID 及其 DataLoader 后代，保持 guard/lease/协调器运行并留回执；随后在同进程 CONT。本文未执行、未验证这个操作|
|结束旧控制以让出 GPU，之后从 last 恢复|不保持原状态|不推荐当作无损暂停；旧对照会缺 E200、归因完成时间后移，恢复另需正式技术方案和审阅|
|把正在跑的 C1 改到其他 GPU|现有进程不支持 CUDA 上下文热迁移|CUDA_VISIBLE_DEVICES 在启动时绑定，修改文件或环境不能移动已建立上下文。退出重启会遇到上述恢复缺失，不能称当前冻结 attempt 无损迁卡|
|中途改 C1 loss/selector/AMP/batch/workers/augmentation/compile|改变冻结实现或轨迹；修改磁盘源码也不会替换已加载 Python 函数|原 run 保留。若有加速实现，应另存版本，做真实算子/梯度及多 batch/update/EMA/随机流等价验证，再决定未来新 attempt；不得把中途代码混用记成单一已验收 release|

原地暂停也**不会释放 GPU 显存、host RSS 或已有 lease**，被暂停 CUDA 进程仍占项目任务/卡数；其 GPU 不能算完全空卡。它只能暂时减少计算和数据读取竞争，不能自动满足“搬到第四张卡”或腾空卡的条件。独立 dispatcher 仍监测资源；不能暂停/杀死 guard 来隐藏占用。源码未见基于训练进度超时自动恢复机制，但现场包装层与整个进程树仍需在操作前核实。没有真实暂停/恢复演练，故不作逐位等价保证。

如果目标是释放整张卡并保留严谨连续训练，需要额外实现完整状态 checkpoint：FP32 live student 与 EMA、原精度 optimizer/scaler、所有 RNG、梯度累积及 batch/epoch位置、loader/sampler/generator/worker预取状态、KD计数和配置/源码/lease转换回执，并验证不中断与中断恢复轨迹。当前保存文件不能事后补回这些缺失状态。该工程与额外演练没有在本轮启动，也不改变现有训练任务。

## 5. 本次核查范围

已独立读 C1 已执行部署快照、持久 coordinator、旧控制入口/dispatcher 及启动 manifest，并用本轮现场 installed trainer/loader/旧控制 dispatcher 源码确认保存/恢复逻辑未变。五任务 last/best 实时 stat 已核实并列上表；没有验证 checkpoint 负载可读性。吞吐、物理卡位置及进程归属以 root 的本轮 host_profile 为准，不拿 9 月 7 日旧快照当 9 月 8 日现场观察。当前没有新 resume/canary 结果，没有 accepted resume analyzer，也没有把任何任务改成停止或失败。
