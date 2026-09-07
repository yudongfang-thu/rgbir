# 实际统计频率吞吐探针

**这是root明确授权的性能补测入口，三臂各24次成功更新；不重做状态等价验证，不做dev/AP，不放行E20。**

`short_profile.py` 使用现有三个E20 YAML及原build_trainer。N/C0保持原计算；C1经显式`--candidate-source selected_only_v1.py`的criterion_factory共同绑定选择与损失。一个CLI仅一臂；三臂的启动顺序、globallease及8192MiB/RSS32768MiB预约由root负责，本包不排队。

保留原首次sanity组合梯度检查；自第二次调用开始sanity=False。原前三批/每100批日志与fixed-epoch shared gradient observer仍按原规则运行。max_steps=24仅复用原真实更新停止计数，最多96 attempts；另外包裹optimizer.step计数，要求24次实调用而非把AMP skip当更新。旧原生canary初始化仍保存initial_student.pt和首批first_batch.pt；均在6批热身界限之前，没有新增逐步参数/梯度/optimizer/EMA大拷贝。

C1结束还必须满足thin_learning_batches等于实际batch数、fallback_batches=0，记录full_diagnostics_batches；每批也留这些累计计数。若触发完整原路径回退，失败保留，不会把回退路径吞吐称作薄路径速度。N/C0不要求这些候选专用计数。

在第6批结束与第24次成功更新所在批结束CUDA同步，两点之间连续wall跨度包含批间loader等待、正常统计/日志、计算和callback工作，不扣任何选择/统计成本。中间每批仅记录CPU时钟、文件序列、GT数、LR/accumulate、更新/skip/EMA计数；中间时间差不是孤立GPU latency。另记录完整探针时间、框架峰值及原guard NVML/RSS峰值。

通常只有30批，所以热身后窗口可能没有第100批统计；该事实在回执明确标注，短测不是完整epoch，不把0次观察外推为长期0开销。首个epoch的shared observer在热身内，长期周期成本须在预算余量中计提。所有三臂用同样窗口/统计规则，源代码、配置、初始化及候选以字节副本/stat留证，不新算hash。

```text
PINNED_PYTHON short_profile.py --reference-dir RELEASE_GPU5 --config E20_N_OR_C0_CONFIG --output NEW_DATA_DISK_PROFILE
PINNED_PYTHON short_profile.py --reference-dir RELEASE_GPU5 --config E20_C1_CONFIG --candidate-source ACCEPTED_SELECTED_ONLY_V1 --output NEW_DATA_DISK_PROFILE
```

新增运行依赖：short_profile.py、screen_common.py、train_short_screen.py及原三个DRAFT YAML；C1还需已接受selected_only_v1.py和其实际依赖，原完整release_gpu5与pinned环境不变。输出为short_profile_receipt.json，status=SHORT_SCREEN_THROUGHPUT_PROFILED，不能冒充训练完成、AP或生产准入。
