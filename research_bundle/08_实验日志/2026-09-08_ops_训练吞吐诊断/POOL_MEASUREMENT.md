# 首次GPU分块pool测量

**两个固定合成形状下，block16算子前向与raw-score反向分别为原实现的约7.47倍和11.66倍速度；数值在预设容差内，但不是bitwise等价，也不是完整训练提速倍数。**

94 pinned torch2.10.0+cu128；固定block16、warmup2、每形状7次旧/新交替测量、显式CUDA同步。仅替换前景/背景masked log-mean-exp的分块执行，原教师/参考/学生输入、掩码及随机状态不变。没有优化器更新。

|合成pool形状（对象×类别×anchor）|旧中位|block16中位|算子速度比|
|---|---:|---:|---:|
|32×5×6400|29.704ms|3.976ms|7.47×|
|128×5×1600|92.177ms|7.908ms|11.66×|

所有valid掩码exact、RNG不变；输出最大绝对差约1.907e-6、学生梯度最大约1.431e-6，满足预先atol1e-6/rtol1e-5，浮点归约次序不同。教师、参考与学生输出均参与检查。尚未验证真实batch完整选择器、完整native/KD/模型梯度、optimizer/EMA及24次更新，不能据此准入替换正在跑的release。

调度使用既有global lease，在GPU4作为第二个项目CUDA任务执行；初始4096MiB为有界测量上限，实测NVML480MiB、RSS1138MiB，最小整卡空闲13314MiB，无资源监测错误。无第四张物理卡，无其他训练暂停或改写。

服务器：`/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_throughput_20260908/pool_probe_attempt1`。
本地：`remote_pool_probe_attempt1/synthetic_attempt1/receipt.json`、`queue_attempt1/`完整准入与资源回执、脚本副本。原实现保留在release_gpu5，独立候选位于本条目performance_candidate。
