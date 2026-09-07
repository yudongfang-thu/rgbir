# 94 CPU首轮验收与自然覆盖

**真实自然64批：Drone旧审核清单命中25批，LLVIP仅11批；均未证明定位几何或非零L信号。参考包18测试在pinned环境通过。**

两数据集各2048唯一训练图，B32/workers4，独立seed20260907、真实增强、无重复采样/换批。Drone238唯一审核图中命中26图、25批；LLVIP42图中命中13图、11批，原名单即使全部几何合格也不能在该固定流达到16非零批。

新增单元测试首轮79项中3个error，均在独立analyzer依赖仍指向邻接旧目录，已交作者修冻结依赖后重验；不将该轮写成全通过。包内18项另行pytest通过，不替代GPU兼容/校准/canary。

CPU调度尝试1遗漏零CUDA进程参数，被guard拒绝；尝试2补齐后由现有全局lease成功运行所有CPU阶段；额外API尝试3因缺少acquire的gpus参数未启动工作负载。三份调度日志保留，不覆盖；没有独立GPU任务由这些CPU launcher启动。

本地原始证据：remote_cpu1/；服务器：/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_independent_kd_v2_20260907/。覆盖输出在coverage_{drone,llvip}_attempt1，代码冻结release_cpu1。

下一步按真实自然流命中与train-only静态D2机会生成最多24对审核候选；LLVIP不能只补原名单。正式L仍不准入。分类继续兼容与校准实现。
