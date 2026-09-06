# 94 RGBIR对象判别蒸馏新增目录回执（2026-09-06）

本地对应：E:/SHARE/光sar/08_实验日志/2026-09-06_train_RGBIR对象判别蒸馏首轮/README.md。

新增94目录：
- `/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_object_evidence_v1_20260906/`：allocation、只读发布副本release_v1/release_v2、CPU/canary证据、正式队列日志与状态。
- `/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbir_object_evidence_v1_20260906/`：两个canary、paired正式run，weight0随后同GPU串行执行。

固定物理GPU4；screen rgbir_oev1_queue_s42。原始历史results/receipt/checkpoint/数据文件没有移动、覆盖或重命名。Ultralytics因新IR训练数据加载生成了标准labels/train.cache，这是派生标签缓存，不改变标注内容。所有新训练输出在数据盘，凭据不落共享盘，不生成hash。

full使用release_v2；与canary release_v1相比trainer/loss/loader/config直接字节比较一致，队列补充异常状态落盘，eval补充指标单位。两版保留，不覆盖已运行代码副本。
