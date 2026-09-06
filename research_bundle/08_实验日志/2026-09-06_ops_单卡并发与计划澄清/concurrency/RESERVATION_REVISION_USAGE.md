# 带实测依据的资源预约修订脚本

状态：脚本已编写并完成13项纯CPU合法性检查；本分工未部署、未执行预约修改。纯CPU结果见`reservation_tests.json`与`reservation_tests.log`。

脚本：`revise_profiled_reservations.py`。仅针对R42/GPU2、P0/GPU5两个明确活跃lease，目标8300MiB VRAM、32768MiB RSS。默认dry-run，无写入；必须显式`--apply`才修改。修改经过现有guard共享文件锁，保留owner/job/CUDA PID、绑定、lease数量、原始admission与全部其他字段，不编辑全局guard源码。

## 执行前提

同卡profile的`concurrency_profile.json`、`concurrency_summary.json`与独立profile run的`completion_receipt.json`必须齐全、完成、通过；该profile租约和全部相关进程退出。两个长训必须仍是指定ID/PID/seed/arm/输出目录，full进度epoch 2–199，GPU上没有第三方新增CUDA PID。

正式训练最新guard峰值与当前实占较大者需满足：8300−VRAM峰值≥512MiB，32768−RSS峰值≥1024MiB。RSS检查计入实际guard后代树，包含数据进程。任何条件不符均停止，不能为通过而降低实测峰值。

## 由root部署和使用

上传只上传这一个脚本，使用新的目标名，不覆盖原训练或guard。示例远端位置：

`/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/oev1_concurrency_20260906/revise_profiled_reservations.py`

只读预检查命令：

```bash
/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python /mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/oev1_concurrency_20260906/revise_profiled_reservations.py
```

预检查通过后，实际锁内重新核验与提交：

```bash
/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python /mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/oev1_concurrency_20260906/revise_profiled_reservations.py --apply --revision-id profiled_reservation_revision_20260906
```

提交在同目录生成`*_before.json`、`*_intent.json`、`*_after.json`、`*_receipt.json`。这些文件以独占新建保存，原件存在就拒绝重复运行。若提交中断且只存在before/intent，应先读当前lease核实四个字段；不能盲目重试或删除回执。修改仅是资源期望值，不是模型训练变更，也不改变实际任务内存上限。

随后新任务仍必须通过同一个guard正常acquire，并在已通过同卡profile时使用`--profiled-second-train`；不能把修订脚本当成训练launcher。random0/123的队列接管另由root协调；脚本不杀、不暂停、不干预在跑R42日志管道。

## 检查覆盖

13项检查包括输入不可变且只有四个标量修改、保留非目标lease/policy、profile失败/未退出/未释放租约、精确PID和seed身份、出现新增CUDA进程、长训消失、重复修订、VRAM余量边界、RSS超界、数据进程RSS计提。未做GPU训练测试；现有独立GPU2真实profile由root采集并在实际提交时读取。
