# 本地预检调用修正

两次修正均发生在GPU任务启动前，没有改已冻结训练实现或配方。

1. 首次pinned unittest discovery导入`test_l3_cpu.py`时未提供其声明的`--reference-dir/--output`，SystemExit 2。该次22项wrapper/queue已通过，原FAIL回执保留；随后用声明的CLI在同一未修改源码上执行14项L3测试通过，终态`PINNED_TRAINING_CPU_ACCEPTED_v1.json`按22+14记录36项，不将失败测试算作通过。
2. 启动器原先把审阅清单元素当字符串，实际是含`relative_path`的字典，路径拼接在任何SSH操作前失败。修正读取字段，旧源保存在`start_training_queue_before_schema_fix.py`；第二次逐字节核对本地snapshot和远端18份运行输入后才启动screen/原lease队列。

后续真正的GPU attempt从`QUEUE_LAUNCH_attempt1.json`开始，技术失败或结果不覆盖。
