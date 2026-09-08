# F-rel-GM 小队列独立复核

限定 PASS：已只读复核 `../feature_gm_release/run_feature_gm_queue.py`，独立复跑8组CPU真值全部通过，回执为 [INDEPENDENT_QUEUE_CPU.json](INDEPENDENT_QUEUE_CPU.json)。作者随后增加两处eval的training_configuration/training_completion路径exact绑定；已最终回读第206–207及267–268行，并核作者 [QUEUE_CPU_attempt2.json](../feature_gm_release/QUEUE_CPU_attempt2.json) 8/8 PASS。没有改作者源码、运行GPU或读取F AP。

队列等待confidence成功时不申请GPU lease；随后只走原global resource_dispatch的固定canary→匹配控制→train→eval，没有校准、自动重试、AP分支或新资源池。初始8192/32768为测量上限；正式训练预算来自canary NVML/CUDA峰值与完整进程树RSS加固定余量，超过固定上限时停止。

跨scope projection对主N/C1分别核公共配置和screen身份、初始化实际stat及已有full-state check、canary和完整训练first30源图/增强/双标签记录exact；候选完整训练再对自己canary核同一流。控制和候选训练/评估checkpoint stat绑定，eval配置/训练完成回执路径exact；不从同dataset推定匹配。收集分析器 additionally 要求候选queue完成且完整训练stream check通过。路径与控制配置/eval副本接口已经一致。

这只接受有界新协议运行与匹配readout的准备。没有证明真实训练必然成功、全参数剂量一致或任何F收益；原F-rel BLOCKED不改。
