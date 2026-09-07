# 六个 legacy N/C0 端点补采的独立审阅

**结论：代码范围接受，可由根按现有授权放入统一评估队列。15/15 CPU 测试独立复跑通过，未发现阻塞问题；本审阅没有启动任何 GPU 或任务。**

日期：2026-09-07。审阅者 `/root/review_c1_spec` 未编写或修改这四份候选文件；作者 `/root/review_l1_spec` 确认稳定后审阅完整源码、实际六项候选及测试，并保存独立原字节副本。

## 所接受的补采路径

- 原 `weight0/paired × seed0/42/123` 六个 E200 last/EMA 模型保留各自真实身份。旧 shared YAML 的 seed42 默认值不被当作执行 seed；新派生评价配置只补实际 seed/arm/source 等诊断元数据，旧训练配置和回执原字节保存在带原路径索引的 origin_evidence。
- 原 run/last/completion/train-eval receipt/metric/roster/config 通过真实 legacy loader 和附加同 recipe 检查，服务器运行时继续验证物理 checkpoint、当前数据 YAML/完整 dev roster、已实测 profile 的实际源/配置绑定及 pinned 版本。不存在新造的训练 completion 或根 run_evidence。
- 只进行一次 evidence 前向，实际调用冻结 `make_evidence_validator`、`capture_contract`、`verify_population` 和原生 metric_record。原生 kwargs 不变，on_val_start 记录并核对实际精度/NMS/rect/批量等设置。新增 wrapper/helper 原源码诚实进入新 eval receipt，不伪装成历史 canonical 七源。
- 新指标要求五类完整、1469 唯一图像及实际 seen/loader/objects；metric、对象原字节和 contract/config/roster 进入本次实际 eval receipt。现有对象分析器的真实文件读取接口通过 CPU 合成完整名单布局检查，不要求伪造一次新训练。
- 五个新旧总指标必须逐项完全一致才发布 completed metric。单指标差异保留 observed、comparison、objects、contract 并拒绝完成；原结果和失败 attempt 不覆盖。完整队列成功回执最后写出，receipt 写入失败不能由 dispatcher 误判已完成。失败后是否建立新 attempt 由后续技术复核决定，包装器不自动重启。
- 六项 job 都是 stage=evaluation，使用真实成功 profile 的原计算路径和全局 dispatcher。固定预约为显存1626 MiB、进程树 RSS10356 MiB，分别来自真实1370/6260 MiB峰值加已确定余量，不能解释为本审阅重新测量。任务不指定物理 GPU，仍受原共享 lease 的卡数、槽位、70%、全卡2 GiB余量和总RSS约束。

## CPU 验证与限制

`D:/Anaconda/envs/KGJ_proj/python.exe -m unittest test_legacy_endpoint_eval -v`：15/15 通过，实际输出见 `cpu_tests.log`。包括六份真实旧 metadata/实际 seed、recipe/arm 错配拒绝、原 metric 快照不符拒绝、物理 last 路径、输出不覆盖、原文件/weight stat 改变检测、五指标单位与非有限值、完整五类、真实 profile 预约、六任务配置、独立源码回执字节绑定、新旧差异不发布 completed，以及现有对象分析器实际读取/对象字节篡改拒绝。

这些测试没有执行模型前向。物理 checkpoint 测试是明确标注的临时 fixture；origin copy-layout fixture 只使用本地已下载的旧文件，生产代码仍要求全部服务器原引用存在。对象读取 fixture 的1469行是合成空数组，只证明文件接口，不冒充真实对象表现或AP。对象匹配规则的科学验收仍由它已有的独立已知真值测试与正式分析结果负责。

## 不随这份回执升级的结论

本次接受不重建历史原生库源码字节，不把 N42 的一次 probe 外推成其他五个模型已经逐位等价。六个真实新评估需要实际执行并各自通过比较和回执检查。新诊断目录不能直接冒充 `analyze_independent.load_endpoint` 已接受的完整训练端点；总指标 bridge、逐seed harm 判定、三seed结论和论文归因仍需按独立分析链完成。

## 部署入口

`review_receipt.json` 的四个 accepted_copy 使用相对 `reviewed_sources/<文件名>`。根可将 receipt 和该子目录原样部署到候选代码目录，CLI 的 `--review-receipt` 指向该 receipt；运行时再次核验四份实际源码字节。候选本体和配置留存于 `reviewed_candidate_bundle_v1`。任何改源码、换 release/profile 或改 recipe 需要按实际变化重新核对，不能直接复用本次范围。

只有新派生文档/回执落盘；没有改 NEW、旧端点、原候选状态或原训练进程。
