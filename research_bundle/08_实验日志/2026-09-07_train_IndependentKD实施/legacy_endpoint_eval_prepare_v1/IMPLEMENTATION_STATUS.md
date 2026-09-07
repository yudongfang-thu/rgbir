# 实现与本地验证回执

结论：六个旧 N/C0 last 的最小诊断补评包装器、六份真实配置和串行队列候选已完成，**CPU 15/15通过，尚未运行GPU，独立源码审阅进行中**。没有更改 NEW/release_gpu5 或原始旧run。

本次实际读取并复用的接口来自：本日志 `EVALUATION_BRIDGE_PREPARATION.md`、`prepare_evaluation_bridge.py`、`remote_admission_1532/formal_campaign_gpu5_attempt2.py`，以及冻结 `evaluate_independent.py`、`evaluator_profile.py`、`resource_dispatch.py`、实际成功 evaluator_profile_a2 resource receipt。

- 生产文件：`legacy_checkpoint_evaluate.py`、`prepare_queue.py`；独立审阅绑定范围同时含 `test_legacy_endpoint_eval.py` 和 `README.md`。
- 真实候选：`candidate_bundle_v1/queue_candidate.json`，6 jobs，seed42/0/123各N/C0；`configs/` 原recipe只加明确本次诊断元数据。使用原completion/checkpoint真实identity，不改原YAML默认seed或kd_weight来伪装历史执行。
- 容量：实测profile1370MiB VRAM/6260MiB RSS，按既有formal评价余量生成1626/10356MiB预约。所有任务调用冻结dispatcher/sharedguard，没有固定物理GPU。
- 独立审阅者：`/root/review_c1_spec`，已收到稳定四文件和测试入口。最终接受只限包装器代码/路径复用，不构成新补评已执行。

本地命令：在本目录执行 `D:/Anaconda/envs/KGJ_proj/python.exe -m unittest test_legacy_endpoint_eval -v`。

测试过程保留说明：首轮13项有1项fixture错误，原因是本地旧snapshot有意未下载训练映射JSON，origin复制测试要求了这个远端才有的大文件。只调整该明确标为synthetic的临时复制夹具，使它使用本地实际已有输入；生产仍要求原服务器所有引用文件存在，未添加静默跳过。随后增加review源变更拒绝和真实发布helper的单指标差异检查，最终15项全通过。

对象接口测试使用明确标为SYNTHETIC的1469行临时对象/回执夹具，实际调用现有 `object_error_analysis.load_evaluation`，证明无需伪造根训练receipt即可读取新eval布局，并检验对象bytes篡改会拒绝。它不是实际模型推理或AP。

正式入队前仍须：部署稳定四文件、六配置及实际独立review回执和副本；冻结dispatcher在94核实际evaluation_profile源/配置/数据binding和资源，然后在screen内动态lease串行入队。包装器每次真实callback核kwargs、全1469对象、五类指标和旧五总指标exact，缺一则留下失败attempt，拒绝完成回执。

本补采先补齐逐类/objects。因目录只含旧训练origin证据且新增包装器源，不能把它直接当作 `analyze_independent` 已接受的完整新训练端点；正式总AP/旧训练来源桥接由独立分析入口继续处理。本次不阻塞已运行的三seed C1。
