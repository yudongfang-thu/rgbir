# 独立蒸馏跟进与补评逐类接口（2026-09-07）

**17:06 更新：补评逐类接口已获独立 ACCEPT，六端点完整接入；16:53 三个 C1 seed 均进入第5轮，原进程和全局 lease 有效。没有新 C1 完整 AP。**

[独立审阅回执](../posthoc_class_adapter_independent_review_v1/review_receipt.json)与[源码绑定核对](adapter_acceptance_binding.json)已完成。接受范围只限补评逐类接口；CLI 保留非自动授权预览身份，不能据接口接受扩展方法主张或忽略 C0 的专项复核。

## 目的与设置

继续用户冻结的独立 C/L 计划，核对实际任务并补正式结果分析缺口。没有新增 GPU 任务，没有改变 C1 的 λ、随机流、recipe 或端点。L 的严格几何条件保持不变。

## 实际运行状态

|seed|C1轮次|成功更新|AMP skips|实际GPU|
|---|---|---|---|---|
|42|5/200|1713|7|2|
|0|5/200|1672|8|2|
|123|5/200|1693|7|5|

三个已启动 PID 均存在，对应 lease 均有效，无完成/失败回执，不能视作已有新 AP。旧 C0 shuffled42 与 same-modal42 分别进入58/41轮。项目总 RSS 约141.38GiB；三张物理卡2/4/5，单卡最多两个项目 CUDA任务。资源原始证据保留在本轮快照。

## 本次分析实现

新增 [posthoc_class_adapter.py](../posthoc_class_adapter_v1/posthoc_class_adapter.py)，不修改冻结 trainer 或原分析器。旧 N/C0 先通过原端点校验，再显式 opt-in 读取同 checkpoint/seed/dev1469 的实际补评逐类指标。来源路径和补评身份保留在 `per_class_provenance` 中；不往原始旧 JSON 注入字段。

17项本地 CPU 测试和94 pinned CPU测试通过。六个真实端点全部载入，每个五类，五汇总指标与旧值保持精确相等。接入原已接受对象诊断后，冻结 `harm_review(records, 'C0')` 返回 `REVIEW_REQUIRED` 且 `missing=[]`；背景误检三seed增加的结论未改变。此处没有新的科学增益或 C1 完整端点。

首轮测试夹具复制 unittest 输出流失败，以及首轮接口误读 receipt 内 seed 层级的失败均已记录，修复仅涉及新适配器和测试。[开发记录](../posthoc_class_adapter_v1/DEVELOPMENT_FAILURES.md)。

外部审阅服务返回402余额不足，未得到该服务的科学审阅；继续由已有分类审阅责任独立复核。原分析输出保留 DRAFT，不能把已有观察桥接当成新适配器的自动验收。

## 产物与下一步

- [只读状态与真实接口探针](summary.json)；[C1原始状态](../formal_live_20260907_165319.json)。
- [全局快照](../snapshots/2026-09-07T165357.033246_0800/snapshot.json)及同目录原始 leases。
- [新接口真实输出](../posthoc_class_adapter_v1/actual_analysis_attempt2.json)；[94 CPU测试](../posthoc_class_adapter_v1/pinned_cpu_receipt.json)。
- 94小体积代码和测试镜像：`/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_independent_kd_v2_20260907/posthoc_class_adapter_cpu_v1`。CPU测试不代表GPU训练canary。

等待 C1 E200，之后核验实际评价源码、固定完整 dev 端点及三 seed 效用/损伤门。C0的后续自动扩展仍暂停，其已有训练继续。L仍缺合格物理对应证据；不得据此宣称定位无效。
