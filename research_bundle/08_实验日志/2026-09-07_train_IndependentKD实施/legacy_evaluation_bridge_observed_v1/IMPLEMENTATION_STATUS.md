# 实际复评桥接的生成状态

**最终候选 `prepared_a3` 已生成六份完整 observed DRAFT；9/9 CPU 测试通过，313条独立副本及回执快照别名与实际输入原字节一致。仍未签 ACCEPTED，等待根独立审阅。**

固定输入为 `legacy_diagnostics_snapshot_20260907_161415`。六个相同 checkpoint 的旧/新五指标逐项精确相等，六次均实际观测1469张图、保留五类AP和全部九个实际评价源，六份逐图GT/原始尺寸/实际canvas完全相同。canonical七源从新实际 eval receipt 中提取并与旧候选内核对过的正式 C1 emission 顺序、源码字节一致。两个附加实际源和独立包装器审阅保留。

验收入口见 README 的命令；代码是 `build_observed_bridge.py`，测试是 `test_observed_bridge.py`。当前代码与 `prepared_a3/generator_sources/` 原字节一致。`cpu_tests_attempt3.log` 是最终9项实际CPU输出；`prepare_attempt3.log` 保存完整真实生成argv及输出；`artifact_verification_attempt3.json` 保存独立副本与receipt快照别名的逐项字节核查结果。

首轮真实对象检查暴露旧对象分析器对max640的不适用断言，失败摘要保存在 `cpu_tests_attempt1_failure.log`。本桥接已与对象错误分析器解耦，仅比较实际记录的几何/GT身份，不约束实际canvas最长边等于配置imgsz。实际为544×672；不改写任何原始geometry。

`prepared_a1` 是首份完整DRAFT，保留不改；`prepared_a2` 增加原receipt每个metric_snapshot到独立载荷的显式别名映射，使对象只复制一份仍能复核全部绑定边。旧回执中的原路径没有重写。原输入、旧AP、服务器补评结果、NEW和其它分析器均未修改；本次只有只读结果收集和CPU生成，没有启动GPU。

根对生成器主体的初步审阅认可了九源保留/七核观察等价的范围，并要求明确当前正式分析器不会自动读取补评逐类 AP。`prepared_a3` 只更新 README 的这项说明和新的冻结副本；生成器代码、测试和真实数值均与a2相同。旧a1/a2未覆盖。逐类证据接入仍是后续显式独立审阅的工作，本工具没有静默指标注入。

根审阅后若接受，应另存 accepted bridge 并将分析manifest引用该文件，保留这两份 DRAFT。本生成器不提供接受开关，也不代签正式三seed/harm/论文主张。
