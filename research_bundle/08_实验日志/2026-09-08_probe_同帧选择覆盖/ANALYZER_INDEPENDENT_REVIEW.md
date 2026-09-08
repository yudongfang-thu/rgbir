# 独立轻量接口复核

READY：只读检查当前 `analyzer.py` / `test_analyzer_cpu.py`，独立运行 4 个已知真值全部通过，见 `ANALYZER_INDEPENDENT_CPU_attempt1.json`。未读取真实 probe 结果、AP、模型权重，未使用 GPU 或计算 hash。

逐图/all RGB/matched 分母、T own-GT 主状态与 T-to-RGB 次状态、unpaired 为 unknown、原 any-candidate gate 与一对一 assigned state 分开，均与冻结 exporter 接口一致。全 batch rho=.5、q 降序、matched-order tie、1 起 selected rank 与实际 core 相同；JSON 合同 `levels=[0,1]` 与 dataclass tuple 经 JSON 后一致。

全计数直接与原 selector counts 对照；reference-candidate 原 primitive count 未被当作 regions 后的累计数。累计首次流失有固定排序和非因果限定。空对象分母使用 null 而非 0 率。32 图单个 train batch 不声称全量/AP oracle 覆盖或蒸馏增益。

本回执只接受合成真值与接口范围；真实 pinned 源绑定、稳定源 GT trace、一次真实 forward 的完成与全状态不变证据，由 producer 完成回执及 root 验收提供。
