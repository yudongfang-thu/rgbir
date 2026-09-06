# 94 只读运行证据快照

> 采集区间为 2026-09-06 08:53:54–08:53:57（北京时间）。OEv1 尚无完整检测端点；运行状态不能作为性能收益结论。本快照在训练继续运行期间采集，不是跨文件原子快照。

## 当时实际状态

| run | CSV 已完成轮数 | progress 当前轮数 | 完整 E200 终点 |
|---|---:|---:|---|
| OEv1 paired seed42 | 110 | 111 | 未完成 |
| OEv1 weight0 seed0 | 21 | 22 | 未完成 |
| OEv1 paired seed123 | 20 | 21 | 未完成 |
| OS-SSL-IR shuffled seed123 微调 | 50 | 未提供 progress.json | 未完成 |

OEv1 另三个配对臂仍排队，不能记成已训练。OEv1 `results.csv` 的 AP=0 是关闭中间验证后的占位。`status_at_snapshot.json` 保留逐 run 的 progress 与最后 CSV 行；正式队列和扩展 worker 状态位于对应 `artifacts/` 目录。表中 CSV 的完成轮数与 progress 的当前轮数概念不同。

## 文件结构和来源

- `runs/rgbir_object_evidence_v1_20260906/`：seed42 的两臂 canary、paired 正式训练及训练时源码快照。
- `runs/rgbir_object_evidence_expand_20260906/`：seed0/123 四个 canary、当前正式训练及源码快照。
- `artifacts/rgbir_object_evidence*/`：实际 release_v1/v2、队列、worker、配置、canary 汇总、状态与日志。
- `artifacts/osssl_ir_20260906/`：实际 manifest、协议、worker 历史版本、注入补丁、SSL 配置及日志。各版本原样保留，不能仅凭文件存在认定已执行。
- `runs/osssl_ir_20260906/`：当时唯一已创建的微调 run 的小文本证据。
- `framework_snapshot/`：实际环境 Ultralytics 8.4.115 的 12 个关键源码文件、发行包 LICENSE 和通过 `importlib.metadata` 读取的版本信息。
- `snapshot_manifest.json`：每个已采源文件的绝对路径、读前后大小/mtime、读取时间、目标文件、完整/部分/压缩标记，以及缺失与省略原因。

采集期间未导入 torch、未运行 GPU 计算、未读取环境变量或凭据，也没有修改原运行目录。输出写到 94 的 `RGBT_campaign/artifacts/github_full_evidence_20260906/`。没有新计算 hash；原文件内既有的 digest 字符串原样保留，不代表本次重新核验。

## 完整性和版本边界

完整 KD JSONL、JSON/YAML/CSV、source_snapshot 均保留。超过 128 KiB 的 stdout 日志保存末尾 128 KiB，文件名带 `.partial_tail.txt`，manifest 记录截取偏移；不得将其当作完整日志。

为控制体积，四份大 OS-SSL manifest 和重复 canary 配对 roster 使用无损 gzip。内容完全相同的来源通过完整字节比较指向同一份归档；每个来源仍独立登记。旁边的 `.all_rows.tsv` 包含全部 pair/object 文件名和 donor/group 身份，`.readable_metadata.json` 提供完整路径规则/原始示例。TSV 是可读投影，原字段和完整路径以 `.gz` 解压内容为准。

例如可以用 Python 标准库读取精确原始内容：

```python
import gzip, json
from pathlib import Path
p = Path('artifacts/osssl_ir_20260906/ssl_data/manifest_shuffled.jsonl.gz')
rows = [json.loads(line) for line in gzip.open(p, 'rt', encoding='utf-8')]
```

OS-SSL-IR 的本地实验记录说明 SSL 入口曾由 pyc 恢复运行。此次没有把任意 `.py` 当作该 pyc 的等价源码，也没有反编译/执行 pyc。实际 manifest、下游配置和补丁提供可审查证据，但 SSL 执行源码身份仍存在此边界。历史 worker 日志包含资源等待与失败尝试，应一起审查，不能只读最后成功的 run。

这不是独立可运行的全环境镜像：未上传权重、原始图像、缓存和编译字节码；源文件含原服务器绝对路径。Ultralytics 源码许可见发行包附带的 LICENSE；其他依赖只登记版本，没有复制整个发行包。
