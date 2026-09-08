# 本条目同步脚本准备

**仅准备 [sync_evidence.py](sync_evidence.py)，没有执行 manifest、mirror、publish、Git 操作或远端写入。** 训练源码未修改。

root 更新并冻结条目 README/本地索引后，使用相同 `--state` 与 `--note` 顺序执行，例如：

```
python sync_evidence.py manifest --state RUNNING --note "L3三臂短筛执行中，失败校准与修复证据保留；尚无完整端点结论。"
python sync_evidence.py mirror --state RUNNING --note "L3三臂短筛执行中，失败校准与修复证据保留；尚无完整端点结论。"
python sync_evidence.py publish --state RUNNING --note "L3三臂短筛执行中，失败校准与修复证据保留；尚无完整端点结论。"
```

以上只是调用示例，不是当前运行状态回执。`COMPLETED` / `BLOCKED` 亦可显式选择；脚本不从旧独立 CPU PASS 推断新训练完成。发布 banner 指向本条目 README，不硬编码尚不存在的最终报告或 AP。

远端新镜像固定为 `/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_object_dfl_screen_20260908/review_v1`，只允许新目录。Git 仍为 `research/full-evidence-20260906`；`publish` 仅准备 worktree 文件、条目索引和 `publication_checks/update_20260908_object_dfl/`，不 stage/commit/push。root 独立处理提交与远端检查。

只遍历本条目，目录遍历前排除外部 symlink/junction 与权重/凭据/cache目录，不递归复制引用的外部路径。纳入小源码、Markdown、JSON/JSONL、YAML、表格、文本、`.diff` 以及有界 `.json.gz` / `.jsonl.gz`；压缩 JSON 解压后也检查凭据标记。每文件≤16MB，压缩 JSON 检查上限32MB；超限要求显式整理，不静默上传。

排除权重、原图、完整 `snapshot.json` / host/process/gpu snapshot、`*_resource_profile.json` / `resource_profile.json`、`*_admission.json` / `admission.json`、凭据、pycache；压缩形式同样排除。所有已收原失败、source副本、CPU结果和独立审阅在允许范围内保留。manifest 逐项登记排除原因，不算新 hash。

镜像写入后逐字节核对；publish 要求文件列表/大小/mtime 与镜像时本地快照一致，源或状态变化即拒绝。已有远端 review_v1 不覆盖；如之后还有新增结果，由 root 明确安排下一镜像版本，不改原镜像。Git root README 只新增或更新本条目独有 `对象坐标DFL短筛` banner，保留其他线程的 C1 提速 banner 及其余内容。实验索引 marker 为 `probe_对象坐标分布目标`。
