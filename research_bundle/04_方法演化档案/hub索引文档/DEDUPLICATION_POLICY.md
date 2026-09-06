# 去重与归档策略

> 制定日期：2026-07-26。

## 已采用

- 方法版本按 family 合并到一个导航目录。
- 代码、文档、结果和 artifact 均使用相对符号链接，不复制内容。
- 当前工程与 `LADD/LADD_public` 历史工程分层，不再混用 launcher 或结果口径。
- 顶层 7 组逐字节相同论文 PDF 保留描述性文件为正文，旧文件名改为相对符号链接；
  映射与原始 SHA 见 `TOP_LEVEL_PDF_DEDUP_RECEIPT_20260726.csv`。

## 已核验但暂不物理删除

- `LADD_experiment_artifacts_20260704_153834/3090` 的 949 个文件与
  `LADD_3090_migration_20260710_224458` 中对应文件逐字节一致；较新快照还多 2,021 个文件。
  旧包同时含独有的 `90/` 与 `local_context/`，且可能有包级 manifest，因此本轮不破坏旧包。
- `ogsod400_clean_protocol` 内约 3 GiB `.mypy_cache`、约 0.65 GiB `.venv` 和大量
  `__pycache__` 属可重建空间，但方法 source-manifest 是否包含这些路径需逐个验证后再清理。
- `LADD_public/.git` 约 44.6 GiB；已检查 pack 内没有明显重复 object ID，盲目 `git gc` 未必回收空间，
  且该 worktree 非干净状态，本轮不做 destructive Git 操作。

## 绝不自动去重

- `/private/results` 或本地镜像中的失败 attempt、终态 root、receipt、checkpoint 和 analyzer bundle。
- frozen runtime、source lock、campaign lock、registry、claim ledger。
- `.aris/traces`、密钥或凭据；这些应隔离和轮换，而不是纳入研究档案。

## 下一阶段可安全执行

1. 对缓存目录生成“是否出现在 source manifest”审计，再删除明确未绑定的 cache。
2. 为两个迁移包生成 package-level dedup receipt；若确认旧 package manifest 允许，可将重复
   `3090/` 子树迁往外部冷存储，而不是在科研证据树内直接删除。
