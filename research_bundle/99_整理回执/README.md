# 99_整理回执

> 本目录保存 2026-09-05 结构重整的全部操作回执，保证旧路径可追溯。

## 本轮重整（2026-09-05）

### 一、顶层移动映射（旧 → 新，共 73 项 + 拆除）

| 旧位置（顶层） | 新位置 |
|---|---|
| 全部论文 PDF（53 个真实文件 + 7 个硬链接别名）、RGB_SAR_KD_LITERATURE_ANALYSIS.md、PAPER_FILE_MANIFEST.md | 01_文献/ |
| 4 个 .pptx、draw/ | 02_汇报材料/ |
| ogsod400_clean_protocol | 03_现行工程/ogsod400_clean_protocol |
| SpaceNet6_OTD_official_reproduction | 03_现行工程/SpaceNet6_OTD_official_reproduction |
| TSKD_METHOD_MAINLINE_CN.md | 04_方法演化档案/2026-03~04_TSKD探索/ |
| LADD_EVOLUTION_AND_EXPERIMENTS.md | 04_方法演化档案/2026-04~06_LADD主线/ |
| LADD_0723、AUDIT_CLAUDE_CODE_20260727 | 04_方法演化档案/2026-07_重审与交接/ |
| 6 个服务器/状态总结 md（2026-08-30） | 04_方法演化档案/2026-08_服务器环境总结/ |
| LADD_experiment_artifacts_20260704_153834（+.zip.sha256） | 05_实验证据_按服务器/90+3090迁移包_20260704 |
| LADD_3090_migration_20260710_224458 | 05_实验证据_按服务器/3090快照_20260710 |
| LADD_public_checkpoints_shutdown_20260614 | 05_实验证据_按服务器/autodl备份_20260614 |
| LADD_public_local_archives | 05_实验证据_按服务器/本地归档_LADD_public |
| LADD / LADD_public / CoRe-LADD | 06_历史工程_只读/ |
| 00_RESEARCH_HUB（catalog/README/去重政策/build 脚本/literature README） | 04_方法演化档案/hub索引文档/ |
| 旧顶层 README.md | 99_整理回执/旧顶层README_20260726.md |

### 二、删除（均经确认）

- 4 个 git 仓库共 **44.61GB**：`LADD_public/.git`（44.61GB）、`CoRe-LADD/.git`、
  `SpaceNet6.../reference/official_code/{Hnewa_CMKD,SN6_OTD}/.git`
- 可重建缓存 **4.5GB**（venv/.mypy_cache/__pycache__/.ruff_cache/.DS_Store，审计后删除，
  明细见 `smb_transfer_repair_20260905/`）
- 空/骨架目录：methods、tests、logs、runs_public、scripts、src、CoRe-LADD/tmp
- 旧 hub 链接层（205 个 junction + 133 个文件链接，junction 感知删除，目标数据验证完好）

### 三、链接层重建

- 随目录移动失效的内部链接重定基重建：**4,924 个全部成功，0 失败**（`rebase_log.txt`）
- 退役 268 个 hub 导航链接（其功能由新 README + 04 目录取代）

### 四、凭据（2026-09-05 早前轮）

- `LADD/key`、`key117`、`key2`、`LADD_public/key` → `C:\Users\MSI-PC\guangsar_credentials\`（ACL 已限制）
- 原位置留指针说明；建议轮换 seetacloud root 口令

## 历史回执

- [2026-09-06_OEv1优先级与对比新增.md]（未导出的工作区路径：2026-09-06_OEv1优先级与对比新增.md）：优先级调整、OS-SSL安全停止、CCLKD独立评估及随机选择对照新增路径。
- [2026-09-06_实验全景审计资料新增.md]（未导出的工作区路径：2026-09-06_实验全景审计资料新增.md）：新增方法/设置/结果全景、22:26进度与22:28 OS-SSL独立last证据；仅新增资料及更新索引。
- [smb_transfer_repair_20260905/]（未导出的工作区路径：smb_transfer_repair_20260905/）：SMB 传输链接修复、
  缓存清理、全量盘点（链接计划/执行日志/清理清单/盘点数据）
- `rebase_log.txt`、`cleanup_log.txt`：本轮链接重定基与缓存删除明细
