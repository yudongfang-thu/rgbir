# 21:46 增量证据独立发布审查

> 结果与内容检查通过；21:57 首轮导航检查仅有本审查和两份扫描报告的三个待生成链接，需复制报告后再核对。无方法有效性或 accepted analyzer 升级。

## 新结果的独立复算

直接在待发布仓库根执行本包 CPU 脚本，两条结果检查均通过，没有借用本地 `09_外部审计_rgbir` 或服务器 checkpoint。

- OEv1：weight0 seed0 mAP50–95=54.346185，paired42=54.658162，paired123=54.636847；三个固定 E200 last/EMA 端点有效，但同 seed 完整对为 0/3。三份 train/eval receipt、metric snapshot、引用源码/配置/roster、fraction 单位与原始数值均通过；val roster 为 1469 个唯一条目且三个端点逐内容相同。
- 同 seed 历史 native 背景差分别为 −0.011463、+0.842606、+0.949415 pp；仓库正确标为历史背景，没有把跨 seed 两个 P 与一个 N 拼成净收益。首轮遗漏大 manifest 的失败检查及补采来源保留，最终通过产物明确标名 `review_oev1_endpoints_complete.json`。
- OS-SSL：保存的 E200 CSV 逐 Decimal 复算，paired123−shuffled123 为 +0.66900 mAP / +0.49500 AP50 pp；五个完成/历史 CSV 均为连续 1–200 轮、seed 匹配，内部两臂实际 args 仅 model/name/save_dir 不同。+0.495 不宣称通过 +0.5 门。
- CSV 与独立 last/EMA 评估分开；native 初始化混杂、RGB-only SSL 缺口、仅一次每臂 SSL 预训练和三 seed 尚未齐备都在新入口明示。新内容没有据正面单次差值升级配对因果、选择有效或避免负迁移结论。

## 材料保持性与发布范围

21:57 首轮扫描以原分支 HEAD 为基准，检查 400 个新增/改动路径。376 个有本地来源的非 Markdown 复制文件逐字节相等。已跟踪改动只有一个总 manifest、根入口/范围文档、研究日志 README 与工作区入口副本；没有旧原始 JSON/CSV/Python/配置被改写或删除。

复用早晨有界凭据规则，新增材料中候选命中为 0；没有权重/凭据类文件，没有大于 20MiB 的文件。此为有限规则检查，不能证明排除了所有可能敏感模式；没有刻意访问凭据，也没有生成新 hash。初次扫描全包 1193 文件、140,534,165 字节，最终文件数量以发布回执为准。

CommonMark+表格解析覆盖 118 个 Markdown、326 个本地链接/图片目标，其中 19 个图片；没有嵌套普通链接或绝对本地链接。仅三项待生成目标为 `publication_checks/update_20260906_2146/` 下的 `publication_review.md`、`publication_scan.json`、`markdown_navigation.json`，即本次待复制的审查产物；其他导航有效。

人工复核了 LATEST_RESULTS.md、README、阅读指南、审计提示词、范围说明和两次审计的新状态。新文档优先链接 21:46/17:30 证据，同时保留 08:53 旧快照且解释快照非实时状态。LATEST_RESULTS.md 的 OEv1 仓库根复算命令已实际执行通过。

## 边界

这是新证据的发布和解释检查，不是训练复现实验，也不检查本包未收录的 checkpoint 张量。未启动推理或 GPU 任务，未更改训练、阈值、队列或 GitHub 克隆。本报告不验证外部论文或网络链接、不替代三 seed 与四臂归因要求。
