# 夜间更新独立结果与发布复核

> 结果复核已通过：21:46 快照为 3/6 个有效 OEv1 固定端点、0/3 个完整同 seed 配对。新增 N0=54.346185、P123=54.636847 mAP50–95；GitHub 暂存材料的发布检查仍待齐备。

本次独立复核只读本地证据与 GitHub 克隆，写本目录 review 产物；不更改方法、队列、原始结果或克隆，不占 GPU，不接触凭据，不生成新 hash。

## 本次独立复算结果

来源采集时间：2026-09-06 21:46:12 +08:00。

| student seed | arm | 固定 E200 last/EMA mAP50–95 (%) | 相对历史同 seed native 的背景差 (pp) |
|---|---|---:|---:|
| 0 | weight0 | 54.346185 | −0.011463 |
| 42 | paired | 54.658162 | +0.842606 |
| 123 | paired | 54.636847 | +0.949415 |

三个完成端点的身份、预算、checkpoint 路径、train/eval COMPLETED receipts、metric snapshot 内容、引用源文件、fraction 指标及 collector 数字逐项通过。三个评估 roster 均为 1469 条、全部唯一且逐内容一致。未访问本地没有收录的 checkpoint 张量，服务器 checkpoint 存在性来自原收集器检查。

以上历史差只供背景，不能当本轮 P−N，也不计算 P123−N0 等跨 seed 差值。N0 与历史 N0 接近，仅说明这个 seed 没有显示明显的 native 改动上涨；它不能替代仍未完成的 N42/N123。已有两个 paired 都超过历史同 seed 参照，仍不足以称三 seed 稳定净收益。

三个完成 run 的训练 CSV 都是 header 15 列、E200 最末行 8 列；独立 eval JSON 不依赖该行。实际更新尝试均 56722，N0 成功更新 56697/AMP 跳过 25，P42/P123 成功更新 56695/跳过 27；固定 E200 不等于每个 run AMP 跳步次数相同。

复算脚本：[review_oev1_endpoints.py](review_oev1_endpoints.py)，当前通过产物：[review_oev1_endpoints_complete.json](review_oev1_endpoints_complete.json)。首次 [review_oev1_endpoints.json](review_oev1_endpoints.json) 在补齐大于 collector 2MiB 上限的 train manifest 之前执行，三个 train 引用存在性检查失败；原指标、评估引用和其他检查均通过。补充来源见 [receipt_supplement_sources.json](receipt_supplement_sources.json)，保留首次失败记录以追溯采集缺口，不把它解释为训练或评估失败。

脚本仅使用 argparse 的 `--snapshot`、`--historical`、`--output` 三参数；发布后可将历史输入指向仓库根 `02_raw_results_dronevehicle`，不依赖 E 盘或本地 `09_外部审计_rgbir`。

OS-SSL 的 [独立复算脚本](review_osssl_numbers.py)与[通过产物](review_osssl_numbers.json)同时核对五个完成/历史 run 的 200 连续轮、seed 和 CSV 数字：paired123−shuffled123 为 +0.49500 AP50 / +0.66900 mAP pp；两臂实际 args 仅 model/name/save_dir 不同。此为保存 CSV 精度的 Decimal 运算，无法恢复完整浮点 evaluator 值。+0.495 不四舍五入成通过 +0.5，单微调 seed 也不满足三 seed 门。初始化混杂、缺 RGB-only SSL 和未做独立 last 评估的限制均保留。

## 结果解释检查条件

1. OEv1 新端点需同时有 E200 completion、固定 last/EMA val 指标及 train/eval COMPLETED receipts，引用源码/配置/roster/指标快照齐全。复制的 metric snapshot 与原 JSON 应逐内容一致，评估 roster 应为 1469 个唯一条目。
2. 用原始 evaluation_val.json 的 fraction 数值乘 100 得到百分数；主量只在同学生 seed 的 paired 与 weight0 全部完成时计算。P42 与 N0 等跨 seed 差值不能当蒸馏净收益。
3. 仅三对 0/42/123 全部有效才计算 mean±sample SD（ddof=1），不把不同种子的单臂到达顺序当结果优劣，也不填补未完成端点。教师与 RGB reference 仍固定 seed42，已有跨 seed 数据顺序/首批增强重合限制保留。
4. 历史 native 是描述性背景，不能替代本轮同代码 weight0。训练 CSV 最终列数缩短和 progress.json 的旧 running 字段不作为 OEv1 性能或仍在训练的判断依据；完整 completion/eval receipt 优先。
5. OEv1 P−N 即使为正，也只支持整套干预的净差，不自动证明跨模态独有信息、选择优于随机或避免负迁移。尚缺 shuffled、same-modal、同剂量随机选择等归因控制；不升级 accepted/confirmatory claim。
6. OS-SSL 明确区分 E200 CSV、best 末尾输出与固定 last/EMA 独立评估。不同口径不混表，跨 seed 不相减；共享非骨干初始化的三 SSL 臂可作同 seed 内部比较，但旧 W1 native 的检测头初始化不同，不能把对它的正差归因于 SSL。
7. RGB 下游的自模态控制应为 RGB-only，现有 IR-only 不替代。OS-SSL 迁移门 +1.0 AP50 与旧文 +0.1 勘误保留；并未根据新数值改门槛。
8. 所有进度带独立采集时间，训练/评估完成状态分开；新失败、待补证与原始快照保留，不以重写历史记录掩盖偏差。

## 发布检查条件

- 扫描新增材料的凭据模式、排除权重/原始数据集/凭据文件，保留源码与历史 receipt 内已经存在的身份记录。
- 检查新增 raw JSON/CSV/Python 与来源字节一致（允许 Git 换行处理解释）；旧原始证据不改写。Markdown 只在导出副本修正导航、添加新状态。
- 检查 Markdown 渲染目标存在、无绝对本地链接与嵌套普通链接；保留远端证据路径为代码文本。
- 入口 README、阅读指南、审计提示词应从 08:53 旧快照更新到本次状态，并链接本次与晚间审计；原快照按时间留存。
- 复用早晨 review_publication_safety.py 和 review_markdown_navigation.py；自动检查结果仅是有限规则范围内的发布检查，不称无条件安全证明。

## 已读来源

晚间审计 README / oev1_review_notes.md、OEv1 三 seed 扩展 README 与 analyze_three_seed_endpoints.py、OS-SSL-IR 迁移 README，以及早晨两个发布检查脚本。
