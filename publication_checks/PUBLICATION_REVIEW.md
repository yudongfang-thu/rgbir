# GitHub 审计包独立发布审查（2026-09-06）

> 最终发布审查通过，可以推送：未发现阻止公开的凭据、原始结果保持性、结论边界或 Markdown 导航问题。本报告不授予实验收益 claim，也不替代 accepted analyzer。安全范围见 publication_safety_snapshot3.json；最终 Markdown 复核时间为2026-09-06 09:11:54（北京时间）。

## 范围

独立只读检查 `C:/Users/MSI-PC/rgbir_review_worktrees/evidence-20260906` 的继承材料和新增审计包，核对拟上传 inventory、入口说明、原始结果身份、运行快照、代码依赖和公开内容风险。没有 Git 修改、GPU 计算、凭据读取或上传操作。

## 发现与处理情况

1. **已处理：旧入口过度概括**。原 README 的“RGB→IR”“监督 KD 从未净胜”“OS-SSL 尚未在 RGBIR 测试”不能作为当前结论。新根入口和指南已按真实模态方向、有限方法矩阵和新增 OS-SSL-IR 记录修正，旧入口保留为 ARCHIVE_README_v1.md。
2. **已处理：历史文件误名**。`02_raw_results_dronevehicle/N_llvip_seed{0,42,123}_metrics_record.json` 的内部 dataset 实际为 `dronevehicle`，也是 CMDistill 对应 W1 native 的来源。新根入口已明确警示，原 JSON 保持不变。
3. **已处理：历史链接与执行规范**。最终目标存在性检查未发现失效相对链接或绝对 Markdown 链接；核心材料可由根指南导航。复制的 AGENTS 已被明确界定为研究治理来源，不授权外部模型 SSH、修改服务器或重启训练。
4. **已处理：历史链接恢复格式**。二次恢复历史链接时，`00_project_context/AGENTS.md` 第7行和 `00_project_context/README.md` 文献行、继续研究段曾形成嵌套 Markdown 链接。主 agent 已从来源原文恢复后单次适配。独立审查先验证104个 Markdown，加入最后发布说明后主 agent 复跑同一 CommonMark+table 检查脚本，覆盖106个 Markdown、256个本地链接/图片目标（含19个图片目标），无嵌套普通链接、无渲染后失效或绝对本地目标；上述具体行另经人工复查。

## 已完成核验

- inventory 的 335 个源文件全部存在。09:04:58 安全检查的 stage 共 796 个文件、109,019,860 字节。最终 Markdown 归一为LF后，git diff 中原97文件仅8个 Markdown 有入口说明或链接变化；先前11个仅换行的变化已消除。无原始结果、receipt、CSV、JSON、代码变更和删除。详情见 `publication_tracked_markdown_final.json`。
- 最后规则扫描覆盖所有候选文本及4份 gzip 文本解压内容，没有发现私钥块、常见 GitHub/API token、带口令 URL、明文口令候选和排除文件类型。具体范围与局限见 `publication_safety_snapshot3.json`；这是有限规则扫描，不能证明不存在任何未知格式秘密。普通用户名、私网地址、服务器绝对路径及历史 digest 不属于命中条件。
- 无单文件超过20 MiB，无模型权重、私钥、环境凭据或pyc。六份 PDF 均为项目自产 diagnostic overview / registration 图，未混入已下载的第三方论文全文。
- 指南引用的五组历史 mAP50–95 比较已从原 JSON 独立复算，单位为百分点，SD 为 ddof=1，数值与指南一致。记录见 `publication_numbers_check.json`，可由 `review_publication_numbers.py` 在 CPU 上复核。
- Probe 三个数据集的共同对象四格计数相加正确，总样本 521 对。未把这些摘要当作 AP、学生可达上限或 KD 增益。
- 新指南清楚区分 Drone/LLVIP IR→RGB、VEDAI RGB→NIR，保留共享标签与配准边界、IR 独立训练标注辅助、teacher/reference 固定 seed42、学生随机性范围和 paired/weight0 尚不能完成跨模态归因的限制。
- 当前方法代码的 loss、双标签 loader、trainer、固定端点 evaluator 以及 resource guard、receipt/data-contract 依赖已纳入拟复制范围；固定环境和服务器路径限制已说明，未声称独立机器一键复现。
- 94 快照说明区分已完成 CSV 轮数和当前 progress 轮数，标明训练中非原子采集、stdout 截尾、manifest 无损压缩、OS-SSL pyc 恢复执行的源码身份缺口。

## 复核边界

全部核心内容已完成上述检查。最后新增的发布说明与审查报告应保持同一范围，不能在本报告之后追加未经检查的源目录再沿用本报告表示全部检查完成。最终 Markdown 检查见 `publication_markdown_final.json` 及 `review_markdown_navigation.py`；采用实际 Markdown token 解析核对导航，没有执行浏览器截图或外部 HTTP 链接检查。

本次仅审查证据包可阅读性、来源边界和发布内容，不重新认证历史训练全部契约，不进行新的模型评估，不宣称 Object Evidence v1 已经取得收益。
