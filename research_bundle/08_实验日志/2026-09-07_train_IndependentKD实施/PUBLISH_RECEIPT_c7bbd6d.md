# 独立蒸馏补充证据发布回执

**2026-09-07：补充证据已推送至既有 GitHub 分支，并经远端读取核对。**

- 仓库：https://github.com/yudongfang-thu/rgbir
- 分支：`research/full-evidence-20260906`
- 本次提交：`c7bbd6da3abe7ba464b322964865ecfb2a6fd1a0`
- 前次已发布提交：`1218adfb6553ffd60928174d5290b590cee6fb8c`
- `git push`：成功；此前两次连接 443 超时，未修改远端，第三次成功。
- `git ls-remote origin refs/heads/research/full-evidence-20260906`：返回本次完整提交 ID，与本地 HEAD 一致。

本批补齐六个旧 N/C0 端点的完整 dev 补评、五个汇总指标逐项一致证据、已接受的观察评价桥接、实际矩形画布接口修复及其独立验收、三 seed 对象修复/损伤和背景误检诊断。发布不改写旧结果或失败 attempt，不包含凭据、大权重或数据集原图全集。

导出清单 `INDEPENDENT_KD_BUNDLE_MANIFEST_20260907_164107.json` 共 7423 个文件；[发布检查]（未导出的工作区路径：publication_review_20260907_164729.json）为 ACCEPTED，原始产物字节不一致、缺失文件、敏感信息候选、失效 Markdown 链接及超过 20 MiB 文件均为零。Git 内保留同一检查回执 `PUBLICATION_REVIEW_INDEPENDENT_20260907_164729.json`。派生 Markdown 的链接适配不改变原始代码、数值或图片。

截至 16:35 的阶段事实见 [阶段报告](STAGE_REPORT_1635.md)。为避免该报告短句产生歧义，明确：C1 与 C1_y 各完成至少 24 次成功更新；实际单卡双开验收对应 C1，未宣称 C1_y 自身已完成双开实测。正式三个 C1 seed 继续 E200，目前没有完整 C1 AP；L1 因几何证据不足未进入正式长训。

后续责任：显式接入并独立验收补评逐类指标；继续 C1 完整端点评价；C0 背景误检三 seed 同增触发专项复核，仅暂停其后续自动扩展；最终消融与四臂按冻结条件推进。每小时跟进已更新。整轮冲刺尚未完成。

本回执在远端核对成功后写入本地，用于记录已发布提交，不伪称其自身包含于该提交；后续阶段同步时纳入。
