# GitHub完整审计包（2026-09-06）

> 已发布并回读确认：新分支 `research/full-evidence-20260906`，共803文件、109,188,208字节（约104.1 MiB）；主分支未改。

## 目的
用户明确授权将此前分析和相关本地结果上传到GitHub，供其他模型独立审计。完整保留负结果、失败记录、早期版本、勘误与来源，不把运行中实验写成方法增益。

## 设置
在原main基础上新建独立clone与分支。原09_外部审计_rgbir和全部本地结果保持原位。新增research_bundle镜像相关资料，保留旧审计包；只对导出Markdown适配链接并补根级指南。94仅做只读补采，本次未启动、停止或修改GPU实验。

## 结果与结论
- [已发布分支](https://github.com/yudongfang-thu/rgbir/tree/research/full-evidence-20260906)；Git提交 `8fa61ba777820043d810ad8f3daad3d9cd6af979`。
- [模型阅读指南](https://github.com/yudongfang-thu/rgbir/blob/research/full-evidence-20260906/MODEL_REVIEW_GUIDE.md)；[可直接交给其他模型的审计提示词](https://github.com/yudongfang-thu/rgbir/blob/research/full-evidence-20260906/REVIEW_PROMPT.md)。
- 内容包括07研究分析、相关08实验日志、三数据集六baseline/521对诊断及完整图册、逐图/逐目标/特征数组、历史原始JSON/CSV、OEv1协议/源码/三seed记录、OS-SSL-IR记录、文献笔记、选定SAR历史证据。
- 94补采至2026-09-06 08:53:57 +08:00，包含真实运行源码/依赖、配置、压缩manifest、完整小型产物及明示截断的日志。尚无OEv1完整E200终态性能端点。
- 803个文件共约104.1 MiB，最大单文件11,389,987字节；原97文件仅8个Markdown内容改动，原数值/代码/receipt未改。
- 独立复算5组历史比较、核对521对摘要；106个Markdown、256个本地链接/图片目标解析通过；内容规则扫描未发现凭据候选。有限规则扫描不等于证明不存在任意未知格式秘密。
- Git推送成功，远端分支与本地提交一致；GitHub API回读README和REVIEW_PROMPT成功；本地clone干净。

## 产物路径
本目录：PUBLISH_RECEIPT.json、PACKAGE_FILES.json、github_readback_verification.json、PUBLICATION_REVIEW.md、publication_*检查产物、来源inventory与打包/补采/适配脚本。大清单的详细来源见仓库BUNDLE_MANIFEST和remote快照manifest。

本地发布clone：`C:/Users/MSI-PC/rgbir_review_worktrees/evidence-20260906`；94补采归档：`/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/github_full_evidence_20260906/`。

## 局限与下一步
未上传原始数据集、checkpoint、凭据、第三方论文全文或环境缓存。完整训练仍需数据、权重、对应环境与路径适配，本包不声称一键复现。日志为带时间戳快照，未来终态需要另行追加。git diff --check保留来源的空白/CRLF/Markdown硬换行提示；无冲突标记，未为格式检查改写原始证据。其他模型应按新分支的阅读指南独立复核，不能把旧强结论当作当前结论。
