# Independent KD 阶段发布回执

**2026-09-07 已成功推送到既有 GitHub 分支；本回执记录阶段实现与启动证据，不代表 C1 完成 E200 或本轮方法已获支持。**

- 仓库：[yudongfang-thu/rgbir](https://github.com/yudongfang-thu/rgbir/tree/research/full-evidence-20260906)。
- 分支：`research/full-evidence-20260906`。
- 提交：[1218adfb6553ffd60928174d5290b590cee6fb8c](https://github.com/yudongfang-thu/rgbir/commit/1218adfb6553ffd60928174d5290b590cee6fb8c)。
- `git push` 实际退出码为0，远端确认 `7bfaf2f..1218adf research/full-evidence-20260906 -> research/full-evidence-20260906`；本地 HEAD 为上述完整提交且工作树干净。
- 随后的两次独立 `ls-remote` 读回因 HTTPS 连接超时失败，因此不声称另一次远端读回已验证；这不改写已经返回成功的 push 回执。
- 本次提交3857个新增或修改文件；最新导出清单包含5216个文件（含既有未改文件）。
- [发布检查](publication_review_20260907_155755.json)为 ACCEPTED：0凭据候选、0非Markdown原始字节差异、0缺失文件、0失效Markdown链接、0超过20MiB文件。Markdown仅适配导航及换行；不上传权重或数据集原图全集。
- 内容包括独立模块、冻结计划/用户规格包、实际六轨迹兼容、64批校准、canary及双开资源证据、正式启动、旧九端点、24对有限几何审核、失败attempt与复核入口。
- Git空白检查报告来自保留的历史源副本、原始文本/日志和Markdown硬换行；不为消除这类提示修改原始证据。

后续六个旧N/C0逐类/逐对象补评代码仍在独立审阅，不属于本提交已经运行的结果。其结果及新的C1阶段证据按后续批次同步。

后续补核：第三次 `ls-remote` 已成功，远端分支完整提交与本地逐字一致，见[独立读回回执](PUBLISH_REMOTE_VERIFICATION_1218adf.json)。先前两次网络超时仍保留为历史记录。
