# C1 协调器 Python 入口与余量差异复核

**代码范围接受，无阻塞问题。16/16 CPU fixture 独立复跑通过；没有启动 prepare/screen/SSH/GPU。**

在此前完整控制流审阅基础上，只复核作者后续的窄改动：训练使用真实当前 C1 canary 峰值向上取整 +2048 MiB，评价仍为实测峰值 +256 MiB，RSS 仍为向上取整 +4096 MiB 且至少8192 MiB。manifest 分别记录两种余量及 native TAL 目标密度波动依据，不用历史 C0 峰值替换当前测量，原 profile/全局资源绑定保持。

`initialize` 构造 job argv 和 manifest 的两处 Python 路径使用 `absolute()`，保持用户指定 venv/bin/python 的符号链接入口，不把它 resolve 成 base interpreter。新增 fixture 对这两处实际输出做模拟符号链接验证。本审阅不将该 CPU 模拟称为真实 Linux 启动证据；根/作者保存的真实环境核查属于另份执行证据。

源码副本和 `cpu_tests.log` 与本文件同目录。原 v1/v2 和根的 v3 审阅、失败 campaign 均保留，本次没有修改 NEW 或作者源码。此回执只接受所存代码范围，不自行授权或证明正式训练完成。
