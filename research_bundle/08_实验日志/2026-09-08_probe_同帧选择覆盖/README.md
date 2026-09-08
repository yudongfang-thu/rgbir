# LLVIP 同帧错误与实际选择门覆盖

**状态：两次同批有界推理均已完成，零训练更新。旧 assigned 定义的教师正确/学生错误为 9 个，原生 NMS 后为 1 个且已选中；32 个 selected 中 31 个双方检出。旧低置信候选不能直接解释为真实漏检。详见 [最终报告](FINAL_REPORT.md) 与 [原生四定义读出](witness_analysis_final/README.md)。**

首批结果：[最终完整回执读出](analysis_final/README.md) · [真实对象独立复核](REAL_READOUT_REVIEW.md) · [候选定义差异及具体替代框](INTERPRETATION.md)。80对象全部配对，79进base，78通过teacher正确候选门，64 eligible，32 selected；其中旧诊断both-correct为19个，不能当梯度份额。

首probe入口38.802秒，含排队/初始化/清理的队列166.045秒，原guard已COMPLETED。数据流与历史首批记录逐字段一致，参数及buffers无变化、0次优化/EMA更新；并未保存历史像素张量作逐像素比较。CPU映射8项、选择导出6项在94固定环境全部通过，聚合器及实际对象计数已独立复核。

后续[原生检测见证固定计划](WITNESS_FOLLOWUP_PLAN.md)现已完成：80 行旧对象和选择逐项复现，新增的原生匹配读出与结论见[最终报告](FINAL_REPORT.md)。原始首批读出与全部回执保留。

## 固定问题与范围

使用已完成LLVIP短训的相同初始化S、IR教师T与冻结RGB参考R（本数据集S初始化与R为同一checkpoint），复现既有首个32图批次的顺序、双标签及增强。S沿用训练模式但冻结BN统计，T/R为eval，AMP设置不变。只执行一个真实批次的S/T/R无梯度前向，不执行原生损失、反向、optimizer step、EMA update、训练或AP评价。

复用原trainer初始化以保留数据随机流；它会创建空optimizer与EMA对象，检查它们始终零更新。没有调用训练循环。新增映射元数据不得改变图片、标签、增强、RNG或顺序；实际首批必须与已完成run的sample_stream逐字段相同。每个原始RGB/IR GT通过稳定ID映射到增强后行，不能用近邻框匹配代替已知行传递。

在同一次前向内导出全RGB GT、配对、有效区域、R候选、T条件、质量、eligible、selected和matched/base索引。用既有错误状态规则区分教师正确而学生错误的无粗候选、低置信、定位等桶，并列学生正确而教师错误的风险桶；每步报告自己的分母。

这批来自train且S为成熟初始化，不能直接连接旧200图dev的修复比例，不能估计总体AP或宣称避免负迁移。原C阈值、选择规则和损失均不改。不打开新的长训矩阵。

## 验证与执行

先通过已知ID/翻转/缩放/裁剪小样例和聚合真值，独立复核原selector索引映射；再部署独立不可变release。唯一GPU推理本身作为有界资源短测，经原global lease、原guard动态选卡，保留实测峰值；不另建资源池，不改旧任务。

代码、小回执与逐对象表存本目录；94输出位于 `/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_selection_coverage_20260908/`。不新建文件hash、不下载checkpoint、不用test。
