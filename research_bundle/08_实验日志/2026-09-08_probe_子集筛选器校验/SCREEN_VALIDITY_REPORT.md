# 子集 E8 筛选器有效性对照

**本次固定 2048 图、seed42、独立 E8 未保留已知 C0−N 的正向：mAP 差为 −1.128105 pp；此前全量训练集独立 E8 为 +0.544042 pp。能确定的是这两个已执行窗口出现排序反转，不能分解归因于 BN、训练图像构成、重复曝光、优化步数或学习率时程。**

本报告只读取已存在小回执、已接受分析器输出、配置和 CSV，不重算 AP、不运行模型、不访问权重、不新增统计检验或 SD。

## 本轮实际端点与五类原值

单位均为百分制 AP，差值为百分点 pp；原始 fraction 保存在[已接受读出](analysis/output_attempt1/summary.json)。本轮原生完整 dev 为 1469 图、22462 GT、五类，固定第 8 轮 last/EMA，无 best 选择。

|臂|mAP50–95 (%)|AP50 (%)|AP75 (%)|
|---|---:|---:|---:|
|N|27.588046|43.153525|31.457494|
|C0|26.459942|41.735800|30.036811|
|C0−N (pp)|-1.128105|-1.417726|-1.420683|

|类别|N mAP|C0 mAP|ΔmAP (pp)|N AP50|C0 AP50|N AP75|C0 AP75|
|---|---:|---:|---:|---:|---:|---:|---:|
|car|56.833528|55.055964|-1.777564|88.102453|86.959696|64.773908|62.231919|
|freight car|10.260852|9.355198|-0.905654|17.314459|15.760150|10.899132|9.820973|
|truck|21.187857|19.388008|-1.799849|32.568268|29.621421|24.577823|22.122455|
|bus|43.010989|42.408233|-0.602756|66.291642|65.980963|50.305471|49.509003|
|van|6.647004|6.092305|-0.554700|11.490806|10.356768|6.731136|6.499703|

五类的 mAP、AP50、AP75 在 C0 均低于 N。这不是跨 seed 稳定性或蒸馏内容有害的证据。

来源：[N 独立评价](evidence_final/evaluations/N/short_evaluation_receipt.json)、[C0 独立评价](evidence_final/evaluations/C0/short_evaluation_receipt.json)。其 checkpoint stat 与各自训练完成回执相同，原值与本轮已接受分析器的 fraction 逐项一致。

## 训练量与身份，不把 batch 当 optimizer update

|窗口/臂|训练集不同图像数|每轮 batch|总训练 batch|成功 optimizer 更新|attempt / AMP skip|训练样本曝光数|
|---|---:|---:|---:|---:|---:|---:|
|本子集 E8 / N|2048|64|512|298|304 / 6|16384|
|本子集 E8 / C0|2048|64|512|297|304 / 7|16384|
|原全量 E8 / N|17990|563|4504|2667|2674 / 7|143920|
|原全量 E8 / C0|17990|563|4504|2667|2674 / 7|143920|
|原全量 E8 / C1|17990|563|4504|2667|2674 / 7|143920|

曝光数按已接受训练人口与完整轮数计算（2048×8、17990×8），是原图/配对样本进入训练的次数，不是新增独立图像，也不是增强后像素完全相同的次数；不含可丢弃的 fresh canary。原全量每轮最后 batch 为 6 图，不能用 4504×32 的填满容量冒充实际人口曝光。当前本地前缀只记录 30 批，不声称本报告逐图重放了全部曝光。

两种 E8 均为相同通用 yolo11n.pt 初始化路径、seed42、B32/nbs64/workers4、SGD、AMP、正常 BN、lr0=0.01/lrf=0.01、线性独立 8 轮日程、warmup=3 轮，N/C0 λ=0/0.1。三个 warmup epoch 分别对应 192 与 1689 个 batch；相同 epoch 名称不等于相同更新数量或按 batch 的 LR 路径。两种 E8 都不是 E200 的前 8 轮。

本轮两臂初始化回执核对的是重建为五类后的学生全部 499 个参数/buffer 张量，包括检测头；并非声称该状态与原 80 类通用 checkpoint 全键相等。两臂完整初始化、初始 model/T/R stat 相同，前 30 批 RGB/IR 全像素与双标签逐元素相等；此证据范围止于该前缀。两臂正常 BN 的 243 个运行 buffer 改变。训练期额外一次 AMP skip 导致 C0 比 N 少一个成功更新，不能单凭此把 −1.128105 pp 归因于该差别。

本轮来源：[N 配置](evidence_final/runs/N/short_screen_config.yaml)、[C0 配置](evidence_final/runs/C0/short_screen_config.yaml)、[N 训练](evidence_final/runs/N/short_training_receipt.json)、[C0 训练](evidence_final/runs/C0/short_training_receipt.json)、[N 初始化](evidence_final/runs/N/initialization_check.json)、[C0 初始化](evidence_final/runs/C0/initialization_check.json)、[N flow](evidence_final/runs/N/flow_check.json)、[C0 flow](evidence_final/runs/C0/flow_check.json)、各自 runtime_ready.json。

原全量来源：[已接受三臂汇总](../2026-09-08_train_分类快速反馈E8/three_arm_summary_20260908/summary.json)、[实际配置的冻结源](../2026-09-08_train_分类快速反馈E8/launch_evidence_attempt1/source/configs/drone_N_s42_E8.yaml)、三个 endpoint 目录中的 runs/{arm}/short_training_receipt.json；人口与完整末端已由各 endpoint_review_receipt.json 接受。

## 原生 loss 轨迹：只用结构完整的行

|臂|首个有效 epoch|box / cls / DFL|最后有效完整 epoch|box / cls / DFL|
|---|---:|---|---:|---|
|N|1|1.55497 / 2.7887 / 1.35365|7|1.29236 / 1.02004 / 1.19033|
|C0|1|1.55966 / 2.80023 / 1.35664|7|1.29902 / 1.03499 / 1.17157|

来源：[N results.csv](evidence_final/runs/N/results.csv)、[C0 results.csv](evidence_final/runs/C0/results.csv)。第 8 轮只有 8 个字段，而表头有 15 个；禁 val 导致末行布局缺损，本报告整行排除，不把错位的 LR 当成指标，也不取该行推断最终 loss。训练完成第 8 轮由独立训练回执确认，AP 仅来自独立完整 dev 评价；第 1→7 轮原生 loss 下降不能证明收敛、解释 AP 排序或证明过拟合，且这些三项不是加权 KD 总损失。

## 旧全量 E8 与成熟 FT3 的边界

|原全量 E8 臂|mAP50–95 (%)|AP50 (%)|AP75 (%)|
|---|---:|---:|---:|
|N|43.376542|64.983495|50.301986|
|C0|43.920584|65.071726|51.113261|
|C1|43.408107|64.712517|50.145839|

原全量 C0−N 为 +0.544042 pp，C1−N 为 +0.031565 pp，均直接复用原已接受汇总，不与新子集结果混算。C1 使用另一损失及原固定 λ=0.09227393550836771，本轮没有 C1，不可推其在新窗口的排序。

近期成熟 checkpoint + BN 冻结 FT3 是另一协议：例如前条目 LLVIP L3 从 visible42 成熟 last 初始化、2048 图、3×64=192 批、lr=1e−4 恒定、warmup0、BN 运行统计冻结，完整 dev 为 2406 图/7879 GT。该协议的成熟初始化/数据集/BN/LR/日程与这里通用初始化的 Drone E8 均不同；不能把其微调漂移与新子集 E8 的低 AP 合并成单一因果解释。来源：[实际预冻结 L3 协议](../2026-09-08_probe_对象坐标分布目标/SHORT_SCREEN_PROTOCOL.md)。

本队列执行 571.548243 秒，ADMISSION 等待 10.687456 秒，总 wall 582.235702 秒；执行含两次 canary、两次训练、两次独立评价及交接。训练入口 N/C0 为 224.976281/214.619190 秒，评价入口为 18.347238/19.024567 秒，不再与队列时间重复求和。来源：[完整队列回执](evidence_final/queue/completion.json)。

本次可结束的是“该固定子集 E8 能保留已知 N/C0 正方向”这一筛选器校验；不能据此排除 C0、分类蒸馏、定位或 DFL 方向，也不能推断数据重复导致过拟合。数据构成、累计曝光、按 batch 的 warmup/LR、成功更新及运行细节没有被分别控制；这里没有足够证据选择其中某项作为排序反转原因。没有新统计检验、SD、一般排序校准或 E200 自动扩展结论。
