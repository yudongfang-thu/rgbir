# 90原文复现继续与94可用性检查（2026-09-09）

**90已完成两项新作者权重复评：LLVIP RGB/IR 的 mAP50–95=52.664239/67.014908%，与修订版论文52.7/67.0接近；各3463图/7931GT，完整进程90.70/88.06秒，44项数值/覆盖复核通过。94通过SSH、8卡枚举、动态GPU1的16次前反向和数据解码检查，旧训练未恢复。CFT三卡训练首批OOM、0更新；LLVIP单模态原训练10次成功更新后OOM，未满足24次准入目标。均未启动E200。**

## LLVIP作者单模态原模型：两次完整复评已完成

使用作者发布的两个93,884,963字节checkpoint（各46,631,350参数）、原作者val.py，official train12025/test3463中的test、previous标注7931GT。输入1280、eval B32、FP32、conf .001、NMS .6，无TTA，原匹配与AP函数；与CFT的1024/FP16/不同匹配器分列。

| 模态 | 指标 | 本次 % | 修订版v2/v4 % | 本次−修订版 pp | 冻结的初版v1参照 % |
|---|---|---:|---:|---:|---:|
| RGB | AP50 | 90.788183 | 90.8 | −0.011817 | 90.8 |
| RGB | AP75 | 56.342893 | 56.4 | −0.057107 | 51.9 |
| RGB | AP50–95 | 52.664239 | 52.7 | −0.035761 | 50.0 |
| IR | AP50 | 96.378278 | 96.5 | −0.121722 | 94.6 |
| IR | AP75 | 76.420674 | 76.4 | +0.020674 | 72.2 |
| IR | AP50–95 | 67.014908 | 67.0 | +0.014908 | 61.9 |

最初资产交接冻结了[v1论文参照](https://arxiv.org/html/2108.10831v1#S5.T3)，其16836对/70:30为早期数据描述。随后的原训练文献核对发现[v2（2021-10-17）Table3](https://arxiv.org/html/2108.10831v2#S5.T3)已经改为15488对与上表修订值，本地v4相同；作者仓库也说明在更新数据上重训。保留原v1参照和冻结JSON，不以事后改参数追分；增加这一可核查的论文版本对照，不把数字差解释为方法提升。

每个模态先在原B32/1280/FP32完成2批64图的无AP资源canary，随后完整评价。两次完整运行NVML进程峰均8908MiB，进程树RSS峰12138/12173MiB，使用同一动态GPU3顺序执行。两个完整回执及原预测/指标数组保存在[执行证据](llvip_execution/)，汇总见[完成回执](llvip_execution/llvip_author_pair_completed.json)。

[数值与覆盖复核](LLVIP_EVALUATION_REVIEW.md)完成：44项通过，原作者AP函数在保存的TP/confidence/class/GT数组上独立CPU复算，三项指标及全部10个IoU列与实际回执的差均为0。RGB无预测的2张图、IR无预测的1张图仍计入完整3463图和7931GT分母。复核没有重做box/GT匹配和模型前向；复核者参与过初始wrapper准备，不称盲审或跨模型审阅。总身份保留PAPER-RECONSTRUCTED及WARN边界，不能把数值复算通过升级成历史训练全过程精确复现。

身份为PAPER-RECONSTRUCTED作者公开权重复评，尚不是从头训练、多seed或新KD收益。现代Torch旧pickle、NumPy别名和无摘要缓存键适配均明确记录；原图、标签、作者代码与CFT数据视图未修改。实际执行以各attempt中的executed_wrapper.py为准：后续准备版对device选择的改动未部署到这些已运行attempt。执行版本在CUDA初始化后调用select_device('0')，实际lease/NVML均为GPU3；保留这一环境变量副作用，不据模板反写执行历史。

同原模型/同评价条件下，两个作者单模态权重的IR−RGB为14.350669 mAP pp，说明这组发布模型存在较大性能差距；**不能把差距当作已经可转移的KD增益**。CFT与这两个模型的输入尺寸、指标实现不同，不能直接用63.54与67.01宣称融合方法不如IR。

## CFT三卡原训练：技术问题进一步定位，仍未准入

新attempt实测每卡11/11/10，AMP均开启，首卷积输出FP16；排除了“完整B32被每卡复制”和“AMP没生效”的解释。仍在首批forward申请44MiB时被.68 allocator额度拒绝，0次成功更新、23.24秒结束。NVML轮询会漏短峰，保留各卡Torch保留峰16560/16516/16562MiB和原错误。

原loss中两处整数索引clamp的浮点边界在Torch2.10确实报错，已用Python整数grid边界作局部兼容，CPU空集合/边界目标返回的全部tcls/tbox/indices/anchors与整型gain参考精确相等；本次GPU仍未到loss，不能称loss实际训练已通过。见[诊断与未执行候选](CFT_TRAIN_MEMORY_DIAGNOSIS.md)及[原失败记录](cft_dp3_execution/)。不再重复同存储方式的尝试，不提高资源上限。

## LLVIP单模态原训练：10次更新后失败，不启动E200

沿用作者train函数，YOLOv5l从核对过结构的COCO80通用权重初始化；B8、1280、E200、SGD、lr0=.0032/lrf=.12/momentum=.843/wd=.00036。其余按固定作者源码补全；禁用当前源码额外early stopping以测完整E200配方，未改变标签、split、增强或batch。初始化release的历史同一性未证明，见[配方核定与缺口](llvip_training_review/README.md)。

| 步骤 | 实际结果 | 耗时 |
|---|---|---:|
| CPU预检 | COCO初始化结构/键转移及旧loss边界兼容检查通过；CUDA未初始化 | 4.93秒 |
| train attempt1 | wrapper的Path未像原CLI一样转为str，YAML序列化失败；0批、0更新、0显存分配 | 5.60秒 |
| train attempt2 | 修复Path序列化后，10次SGD更新、10次scaler尝试、0 AMP skip；第11批backward申请100MiB时OOM | 39.16秒 |

attempt2使用动态GPU6，首卷积确为FP16，输入B8×3×1280×1280；NVML峰16838MiB，Torch分配/保留峰16282.46/16570MiB，进程树RSS峰20793MiB。失败时物理显存仍有6.34GiB空闲，但进程已触及.68 allocator预算；这不证明完整24GiB也无法运行。原[失败回执](llvip_training_execution/llvip_train_canary_attempt2/failure.json)、[事件](llvip_training_execution/llvip_train_canary_attempt2/events.jsonl)、[实际代码与日志](llvip_training_execution/)均保留。

排除前4批后仅6个有效批平均0.960665秒，机械外推E200纯训练80.27小时。**这不是稳定吞吐或完整训练实测**：处在early warmup，GPU另有约1.03GiB活动任务，未计epoch验证/保存/启动。它不足以批准用户要求的10小时内完整实验，因此未排E200；也不能与此前YOLO11n/640的3.47小时直接比较模型效率。

批末allocated约697MiB稳定，reserved逐批增长，OOM发生在backward；这些证据不能单独证明计算图泄漏、碎片或cuDNN搜索中的任一解释。CPU复核原单卡RANK=-1、seed=0已经设置benchmark=False/deterministic=True；只关闭benchmark的重跑将没有实质变化，未运行。后续须先有明确实现原因或可核验的存储/执行改动，再做一次有区别的资源测量；不改batch/图像尺度冒充原协议，也不重复开启相同失败路径。

## 目的与执行边界

接续CFT作者权重复评后的原训练可行性，另接入资产闭合的原作者模型。先排查原CFT双卡训练OOM的实现和资源原因，不据技术失败判断论文方法。90新产物仍只写 `/mnt/dataX/ydf/projects/RGBT_campaign_90`；94本轮先检查，不根据SSH恢复就自动重启旧训练。

所有GPU调用通过各自服务器现有全局lease，沿用每卡项目显存低于70%、整卡留2GiB、通常最多3张项目GPU等工程约束。用户“继续”不是取消资源纪律。原始数据、checkpoint、失败attempt与回执保留。

## 工作线

- 主执行：服务器状态、实际CUDA检查、统一GPU调度与执行。
- CFT源码诊断：原作者DP/DDP、AMP、总B32/1024训练显存及可运行路径；先CPU审核。
- 其他原模型：优先公开作者checkpoint可取得的LLVIP/Drone方法；不重复访问已被站点策略拒绝的下载域，不用随机模型或我方配置替代。

## 产物路径

本地为本目录；90实际执行目录为 `/mnt/dataX/ydf/projects/RGBT_campaign_90/artifacts/original_repro_continue_20260909_attempt1/`，阶段文档及复核副本存其 `stage_evidence/`。作者资产只记服务器路径，不上传权重或整套原图。94检查产物另列具体路径，不覆盖旧实验状态。

## 当前结论

截至首次检查12:19:50，94 uptime 1:21、7张空卡、数据盘19T可用；90当前7张GPU可枚举，本项目lease为空。此快照不等于长时间稳定性证明，不等于旧任务已恢复。

13:06再次只读核对：94仍可连接、8卡可枚举、数据盘19T可用，其他用户已占用全部8卡，不能继续沿用首检“7张空卡”安排任务；各卡剩余约13.1–19.6GiB。90本阶段tmux任务均已退出，全局lease为空；94本项目旧训练仍无进程或会话。原始主机进程清单只保存在本地，不将其他用户路径发布到GitHub；公开状态提要见 `SERVER_STATUS_FINAL.json`。

94实际CUDA检查使用旧项目 `SpaceNet6_OTD_official_reproduction/tools/project_resource_guard.py` 与原全局lease。重启前残留的2条lease先保存副本，再由guard正常检查进程并清理；未另建资源池。16次256×256矩阵前反向、梯度有限且结果正确，原LLVIP训练图解码1280×1024成功。Torch峰分配17.25MiB、保留22MiB；仅测试动态分配的一张GPU，不代表8张卡逐卡稳定性验收。原始[完成回执](health94_cuda/cuda_receipt.json)及[日志](health94_cuda/cuda_health.log)保留，服务器 `/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/availability94_20260909_attempt1/`。

本轮原论文test只用于作者模型/协议复评，不进入本项目KD方法、阈值或超参数选择。新增迁移版继续暂停。当前已经闭合的是作者权重与公开评价数字；尚未闭合的是从COCO初始化重训的资源路径、历史完整配方、三seed和KD归因。AMFD未找到公开LLVIP模型权重，M2D原Drone资产仍受既有下载阻塞影响，未将KAIST权重或替代模型算作这两个数据集的复现。
