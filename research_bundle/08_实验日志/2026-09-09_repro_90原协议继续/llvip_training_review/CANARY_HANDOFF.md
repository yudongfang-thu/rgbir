# 24次原更新 canary 交接

文件 `train_author_canary.py` 与 `loss_index_compat.py` 必须放在同一目录。它调用原 `train.train(...)`，不复制训练循环。本地只通过 AST/CLI help；实际 CPU/GPU 回执由90主执行方产生。

示意调用（将权重与新输出路径填实；不直接运行下面的占位参数）：

```bash
python -B train_author_canary.py preflight --weights COCO_YOLOV5L.pt --reference-weights AUTHOR_LLVIP_VISIBLE.pt --modality visible --output NEW_CPU_ATTEMPT
python -B train_author_canary.py canary --weights COCO_YOLOV5L.pt --reference-weights AUTHOR_LLVIP_VISIBLE.pt --modality visible --output NEW_GPU_ATTEMPT --successful-updates 24 --max-batches 256 --memory-fraction .68
```

GPU命令必须在现有单卡lease中使用既有cft90环境执行。CPU模式隐藏CUDA；两种模式都先在CPU检查：generic COCO80、epoch=-1/optimizerNone、与作者发布checkpoint的四项模型架构字段，以及向作者目标模型的参数键交集。架构不符或交集小于95%先停并保存记录，不能因同名 yolov5l.pt 就视为历史同一初始化；作者checkpoint仅用于核架构，不将其权重载入训练。每次输出目录必须不存在。

原B8/1280/E200不变，论文hyp四值显式覆盖，patience0显式登记。相同整数边界问题已静态核到本仓库utils/loss.py；助手模块从这个原函数重建两处整数边界并执行CPU等价检查，不复用CFT训练函数。数据使用已有独立baseline的officialtrain12025/test3463与previous标签。

原SGD的post-step hook计实际更新；原GradScaler.step前后判断是否跳过，记录scale与每批计数。通过原Callbacks在batch-end停止，此时该批scaler.update、zero_grad及EMA均已执行。逐批记录实际输入、数据等待、训练耗时和显存，首批确认原autocast/FP16卷积。24更新默认上限256批且不越过第1个epoch，不计算AP、不保存可继续训练的checkpoint。

原logger首批会额外JIT trace模型，此包装显式跳过batch图绘制/TensorBoard graph trace，避免额外FP32前向占用与计时干扰；W&B关闭。保留原train、优化器、增强、EMA和原数据加载；不调用main的requirements/git检查，不自动下载权重，不执行dataset MD5。

`post_startup_timing` 排除首4批并给出训练与数据等待均值/中位数。它仍处于原3epoch warmup，梯度累积尚未稳定为8；字段不称作稳态优化过程。E200外推仅为训练部分，未计每epoch官方test评价和保存，不能独自证明≤10小时。若未达到24更新，保存failure而不是通过。
