# LLVIP 作者 YOLOv5l 原训练配方核定

**核心配方足够明确，可继续做单卡原训练资源/成本检查；完整历史训练无法称为 AUTHOR-EXACT。必须用单模态 YOLOv5l、B8、1280、E200，以及论文 SGD 参数，不能直接运行仓库默认配置。此次未 SSH、未 GPU、未写 trainer、未计算新 hash。**

## 论文与代码的差别

| 项目 | 论文/作者明确设置 | 当前原仓库的默认或缺口 |
|---|---|---|
| 模型/输入 | COCO 预训练 YOLOv5l，visible 与 infrared 分别训练 | CLI 默认 yolov5s；须显式传已核定的 COCO80 yolov5l.pt，不能从已在 LLVIP 训练的两个权重再微调冒充原初始化 |
| epoch / batch | 200 / 8 | CLI 默认 300 / 16 |
| 图像尺度 | README train 命令 `--img 1280`；论文 §5.2 未单列检测 resize 数值 | CLI 默认 640；1280 来自作者 README，非猜测或借用 CFT |
| optimizer | SGD；初始 lr 0.0032、末端 0.000384；momentum 0.843、weight decay 0.00036 | 原 hyp.scratch 为 lr0=.01、lrf=.1、momentum=.937、wd=.0005，与论文不同 |
| split | v2 起 15488 对，77.6%/22.4%；README train 作训练、test 作 validation | 当前 official 12025/3463 与该比例一致；采用已冻结 previous 标签视图，不重分 split |

证据：[论文 v1 §5.2](https://arxiv.org/html/2108.10831v1#S5.SS2)、[论文 v2 §5.2](https://arxiv.org/html/2108.10831v2#S5.SS2)、[作者 README 训练命令](https://github.com/bupt-ai-cz/LLVIP#train-2)、[固定源码 hyp.scratch.yaml](https://github.com/bupt-ai-cz/LLVIP/blob/c1a655cce437ebfd990a97b04fc48fbb99f4c47b/yolov5/data/hyps/hyp.scratch.yaml)。本地 `01_文献/RGB-IR_20260905新增/LLVIP__2021_Dataset__Visible-infrared_Paired_Dataset_Low-light_Vision__2021_arXiv_2108.10831.pdf` 实为 v4，首页标 2023-06-14；训练设置在 PDF 页6–7，也给出相同 SGD 数值。不要按文件名中的 2021 将其当 v1。

## 最小原入口建议

使用 90 完整 clone 的 `yolov5/train.py:train(hyp,opt,device,callbacks)`；保留原 Model、loader、ComputeLoss、SGD、AMP 和 callbacks，不新建训练循环。依据原 `parse_opt` 补完整 opt，显式覆盖 `weights/data/imgsz=1280/batch_size=8/epochs=200/save_dir`；单可见 GPU，rank=-1，原 train 图读完整 train12025。hyp 复制固定源码默认值并仅覆盖 `lr0=.0032,lrf=.12,momentum=.843,weight_decay=.00036`；`.12` 是论文终始学习率之比。

其余实现须标为**源码补全**：SGD Nesterov、余弦调度、3 epoch warmup、原增强、EMA；B8 的梯度累积在 warmup 后为8、对应 nominal batch64，不能擅自取消。原 CLI patience=100，可能少于200轮结束；若目标是论文所述完整 E200，建议运行前显式冻结 `patience=0` 并登记“禁用当前代码额外早停”，不称其为 README 原样命令。[固定作者 train.py](https://github.com/bupt-ai-cz/LLVIP/blob/c1a655cce437ebfd990a97b04fc48fbb99f4c47b/yolov5/train.py)

先做真实2次 SGD 更新的资源检查；通过后再测较长一段成本，不凭两步承诺整程。12025/B8 对应1504批/epoch、E200约300800批，10小时的平均总预算约0.1197秒/批，且还须容纳每轮官方test验证与保存。这是预算反推，未测吞吐。单模态较CFT小只支持优先检查，不保证当前显存额度下通过。

执行兼容仅沿用已登记的离线措施：关闭可选 wandb 与自动安装/下载，旧 pickle/NumPy 别名明确留痕，沿用本数据视图的无摘要 cache key；先在 CPU 检查该仓库自己的 build_targets，不能假定 CFT 补丁可直接共用。训练不会改变原标签版本或正在进行的作者权重评估阈值。

## 尚未闭合的历史条件

作者 README 的 yolov5l.pt 链接只指向 Ultralytics releases 列表，未指定 release；原下载函数会请求 latest，失败后回退 v6.0。论文引用 YOLOv5 v3.0 也不能证明训练所用二进制就是该版。须以本地 CPU 身份/模型 YAML/COCO80类别/参数键交集选定具体兼容初始化并记录来源；当前 CFT 的同名 generic 权重不能仅凭名称当作相同资产。[作者下载实现](https://github.com/bupt-ai-cz/LLVIP/blob/c1a655cce437ebfd990a97b04fc48fbb99f4c47b/yolov5/utils/downloads.py)

论文未给出完整 evolved hyp、精确训练源码/运行环境、seed、预训练 release 和作者发布权重对应的 last/best 选择回执。当前源码可把未写明参数补为明确的重建选择，足以做技术资源检查，但还不足以声称精确恢复历史训练。

## 论文版本旁证（不修改已冻结评估目标）

| 参考版本 | visible AP50/AP75/mAP % | infrared AP50/AP75/mAP % | 数据描述 |
|---|---|---|---|
| v1 / 早期会议版参考 | 90.8 / 51.9 / 50.0 | 94.6 / 72.2 / 61.9 | 16836对、70/30 |
| v2（2021-10-17）；本地v4同值 | 90.8 / 56.4 / 52.7 | 96.5 / 76.4 / 67.0 | 15488对、77.6/22.4 |

版本差异由作者原文核查取得，应另列对照，不在看到当前评估结果后替换已冻结的目标。[v1 Table3](https://arxiv.org/html/2108.10831v1#S5.T3)、[v2 Table3](https://arxiv.org/html/2108.10831v2#S5.T3)

本审阅为已有复用 subagent，非 fresh 或跨模型独立审阅；并行复核因并发槽位已满未启动。结构化值见 `protocol.json`。共享 README/index 由主执行任务统一更新。
