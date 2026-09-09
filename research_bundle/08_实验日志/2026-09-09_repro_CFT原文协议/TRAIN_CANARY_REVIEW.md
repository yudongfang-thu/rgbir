# 原 CFT 两次成功更新技术 canary

**已准备可供父任务部署的薄包装，尚未运行训练。** [train_canary_wrapper.py](train_canary_wrapper.py)直接调用完整作者源的`train.train_rgb_ir(hyp,opt,torch.device('cuda:0'),tb_writer=None)`；没有复制训练循环、替换损失或改成我方模型。[静态检查](TRAIN_CANARY_STATIC_CHECKS.json)确认AST可解析，作者函数引用31个opt属性，包装提供38项，无缺项。本代理仅运行标准库AST检查，未导入模型、SSH或使用GPU。

父任务将文件放到90新实验产物目录，待作者通用COCO预训练权重下载完成后，通过**已有统一资源lease**执行：

```bash
<existing_cft_python> -B train_canary_wrapper.py --weights <downloaded_author_generic_yolov5l.pt> --output <new_canary_attempt_directory> --memory-fraction 0.65
```

包装要求现有`require_bound_lease_from_environment()`通过且只暴露一张卡，自己不创建lease、不选物理卡、不启动远程任务。allocator cap默认总卡容量65%，父任务可按已核资源预算显式设置并记录；B32不自动调整。真实峰值取Torch记录及已有guard记录。OOM仅说明原B32在该租约/allocator预算下未通过，不等同数学上任何24GB卡均不可能训练。

## 原参数与数据流

固定模型`models/transformer/yolov5l_fusion_transformerx3_llvip.yaml`，原`data/hyp.scratch.yaml`，1024/1024、B32、E200、SGD、原warmup/累积/EMA/autoanchor及增强。其配置证据等级沿用[PROTOCOL_REVIEW.md](PROTOCOL_REVIEW.md)：1024/B32/E200来自作者权重名，而非完整历史训练命令。

初始化只能使用README的[作者通用yolov5l.pt](https://drive.google.com/file/d/12OFGLF73CqTgOCMJAycZ8lB4eW19D0nb/view?usp=sharing)，包装要求实际model有80类、epoch=-1、optimizer=None，避免把已训LLVIP模型误作预训练或触发作者隐式resume。公开`state_dict`交集迁入行为原样保留，实际迁入键与数量写`initialization_transfer.json`；它不会假定两分支都完整继承通用预训练。

使用`data_author_protocol/llvip_attempt1/previous/LLVIP.yaml`，完整official train12025。原函数还会构建official test3463的loader并扫描标签；本次新授权允许此作者协议访问，并记录test暴露。不会用我方fit/dev名单替代。包装验证两模态列表个数、对应stem、图像存在，以及原loader实际扫描后的数量/顺序。

原入口所需`global_var._init()`和`flag_visual_training_dataset=False`已设置。所有模型构建、算式、优化器参数、AMP与数据增强仍由原作者函数执行。`noautoanchor=False`，不为了加速改锚框政策。

## 停止和记录

注册原SGD实例的`register_step_post_hook`，在原`SGD.step`完成后计数，因此GradScaler因非有限梯度跳过更新时不计成功。成功两次即抛专用`CanaryComplete`，退出原循环并写`CANARY_TWO_SUCCESSFUL_OPTIMIZER_STEPS`回执。第二次更新后的`scaler.update/zero_grad/EMA`不会继续执行；**这只是技术canary，不保存或复用其训练状态**。第一步的后续原逻辑正常执行。

最多读取16个训练batch；仍无两次成功更新则记技术失败。原epoch-end评价额外设不应到达断言，防止逻辑偏离时误跑AP。保留原`epochs=200/notest=False/nosave=False`参数，只靠边界钩子在两步停，不修改学习率日程成两步训练。

保存`plan.json`、事件日志、初始化身份与迁入键、每batch输入形状/base stem、每成功step的学习率/时间/显存，以及成功`receipt.json`或`failure.json`。base stem不是Mosaic全部供图名单；此处不将其冒称完整augmentation exposure ledger。所有输出为新目录，不覆盖原作者源码或既有实验。

## 依赖和明确适配

使用既有独立CFT环境。作者`train.py`导入需要torch/torchvision、numpy、cv2、PyYAML、tqdm、pandas、seaborn、matplotlib、PIL、scipy、requests、**tensorboard**；可选thop缺失时作者代码自行降级。当前环境是否具备tensorboard尚未由本代理runtime验证；缺依赖会写TECHNICAL_FAILURE，不自动安装。

W&B强制走作者可选`wandb=None`分支，`tb_writer=None`，跳过标签分布绘图；这些只影响日志，不改训练公式。保留NumPy旧`int/float`别名及明确允许已知作者pickle。直接导入函数避开作者CLI `check_requirements`和W&B resume逻辑；初始化权重存在性预检查让原`attempt_download`不触发下载。数据YAML禁止download字段。

本次新建模型使用当前作者`myTransformerBlock`类，不应用已训LLVIP pickle的24块旧类名修复。若原loss在Torch2.10出现类型兼容错误，包装如实记录失败，不能自动切到我方loss或任意类型修补后隐瞒差异。

## 对重训预算的解释

前两次成功更新处于原warmup，B32开始时累积通常为1，之后才趋向2。因此两step的吞吐包含初始化/预热、数据缓存和可能AMP跳步，**只能初判原batch前反向和内存是否可行，不能确认E200≤10h**。完整原配方还需稳定batch吞吐、每epoch验证时间和数据加载开销测量。原B32失败时保留资源/技术失败，不自动换小batch并宣称原配置成功。
