# CFT / LLVIP 原文协议核定（2026-09-09）

**先做作者发布LLVIP checkpoint在完整official test上的复评：使用作者双流模型、原`test.test`/配对loader/指标，固定1024、CLI默认batch64、conf=.001、NMS IoU=.5、无增强；优先旧版Annotations。** 这是最快能回答“公开权重能否对齐论文表3”的实验。当前只完成协议审阅，未SSH、未启动资源或评价、未计算新hash。用户已授权原论文协议复评，本实验登记official test暴露即可，不以我方旧封存政策阻塞；新增我方配置迁移实验暂停。

## 对齐对象、证据优先级

[CFT论文arXiv v2](https://arxiv.org/html/2111.00273v2)表3/表1给LLVIP RGB+IR：AP50=97.5、AP75=72.9、AP50:95=63.6。正文明确YOLOv5双流、SGD lr=.01/momentum=.937/weight_decay=.0005和Mosaic，但没有明确列出LLVIP专用img/batch/epochs/评估NMS。README另报MR=5.40%，本次主目标为上述三个AP；完整归档未找到对应MR/FPPI评价脚本，不以另写MR程序冒充作者指标。

[作者仓库](https://github.com/DocF/multispectral-object-detection)README明确论文使用`transformerx3`配置，发布[LLVIP权重](https://drive.google.com/file/d/18yLDUOxNXQ17oypQ-fAV9OS9DESOZQtV/view?usp=sharing)。现有90权重路径`/mnt/dataX/ydf/projects/RGBT_campaign_90/external_reproductions/cft/author_llvip.pt`，下载回执413453231字节；原下载文件名`yolov5l_transformerx3_llvip_s1024_bs32_e200`，支持1024/B32/E200的训练意图，**文件名不能证明评估batch为32，也不能替代完整训练日志**。现有checkpoint接口回执epoch=-1，不足以恢复实际历史epoch。

固定源引用`fb591c9b163177c0e950db08e213e24ddc912d41`。本次读了已有`llvip_research/CFT_source`，并从`cft_forward90/cft_author_source.tar.gz`流式读完整`train.py`，未执行其代码。所有源位置相对上一级`2026-09-09_repro_RGBIR对比方法优先接入/`。

## 数据与标注

作者`data/multispectral/LLVIP.yaml`的`train_rgb/train_ir`是visible/infrared的`train.txt`，`val_rgb/val_ir`是各自`test.txt`；`nc=1,names=[person]`。公开发行版应为12025 train、3463 test，按原目录全量构建对应清单，不使用我方9619/2406 grouped split。

[LLVIP作者README](https://github.com/bupt-ai-cz/LLVIP)说明2021-09-01发行时已删低质图至30976张，2023-02-21又修正少量标注。CFT正文仍写16836对，故不能仅凭总数文本证明它实际用了更早全部图。当前ZIP顶层或单XML时间戳也不能认证标注版本。

旧版标注来源是[LLVIP官方previous annotations](https://github.com/bupt-ai-cz/LLVIP/blob/main/previous%20annotations.md)所链[Google Drive](https://drive.google.com/file/d/1RZqYKHXUVgSOi_eq15EDjfiR2D-tyFVV/view?usp=sharing)。直取候选端点：`https://drive.usercontent.google.com/download?id=1RZqYKHXUVgSOi_eq15EDjfiR2D-tyFVV&export=download&authuser=0&confirm=t`；本审阅确认链接身份，未读取该ZIP二进制。建议保存原ZIP、自带Annotations为`current_archive`，旧版独立目录，不覆盖原件。主复评使用旧标注；旧链接暂不可取时可以先准备图像和current_archive，若先评价则明确`CURRENT_ARCHIVE_LABELS`，不能直接判定原文不成立。

目前[LLVIP官方转换脚本](https://github.com/bupt-ai-cz/LLVIP/blob/main/toolbox/xml2txt_yolov5.py)使用`cx=(xmin+xmax)/(2*1280), cy=(ymin+ymax)/(2*1024), w=(xmax-xmin)/1280, h=(ymax-ymin)/1024`，class0，6位小数，无减1；它未过滤difficult、未裁边。CFT仅要求YOLO转换，没有公开私有转换副本或标签清单。因此用该官方转换可减少自创处理，但历史转换完全相同仍未证明。不得直接套用我方8位小数、退化框排除版然后叫原标签。

原loader会从路径`/images/`推导`/labels/`，需新建作者协议数据视图，两模态各有`images/test`和`labels/test`；列表路径不能resolve symlink后绕回无labels的raw目录。RGB/IR列表顺序、数量、stem和尺寸要核对；作者loader分别排序两流，没有强健的stem绑定断言。实际GT取`labels_rgb`，IR labels仍会被扫描缓存。缺标签、坏图、负值/归一化越界/重复标签须先报告，不静默删样本凑计数。

## 固定官方test入口

| 项目 | 采用值/实际公开行为 | 依据或差异 |
|---|---|---|
| 模型 | 作者现成`models.yolo_test.Model`，YOLOv5l两流，三处GPT，每处8块 | `models/transformer/yolov5l_fusion_transformerx3_llvip.yaml`；不能换YOLO11或单模态学生 |
| 权重加载 | 原`models.experimental.attempt_load`选择EMA或model、FP32、fuse、eval | 现有旧pickle兼容另见下文 |
| 图像尺寸 | `imgsz=1024`；保持比例，不裁图 | 权重文件名；不是论文正文显式超参 |
| eval batch | `64`；只在资源canary不容纳时预先改为可容纳batch并记录 | `test.py` CLI默认64；README未给LLVIP专用eval batch；函数默认32不能冒称论文配置 |
| loader | `create_dataloader_rgb_ir(..., pad=.5, rect=True)`，augment=False | `test.py`直接调用；`task=train/val/test`均仍读`val_rgb/val_ir`，故val路由只有official test |
| 预处理 | 两流cv2 BGR读取；先按长边1024缩放，缩小时INTER_AREA；按rect batch形状做letterbox，auto=False、scaleup=False、114填充；BGR→RGB、CHW，拼6通道，除255后拆两流 | `utils/datasets.py:load_image_rgb_ir/__getitem__`；不同于已有smoke直接letterbox；1024×1280图典型batch形状为864×1056，最终必须保存实测shape |
| 精度 | 原`half_precision=True`：CUDA用FP16，CPU用FP32 | 硬件决定的原公开分支；明确记录设备/精度；不据三图CPU时间承诺全量耗时 |
| 检测阈值 | conf=.001；NMS IoU=.5 | **CLI默认=.5，函数默认=.6**。固定公开CLI值，不能看AP后挑.5或.6 |
| NMS | 作者原torchvision NMS；multi_label=True但nc1时实际关闭；agnostic=False；max_det300/max_nms30000；10秒批内保护 | 出现NMS time limit警告不能把未处理图当完整有效复评 |
| 指标 | 原`test.py`匹配 + `utils.metrics.ap_per_class/compute_ap`；IoU=.50:.05:.95，取AP50/AP75/十阈值均值 | 置信度降序、一对一匹配、101点插值后trapezoid；保留原`>`比较和实现，不换COCOeval |
| 禁用项 | augment=False、save_hybrid=False、save_json=False、plots=False | hybrid会引入GT辅助预测；JSON分支硬编码COCO GT路径，不能用于LLVIP主指标 |
| 保存 | 原预测TXT+confidence、完整stdout/stderr、浮点结果tuple及每图名单/计数、compatibility、设备精度、来源/参数 | `save_txt=True, save_conf=True`可直接用作者功能；输出仅新实验目录 |

建议入口是极薄包装调用原`test.test(data, weights, batch_size=64, imgsz=1024, conf_thres=.001, iou_thres=.5, save_json=False, augment=False, save_hybrid=False, save_txt=True, save_conf=True, plots=False, opt=...)`。在其原`attempt_load`返回处串接已审阅的类兼容函数；保留原test的`model=None`分支与官方loader。不要把模型直接传为`model=`却不给dataloader，否则它按训练调用分支跳过建loader。调用函数可避开CLI `check_requirements()`自动依赖安装，同时保留明确传入的CLI参数语义。

这次评估只比较固定协议所得三个AP与97.5/72.9/63.6，保存各自百分点差值，按论文显示的一位小数报告是否一致。不因不一致增加阈值搜索、择优输入尺度或改标签；不一致结论限定为本次公开权重/源/标签版本/兼容环境。结果无论高低都保留。MR独立待作者对应评价实现，不能从AP声称MR对齐。

## 必须披露的兼容范围

已知作者checkpoint在当前公开源先报`TransformerBlock.conv`缺失。现有`cft_checkpoint_compat.py`将24个匹配旧CFT子模块的类名绑定到当前作者`myTransformerBlock`，保留参数/缓冲对象且不生成随机权重，三图fit已跑通。**这不是新增迁移到我方模型的实验，却仍是必要runtime兼容，不能称历史源码数值等价已证明。** 保留失败attempt1和兼容源码/回执。

Torch2.10需针对已知作者checkpoint允许旧模块pickle（现有`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`）；NumPy2.2缺`np.int`，公开loader使用它，可在薄包装作`np.int = int`兼容别名并记录。`get_hash`虽命名hash，函数实际仅`sum(os.path.getsize(...))`，不读内容、不计算digest；保留原实现即可符合no-new-hash。不安装新框架、不改metric/NMS算式来消除数字差异。

## 同协议重训的最低条件

完整`train.py`实际主入口调用`train_rgb_ir`。重训模型cfg须`models/transformer/yolov5l_fusion_transformerx3_llvip.yaml`、数据使用完整official train/test、通用初始化应为README发布的[作者yolov5l.pt](https://drive.google.com/file/d/12OFGLF73CqTgOCMJAycZ8lB4eW19D0nb/view?usp=sharing)，不是通用YOLO11、现有LLVIP已训权重或我方IR教师。保留源中的state-dict交集加载并记录实际迁入键，不能猜测两分支全来自预训练。

候选显式命令参数：`python train.py --weights <author_yolov5l.pt> --cfg models/transformer/yolov5l_fusion_transformerx3_llvip.yaml --data <official_LL VIP_yaml> --hyp data/hyp.scratch.yaml --epochs 200 --batch-size 32 --img-size 1024 1024 --project <new_runs> --name <new_attempt> --device <root_allocated>`。其中1024/B32/E200来自权重名；公开CLI默认其实640/B16/E100。必须在重训前注明这一推定来源，不能称完整原论文命令已找回。

源配方为SGD Nesterov，lr0=.01/lrf=.2 cosine，momentum=.937，weight_decay=.0005，nbs64且B32累积2，warmup3且至少1000 iterations；Mosaic1、mixup0、hsv(.015,.7,.4)、translate.1、scale.5、fliplr.5、其它几何默认0，rect=False/multi_scale=False/Adam=False；普通BN、EMA、autoanchor按源默认，workers8。原种子为`init_seeds(2+rank)`，single-process rank=-1得到1，DDP与之不同；不能把我方42或三seed方案加进去并叫原文命令。

重训还须先核原loss在现代Torch下的类型兼容、真实双流batch前反向和实际GPU峰值/总时长。**原best选择存在公开行为差异**：`test.test`返回`P,R,AP50,AP75,AP50:95,...`，而`fitness`只取前4列并按`.1/.9`加权，实际选`.1*AP50+.9*AP75`，注释却称AP50:95。为复现公开源不能静默改成AP50:95；若另做修正，必须在原源码实验之外另立身份。原训练默认每epoch在official test路由做验证/选best；暴露登记需体现这一作者行为。

目前最快优先级是完整official test checkpoint复评；同原文重训仍缺明确历史标签/转换证明、精确历史训练配置与初始化回执、现代环境真实训练canary及资源时长测量。不要把“配置可写”记作重训已开跑，也不凭空承诺10小时可完成。

## 暴露登记

此新目录的用途固定为`AUTHOR_PROTOCOL_REPRODUCTION`。official test的样本数、首次读取/指标产生时间、数据/标签版本、参数和结果应单独登记；它不继续充当我方方法开发中的未见测试。复评结果不得回流为我方新方法阈值/架构选择依据。父任务负责实际数据提取、资源租约、执行与实验索引，本审阅不更改它们。

## 数据准备包装静态核对

已读本目录`prepare_author_llvip.py`。中心不减1、固定1280×1024、6位小数、没有difficult过滤，符合当前LLVIP官方转换脚本的数值约定；两标签版本独立，primary previous，列表保留`images`视图路径而不resolve，有利于原loader正确找labels。

包装与官方脚本并非逐字相同：包装仅保留`name=='person'`并读float，当前官方脚本遍历所有object且读int。**当所有object名均person、坐标为整数字符串时，二者框转换数值等价**；应由已提取XML统计验证此条件并记入回执。不得新增difficult过滤。原官方脚本本身无退化框过滤；若遇到非正宽高或异常标注，应报告而不默默沿用我方过滤规则。

## 原评价包装静态核对

已读本目录初版`run_author_eval.py`，full分支以`model=None`调用原`test.test`，保存metric输入后仍委托原`ap_per_class`，保持原paired loader/rect/pad/匹配/AP算式；1024/.001/.5、无augment、FP16 CUDA分支符合本协议。没有使用先前三图smoke替代全量评价。

提交父任务的两项完成性检查：full也应断言实际双流各3463且stem顺序一致并保存实际所见名单；出现原NMS的10秒截止警告应标为不完整，不能仅因函数返回就写完整COMPLETED。初版只在canary断言`len(dataset)==3463`，没有捕捉NMS截止。`save_hybrid`默认False，建议在入口显式固定。这里只记录初版审阅，不预先声称父任务后续版本已处理；执行前最终版本及其回执由父任务留档。
