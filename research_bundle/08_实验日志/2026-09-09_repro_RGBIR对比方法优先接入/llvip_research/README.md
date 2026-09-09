# LLVIP 作者实现可运行性核查（2026-09-09）

> **首先接入 CFT 作者 LLVIP checkpoint 做一对 fit 图像的原模型 CPU 前向/NMS机械验证。** 其权重下载端点已实际返回模型二进制。当前检索未找到同时具备 LLVIP、RGB-only 推理、完整作者训练代码与现成 LLVIP 权重的直接蒸馏实现；不得将融合检测结果记作 RGB-only KD 比较。

## 本次范围

先读本地 `01_文献/RGB-IR_20260905新增/{00_方法调研综述,文献精读笔记_20260905}.md` 和 `08_实验日志/2026-09-09_audit_方向证据与对比协议/README.md`。网页仅依赖作者论文、作者仓库及作者仓库直接给出的权重链接。

未使用 GPU、SSH、安装库或运行作者模型；未计算新 hash。下载了小体积作者源码。CFT checkpoint 只读响应头及128字节模型头，未在本子任务下载完整权重。

本地旧综述将 CCLKD 记为使用 LLVIP、称“94已复现”已经由9月9日audit纠正：论文数据为 OGSOD/VEDAI/Drone，项目旧实现为 LLD+CCL partial。因此不把它冒称 LLVIP 作者原协议已复现。MSDCRD 的本地总清单也标为分类KD，不能直接当检测实现。

## 五个候选与执行次序

| 候选 | 代码/权重核实 | 输入、检测器、LLVIP协议/指标 | 90接入成本与判断 |
|---|---|---|---|
| **CFT，首项**：Cross-Modality Fusion Transformer for Multispectral Object Detection | 官方完整仓库有训练、评价、双流模型、LLVIP YAML。作者LLVIP权重端点返回HTTP200、413453231字节、PyTorch ZIP头；名称`yolov5l_transformerx3_llvip_s1024_bs32_e200` | 推理RGB+IR双流YOLOv5l。官方YAML train指`train.txt`、val指`test.txt`。repo报AP50/AP75/mAP=97.5/72.9/63.6，MR=5.40%；原2021论文数据总数文本与现行15488对有版本差异，不能视作已核同标签版本 | **最短路径**：已训权重+少量依赖兼容，先CPU一对fit图像、无AP。训练完整E200时间未实测，不承诺10h。作者资料：[论文](https://arxiv.org/html/2111.00273v2)、[代码与权重发布](https://github.com/DocF/multispectral-object-detection) |
| **M2D-LIF**，第二候选：Rethinking Multi-modal Object Detection from the Perspective of Mono-Modality Feature Learning，ICCV2025 | 作者2025-11-29声明完整代码发布；有`train_dist.py`、`teacherTraining/`、Ultralytics fork、`data/LLVIP.yaml`和`val.py`；作者提供百度模型链接且README举LLVIP checkpoint。**本次未取得该盘内二进制，不能声称权重已下载验证** | M2D为单模态教师改善多模态学生学习；推理仍RGB+IR。YOLOv8路线；作者评价示例640、batch1、rect=True。LLVIP使用train/val目录，确切清单/标签版本和论文表数字本次未进一步认证 | 同有LLVIP/Drone支持，若作者权重可取可做独立融合参考；完整训练要两个教师与融合学生，成本高于CFT现成权重前向。不是RGB-only KD。[作者论文入口、实现、标签和权重](https://github.com/Zhao-Tian-yi/M2D-LIF) |
| **AMFD**：Distillation via Adaptive Multimodal Fusion for Multispectral Pedestrian Detection，TMM2025 | 官方有`config/LLVIP/Student_Fasterrcnn_r18_fpn_1x_llvip_amfd.py`及`Teacher_r50_fpn_1x_llvip_thermal_first.py`。公开模型说明仅明确给KAIST教师/学生checkpoint；未确认LLVIP checkpoint | **single-stream不是single-modality**：论文III-A明确学生输入为RGB+TIR的图像级融合。双流R50教师→单流R18 Faster R-CNN；另有RetinaNet/DINO。LLVIP official train12025/test3463，COCO AP学生55.6→58.3(+2.7) | MMDetection3.1/MMCV2.0.1/MMEngine0.8.4依赖，教师权重和LLVIP COCO标注要补；不适合作最快首项，更不应误记为RGB-only蒸馏。[作者论文](https://arxiv.org/html/2405.12944v2)、[作者实现](https://github.com/bigD233/AMFD) |
| **ICAFusion**：Iterative Cross-Attention Guided Feature Fusion for Multispectral Object Detection，PR2023 | 有完整YOLOv5代码及`train.py/test.py/detect_twostream.py`。README当前发布权重明确列KAIST、FLIR；未发现LLVIP权重链接 | 推理RGB+IR；README追加LLVIP AP50/AP75/mAP=98.4/76.2/64.5，但未在该条绑定清单/权重版本。作者说明代码持续优化使结果变化 | 可适配重训，但当前没有优于CFT的即取LLVIP权重；不是首项。具体LLVIPsplit不可仅凭名称追认为我方grouped_v1。[作者仓库](https://github.com/chanchanchan97/ICAFusion)、[作者论文](https://arxiv.org/abs/2308.07504) |
| **CrossFusionKD**：A Cross-Modal Knowledge Distillation Framework for UxV Object Detection | 作者repo实际仅`.gitignore/README.md`，说明发布后开放代码，另附demo；**没有训练/评价代码或checkpoint** | 作者预印本描述融合教师→RGB-only学生并涉及LLVIP；检测器、split及指标实现无法从仓库审核 | 任务方向最贴近，却目前不可执行；不能为了排进对比而凭论文自行补实现并叫作者复现。[作者预印本](https://d197for5662m48.cloudfront.net/documents/publicationstatus/291181/preprint_pdf/b528a6abe3984a80807b3711de5c846a.pdf)、[作者仓库](https://github.com/Murtazaabidi1/CrossFusionKD-Framework-for-UxV-Object-Detection) |

## CFT 的具体交接

- 官方仓库：`https://github.com/DocF/multispectral-object-detection`。
- 本次API `commits/main` 返回固定引用：`fb591c9b163177c0e950db08e213e24ddc912d41`。这是读取作者Git对象引用，没有本地计算hash。
- [作者Drive页面](https://drive.google.com/file/d/18yLDUOxNXQ17oypQ-fAV9OS9DESOZQtV/view?usp=sharing)。
- [实际二进制下载端点](https://drive.usercontent.google.com/download?id=18yLDUOxNXQ17oypQ-fAV9OS9DESOZQtV&export=download&authuser=0&confirm=t)：HTTP200、`application/octet-stream`、Content-Length413453231；仅检查128字节，其中含`PK`及`archive/data.pkl`。这证明端点开始返回模型流，不证明完整下载或checkpoint可加载。
- 本地原源码：`CFT_source/`。78个选定代码/文本文件已存54个；20个核心依赖文件齐全，所有已存`.py`均通过AST语法解析。API额度随后耗尽，24个文件未取；**不把本地子集称作完整仓库**。完整列表及缺项见`CFT_SOURCE_ARCHIVE_MANIFEST.json`。父任务另已报告90完整官方clone成功，正式执行应使用该完整clone。

最小前向链路（静态核对，尚未运行）：

1. 读取**现行fit名单内**的`020001`，visible/infrared同stem各一图；从90 `data_attempt1/processed/llvip/yolo/grouped_v1/{visible,infrared}/images/fit/020001.jpg`读入。不要选dev前缀01/04/07/12/25，也不要打开官方test。
2. `cv2.IMREAD_COLOR`读取两模态、原`utils.datasets.letterbox`做相同几何变换，BGR转RGB、CHW contiguous、float32/255。checkpoint名称对应1024；仅做机械前向时若改尺寸，应明确记录。
3. `models.experimental.attempt_load(weights, map_location='cpu')`；单checkpoint返回一个原模型。`models/yolo_test.py:214`签名为`forward(self, x, x2, augment=False, profile=False)`；模型调用为`out, train_out = model(rgb_tensor, ir_tensor, augment=False)`。
4. 用原`utils.general.non_max_suppression`检查有限输出、类别、框shape。CPU线程1，FP32/eval，不跑AP，不写检测标签回数据目录。
5. PyTorch新版默认`weights_only`与旧YOLO pickle可能不兼容，针对已确认作者来源的checkpoint按需明确`weights_only=False`或等价兼容包装；应记录兼容更改，不允许以随机构建模型冒充权重恢复。

**不要直接用作者`test.py --task train`：源码即使task=train也读取`data['val_rgb']`和`data['val_ir']`；作者LLVIP YAML的val就是official test。** `test.py`还会调用依赖检查和完整评价。本次建议直接前向包装，不调用其评价主入口。

CPU最小导入依赖覆盖：`models/{common,experimental,yolo_test,yolo}.py`、`utils/{datasets,general,torch_utils,google_utils,metrics,plots,autoanchor,activations}.py`及`global_var.py`，包`__init__.py`；本地均已保存。`models/common.py`导入pandas/requests/PIL及datasets/plots，故仅装torch仍可能导入失败。

## 结果解释边界

官方LLVIP train包含我方fit+dev。作者若以完整official train训练，其checkpoint已见过我方dev，**即使只在我方dev推理，也不能当未见数据泛化或与我方N/C比较**。当前机械前向用fit图足够；原论文test数字复现需要另行统一数据版本和封存test使用协议，不因已有checkpoint而自动开test。

CFT、ICAFusion、M2D-LIF、AMFD都是RGB+IR部署参考，其AP不进入RGB-only KD胜负表。RGB-only主线的通用KD接入由父任务另行处理，须保留PROTOCOL-ADAPTED身份。

## 本地产物

- `CFT_tree.json`、`CFT_commit.json`：官方API返回，保留作者引用。
- `CFT_source/`、`CFT_SOURCE_ARCHIVE_MANIFEST.json`：54个原源码/文本及78项完整性清单；无媒体/权重。
- `CFT_core_blob_downloads.json`、`CFT_remaining_blob_downloads.json`：成功与限流记录。
- `CFT_WEIGHT_ENDPOINT_CHECK.json`：本次模型端点读取回执。
- `source_download_receipt.json`：此前raw下载成功/失败记录；部分原始尝试遭网络reset。
- `CrossFusionKD/README.md`、`ICAFusion/README.md`、`M2D-LIF/train_dist.py`和`CFT/{test.py,requirements.txt}`：前一轮小文件原始副本。

本条不包含新的实验AP，也不宣称任何作者方法已完成复现。
