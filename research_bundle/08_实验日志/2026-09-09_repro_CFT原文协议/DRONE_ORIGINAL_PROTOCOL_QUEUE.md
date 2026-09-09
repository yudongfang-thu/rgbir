# DroneVehicle 原论文协议复现队列（2026-09-09）

结论：**M²D-LIF 的作者融合 checkpoint 重评估排第一；C2Former 暂列资料待补，不先启动重训。**两者目前都没有形成“我方已取得并验证 Drone final checkpoint → 原数据 → 原评估器”的完整执行闭环。M²D-LIF 已具备完整作者源包、明确融合权重发布 URL、原 OBB 评估入口和 split 说明，缺口更少；C2Former 未核验到 Drone 最终权重，且 matched 标注处理仍不闭合。

本轮遵循最新指令，优先原模型与原论文协议，暂停新增迁移版本。此子任务仅读本地、作者仓库、公开论文和公开链接元信息；未 SSH、未安装、未运行模型、未训练、未计算新 hash。本文件给出接续队列，不将资料审计记成已启动实验。

## 一、只排原模型闭环

| 顺位 | 权重复评 | 原模型与指标 | 当前可执行程度 |
|---|---|---|---|
| 1：M²D-LIF | 作者 README 发布融合 checkpoint 网盘；入口指定 `checkpoint/multimodal/DroneVehicle.pt`。第三方读者在作者 issue #16 报告使用过这一文件，但我方尚未取得。 | RGB+IR、6通道、双分支 YOLOv8/C2f + LIF、OBB；作者 parser 默认 **m**。Drone 论文模型约 **37.1M**，不能拿 FLIR 的36.53M作其精确参数数。原指标由作者 OBBValidator 计算 mAP50、mAP50:95，匹配实现使用 `batch_probiou`。 | 完整源包已落盘；无需教师就能做融合权重复评。先取得作者文件、核对作者标签/split，然后原入口复评。仍缺文件和数据一致性证据。 |
| 2：C2Former | README 只有 KAIST Results 链接；仓库 `resnet50-2stream.pth` 是配置中的骨干预训练入口，不能当作 Drone final checkpoint。所查 releases/issues 没有对应最终权重下载地址。 | RGB+IR、Two_Stream_S2ANet + C2Former 双 ResNet50 + FPN、5类 OBB le135。原数据类使用旋转 IoU .5 的 mAP；本轮没有实测总参数数。 | 原 train/test 源码和 config 齐全，但权重复评缺 checkpoint；重训还缺明确 matched 标签处理/清单。不能因“24 epochs”就认定比 M²D 权重复评更快。 |

两者都不是 RGB-only student。即使都输出 OBB，其匹配指标实现仍不同，分别用各自作者评估器；不接现有 YOLO11/HBB 分析器，不直接混列分数。[M²D 原模型构建](https://raw.githubusercontent.com/Zhao-Tian-yi/M2D-LIF/master/ultralytics/nn/tasks.py)、[M²D OBB 指标](https://raw.githubusercontent.com/Zhao-Tian-yi/M2D-LIF/master/ultralytics/models/yolo/obb/val.py)、[C2 原模型配置](https://raw.githubusercontent.com/yuanmaoxun/C2Former/main/configs/s2anet/s2anet_c2former_fpn_1x_dota_le135.py)、[C2 原数据类](https://raw.githubusercontent.com/yuanmaoxun/C2Former/main/mmrotate/datasets/dronevehicle.py)。

## 二、已核 URL 与实际取得状态

| URL / 资产 | 本轮核验事实 | 不足 |
|---|---|---|
| [M²D 作者库](https://github.com/Zhao-Tian-yi/M2D-LIF) / [README](https://raw.githubusercontent.com/Zhao-Tian-yi/M2D-LIF/master/readme.md) | 作者发布完整实现；此前下载完整 ZIP 4,638,044 B、1,251文件，包内无 .pt/.pth。 | 源包不是 trained weights。 |
| [M²D 权重分享](https://pan.baidu.com/s/1GKDkfhJrKeskrnDNRzmFXw?pwd=vmvr) | 作者 README 给出此 URL；浏览器调用前的 HTTP HEAD 显示302→200提取码 HTML页面。 | **未取得文件列表、二进制直链、checkpoint文件或模型身份。随后浏览器站点安全策略明确拒绝该域访问，已停止进一步访问且未绕过。** |
| [M²D 作者标签分享](https://pan.baidu.com/s/1NBDTPIgdlvnARiO_Uy8c-w?pwd=ir9b) | 作者 README 给出此 URL；此前 HEAD 为302→200提取码页面。 | 未取得标签文件，也未验证90现有标签与作者发布版一致。 |
| [M²D issue #16](https://github.com/Zhao-Tian-yi/M2D-LIF/issues/16) | 2026-08-13第三方读者称发布内容包含 FLIR.pt、LLVIP.pt、DroneVehicle.pt 三个融合权重，单模态教师未包含；公开API显示无评论。 | 是读者使用报告，不是作者确认、我方下载回执或可替代的实验结果；不采用其AP作为本次复现数字。 |
| [M²D releases](https://github.com/Zhao-Tian-yi/M2D-LIF/releases) / [HF发布请求 issue #4](https://github.com/Zhao-Tian-yi/M2D-LIF/issues/4) | 无GitHub releases；HF issue 仅有发布邀请、无作者回复/模型地址。 | 没有核验到可替代的作者 GitHub/HF 二进制直链。 |
| [C2 作者库](https://github.com/yuanmaoxun/C2Former) / [骨干目录](https://github.com/yuanmaoxun/C2Former/tree/main/pretrain_weights) | 主配置 `pretrained='pretrain_weights/resnet50-2stream.pth'`。 | 本轮该 raw 文件 HEAD 超时，不等于文件不存在；即便能下载，也不是配置训练后的完整检测器。 |
| [C2 KAIST Results](https://drive.google.com/drive/folders/1a8Swf5BSgSyE6S6XdGuKhoygV_AyrDQ4?usp=sharing) | 作者 README 明确标为 KAIST Results；HEAD 返回200 HTML页面。 | 没有核验文件内容，不能改称 Drone 权重链接。 |
| [C2 releases](https://github.com/yuanmaoxun/C2Former/releases) / [weights issue #9](https://github.com/yuanmaoxun/C2Former/issues/9) | 无release；权重询问没有回复链接。已读公开 issue 列表及相关评论。 | 在本轮有范围的审计内，仍无可验证的 Drone final checkpoint URL。 |

“页面200”和“作者README列链接”均不等于“checkpoint可直接下载”。本轮**没有新增已验证的二进制checkpoint直链**，因此不能把二者写成已经具备完整原协议可执行闭环。

## 三、M²D：先融合权重复评，明确不盲跑重训入口

作者评估入口是 [val_obb.py](https://raw.githubusercontent.com/Zhao-Tian-yi/M2D-LIF/master/val_obb.py)：`python val_obb.py`。仅绑定既有作者 checkpoint、数据绝对路径和调度分配的设备，保留作者模型/预处理/指标。

原文目标已经重新核对 **CVF正式发表PDF**（10页、3,120,270 B，仅内存读取，未生成hash）：Table 4 的 Drone test 为 **mAP50=81.4、mAP50:95=68.1、37.1M参数**。同表 val mAP50=84.5，正文却写85.2，存在论文内部不一致，应分别留档，不自行选更高数。issue #16读者所报82.6/66.9不是该正式表的完全复现；只能作为权重曾被使用的线索。取得公开checkpoint后，应如实报告其与正式表差值，不调整协议追数。[CVF正式论文](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhao_Rethinking_Multi-modal_Object_Detection_from_the_Perspective_of_Mono-Modality_Feature_ICCV_2025_paper.pdf)

- 原 split：train 17,990 / val 1,469 / test 8,980；测试目录为 `DroneVehicle_test/images/val`，IR 为同名 `images_ir/val`，标签为 `labels/val`。作者用 `val` 字段读取正式 test 目录，不能把目录名当协议身份。新原协议复现实验应明确写入原 test 身份，不混入此前已治理 dev 的方法筛选链。[作者数据配置](https://raw.githubusercontent.com/Zhao-Tian-yi/M2D-LIF/master/data/DroneVehicle.yaml)
- 类别按作者 id 顺序 `car, truck, feright_car, bus, van`；保留 OBB 多边形标签，不从现有 HBB 反构造。原图对齐、边框裁剪和标签版本需与作者发布数据核对。
- 发布 eval 的输入为640、B1、rect=True。**完整配置链实际 conf=.25、NMS iou=.7、max_det=300、half=False**：作者 `default.yaml` 已显式设置 .25，BaseValidator 的 `None→.001` 分支不会触发。不能套用 stock Ultralytics 的默认行为推断作者阈值。[作者默认配置](https://raw.githubusercontent.com/Zhao-Tian-yi/M2D-LIF/master/ultralytics/cfg/default.yaml)
- 先读取得的 checkpoint：names/nc、OBB head、6通道、实际参数数和 train_args；不得用通用 YOLO11n 或随机模型代替。模型规模依据以实际 checkpoint 为最终可执行证据；论文中 FLIR36.53M和Drone37.1M要分开记录。[作者论文](https://arxiv.org/html/2503.11780v2)
- 原 fork 使用 torch2.1.0/torchvision0.16.0/Ultralytics8.1.44；当前90的torch2.10+stock8.4.115导入通过不代表作者模型已匹配。必要运行兼容修正应独立记录，不改模型与协议。
- 当前“无新hash”约束与原 loader 的 `get_hash` 标签缓存调用仍有冲突。它不是方法参数，亦不能靠 `cache=False` 自动关闭；主任务在原协议接入时须明确记录处理办法。本子任务没有新增适配代码或把此问题算作已解决。

权重复评仅需要已训练的融合模型，**不需要 RGB/IR teacher**。相比之下，重训依赖两位教师；[train_dist_obb.py](https://raw.githubusercontent.com/Zhao-Tian-yi/M2D-LIF/master/train_dist_obb.py) 两个教师槽都加载 `dv_ir.pt`，不能原样启动。发布入口100ep/B8/AMP=false/lr0=.001，还与论文SGD/lr0=.01描述存在差别。未获得正确双教师和对应训练配置前，重训不进入首队列；这不妨碍取得fusion checkpoint后单独重评估。

## 四、C2：原评估命令可明确，但还不能闭环

作者 README 的 `--out` 只保存预测。原 `tools/test.py` 只有提供 `--eval` 才调用 `dataset.evaluate`，所以完整原指标入口应为：

```text
python tools/test.py configs/s2anet/s2anet_c2former_fpn_1x_dota_le135.py ACTUAL_DRONE_FINAL_CHECKPOINT.pth --out RUN_DIR/results.pkl --eval mAP
```

这是作者CLI已有能力，不是另写评估器。[原 test.py](https://raw.githubusercontent.com/yuanmaoxun/C2Former/main/tools/test.py)

必须绑定 matched 数据版：同目录 `id.jpg`/`id_tir.jpg`，标签读取 `*_tir.txt`，每行为8坐标+类别+difficulty。原配置路径包含 `train_total/val_total/test_total` 与 `MatchedLabelTxtMVP`；这些路径没有证明成员与原始 Drone split/作者最终表完全一致。实际载入的 test 数据数和标注变换需落盘核对。

重训配置是双ResNet50/S2ANet，B1、workers2、resize(512,640)、SGD lr .001/m .9/wd1e-4，继承 `schedule_2x.py` 的24epochs（16/22衰减）；文件名的1x不能覆盖实际继承结果。依赖老mmcv/mmdet算子，不能直接用现有stock YOLO环境替代。

作者 issue [#14](https://github.com/yuanmaoxun/C2Former/issues/14) 和 [#19](https://github.com/yuanmaoxun/C2Former/issues/19) 的第三方讨论涉及 anchor 设置、白边裁剪及标签处理，并没有可供本次采用的作者最终标注/最终权重闭环；不能用读者自行改标签后的数字认定原协议已复现，也不据此修改标签追分。

## 五、交接资产与下一项

已有完整 M²D ZIP：[M2D-LIF_master_author.zip](E:/SHARE/光sar/08_实验日志/2026-09-09_repro_RGBIR对比方法优先接入/drone_research/M2D-LIF_master_author.zip)。原审计、HTTP源文件回执和选择性作者源码在同目录；不重复制作迁移包。

下一项是**取得作者已训练的 Drone fusion checkpoint 文件及作者标签证据**，随后按上述原模型/原split/原评估器做独立复现。因本站安全策略阻断作者网盘访问，本子任务无法继续获取该文件；可由用户手动提供已取得的作者文件。C2 保留备用资料位，当前不自动消耗训练预算。
