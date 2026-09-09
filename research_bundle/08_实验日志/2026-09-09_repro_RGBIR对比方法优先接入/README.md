# RGBIR 相关工作与实际复现接入（2026-09-09）

**90 上已完成：CFT 作者 checkpoint 经显式兼容适配跑通 3 对 LLVIP fit 图；BCKD 的 BCDL 分类算子 6/6 检查通过；CMDistill-corrected 入口与原 22 项测试通过。尚无新增训练 AP，也未完成论文整表复现。优先把 CMDistill 接到相同 RGB-only 学生协议，作者融合模型复现另列。**

## 本轮执行结果

| 路线 | 已执行 | 证据范围 | 尚缺 |
|---|---|---|---|
| CFT / LLVIP | 完整作者源与 413,453,231 字节公开权重；真实前向 3/3 完成 | checkpoint 恢复执行，显式 PROTOCOL-ADAPTED | 论文 AP、原环境等价、重训 |
| BCKD / BCDL | 作者分类算子对独立表达式，90 CPU 6/6 PASS | 小张量 loss/学生梯度、教师隔离；classification partial | 完整 head、定位、reduction、数据集训练 |
| CMDistill-corrected | 独立 90 来源目录，CLI help、原 22 tests PASS | 既有适配代码接线与单元测试 | 真实 IR 教师、实际数据短测、完整训练 |
| M²D-LIF / Drone、LLVIP | 作者源包 4.64 MB、1,251 文件，本地与 90 留存 | 代码资产与训练脚本核查 | 作者教师权重、运行、协议适配 |

这些层级不能合称“已复现四种方法”。没有以随机模型替代作者权重，也不以三图前向或单测裁决方法效果。

## CFT：实际恢复过程与限制

来源为 [DocF/multispectral-object-detection](https://github.com/DocF/multispectral-object-detection) 和 [Cross-Modality Fusion Transformer](https://arxiv.org/abs/2111.00273)，不是另一篇 Calibrated Complementary Transformer。源 revision 为 `fb591c9b163177c0e950db08e213e24ddc912d41`，采用作者提供的 [LLVIP checkpoint](https://drive.google.com/file/d/18yLDUOxNXQ17oypQ-fAV9OS9DESOZQtV/view)。

第一次真实前向失败：pickle 中融合 block 名为 TransformerBlock，但子模块是 ln_input/ln_output/sa/mlp；当前同名类是另一种 conv/linear/tr 结构。两个公开历史版本也已把融合类称为 myTransformerBlock，**不能声称找到了 checkpoint 原训练代码，或证明只是历史重命名**。

第二次由 [兼容代码](cft_checkpoint_compat.py)对结构严格匹配的 24 个融合 block 重绑到当前作者 myTransformerBlock；不调用构造器，断言原参数和 buffer 对象不变。作者源与权重未改，失败 attempt 保留。[独立兼容审阅](drone_research/CFT_COMPAT_REVIEW.md)未发现结构问题，但不证明与未公开训练源码数值等价。

[运行入口](run_cft_author_cpu_v2.py)：独立 cft90 环境，torch 2.10.0+cu128、CPU 4 线程、FP32，旧 pickle 兼容设置在回执中明确。参数量 206,198,710，不以 CPU 小样本时间外推 GPU 完整训练。

| 预先固定的 fit stem | 每模态输入尺寸 | 原始输出 | NMS 框数 | 前向+NMS 秒 |
|---|---|---|---:|---:|
| 020001 | 1×3×832×1024 | 1×52416×6 | 2 | 1.601 |
| 100118 | 同上 | 同上 | 1 | 1.513 |
| 180408 | 同上 | 同上 | 2 | 1.490 |

输出均有限，含加载和保存共 9.768 秒，峰值 RSS 2,853.47 MiB，CUDA 未初始化。框数不是准确率。

- [完成回执](cft_forward90/cft_forward_attempt2/receipt.json)，同目录有逐图 JSON 和原始 NPZ。
- [失败回执](cft_forward90/cft_forward_attempt1/failure.json)、[权重来源](cft_forward90/weight_downloaded.json)、[源码来源](cft_forward90/source_downloaded.json)、[历史 diff](cft_forward90/cft_source_history.txt)。
- 完整源快照 `cft_forward90/cft_author_source.tar.gz` 为 41.60 MB；checkpoint 仅留服务器。
- 批量复制带回的 `cft_forward90/AUTHOR_ARCHIVE_RECEIPT.json` 实际属于 **M²D-LIF**，原内容保留，不作为 CFT 来源证据。

CFT 部署同时使用 RGB+IR。其作者训练人口包含本项目 grouped dev，因此只访问三个 fit 图，不计算该 checkpoint 的 dev 泛化 AP，不读取封存 test。原论文表复现与本项目 fit 重训是不同协议，本轮都未完成。

## 接下来优先复现谁

**同一 RGB-only 学生、数据清单和训练预算优先；作者原协议单列。**

| 优先级 | 方法 | 决策 |
|---|---|---|
| 1 | CMDistill-corrected | IR→RGB 任务一致、已有工程，先绑定真实 IR 教师；本项目 YOLO11n 为 PROTOCOL-ADAPTED，非作者 YOLOv5s 精确复现 |
| 2 | BCKD | 对应分类置信度和定位问题，作者分类算子已实际核验；补完整实现后才称 BCKD |
| 后续 | FGD、LD | 特征/定位代表对比；旧 FGD-like 缺全局项、旧 LD-style 非完整 VLR，不能改名冒充完整复现 |
| 独立作者线 | CFT、M²D-LIF、C2Former | CFT 权重已执行；后两项有作者工程，但双输入、Drone OBB 和不同骨干必须单列 |

[Drone 审阅](drone_research/README.md)涵盖 CMDistill、CCLKD、CMKD-Net、M²D-LIF、C2Former，另记红外特权 VQ 工作；[LLVIP 审阅](llvip_research/README.md)涵盖 CFT、AMFD、M²D-LIF、ICAFusion、CrossFusionKD。“未核到资产”不表示全网不存在。

重要纠正：

- [AMFD](https://github.com/bigD233/AMFD)的 single-stream 学生仍用 RGB+IR，不能当 RGB-only。
- [M²D-LIF](https://github.com/Zhao-Tian-yi/M2D-LIF)公开 train_dist_obb.py 的 RGB/IR 两教师都指向 dv_ir.pt；需核对真实教师并记录修订。Drone 类别顺序、OBB、实际 scale=m 也与本项目不同。
- CMD 历史相对其旧 native 为 −0.348781 pp；CCLKD partial 为 +0.343753 pp、方向不一致。这是旧证据，不能当 90 新结果。旧 workers8 与后期 N/C0 workers4 的样本随机流也不能直接混比，详见[资产审计](existing_assets/README.md)。

## CMD 与 BCKD 可执行入口

CMD 已在服务器 `external_reproductions/cmdistill_corrected/source_attempt1` 接通，原工程未改。[CPU 回执](existing_assets/cmd90_port/cmd_cpu_receipt_attempt1.json)和[测试输出](existing_assets/cmd90_port/cmd_tests_attempt1.log)：22 passed in 1.68s。双数据集模板已指向 90 prepared 数据，真实 IR 教师仍为 TO_BIND；[说明与命令](existing_assets/cmd90_port/README.md)。

`weights/pretrained/yolo11n.pt` 是通用初始化，不能充当 IR 教师。教师绑定后先测真实 batch 的非零 KD、教师冻结、显存和端到端吞吐，再完整训练。新 90 对比应使用同配方新 N；≤10h 仍需硬件实测，本轮未给出新时长保证，不用分钟 AP 代替整程判断。

BCKD [算子包](existing_assets/bckd_operator/README.md)和 [90 的 6/6 回执](existing_assets/bckd_operator/CPU_RECEIPT_90_attempt1.json)已落盘。BCDL 的学生相关差值权重有梯度：detach 或把加权 BCE 换为加权 KL 都不等价。完整检测头、定位分支及 MMCV reduction 尚未执行。

## 数据、资源、交接

数据恢复已于 02:41:55 完成，耗时 806.68 秒，完整解码和标签计数通过：[总回执](server_state/data_receipt_success.json)。LLVIP fit/dev 9,619/2,406；Drone train/val 17,990/1,469，独立双标签；test 未提取/读取。

本轮未连接 94，未启动 GPU 训练/评估/探针，未改其他 GPU 任务。[10:17 只读快照](server_state/resource_readonly_1018.json)未发现本项目根目录的常驻进程；其他 ydf 项目不能当成本轮实验。GPU 工作继续走迁移后的统一 guard 与同一 lease 路径，不新建资源池。CPU 接线通过不等于 GPU 准入。

下一可比训练依赖真实 IR 权重与性能短测：优先从 90 基线重建形成 N/IR，再绑定 CMD；BCKD 可并行补检测头适配。融合原协议重训不抢这批算力，作者资产缺失也不阻塞其他可执行路线。

远端根：`/mnt/dataX/ydf/projects/RGBT_campaign_90`。证据：`artifacts/reproduction_20260909_attempt1/`，本地与远端互指。全部新文件在 dataX，不写 dataY 大文件或系统盘。

使用 research-lit 与 experiment-audit；遵用户规则未算新 hash，以来源、既有 revision、大小、源码和回执留证。新独立 reviewer 因线程上限未能创建，实际为复用代理的代码/范围审阅，不称 fresh 或跨模型。[证据审阅](EVIDENCE_REVIEW.md)只接受上述有限结论，不接受论文增益。
