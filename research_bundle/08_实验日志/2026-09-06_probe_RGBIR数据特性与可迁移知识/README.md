# RGB–IR 数据特性与可迁移知识诊断（2026-09-06）

> **结论：已完成三数据集、六baseline、521对开发图像诊断。DroneVehicle优先对象级前景/类别判别迁移，现有模型不支持普遍IR定位优势；LLVIP有更明确的定位机会；VEDAI应区分RGB与NIR并核验教师方向。** 当前实验尚无稳定超过强同模态对照的RGBIR增益证据。本轮结果为描述性诊断，不宣称新方法有效。

直接阅读：[完整研究报告](../../07_研究分析/RGBIR数据特性与蒸馏方向诊断_20260906.md) · [12组特征图册](特征图册.md) · [三组配准与局部边缘图](registration_panels/README.md)。

## 目的
为弱势模态单模态部署选择可学习、任务相关、可对应的教师知识，并识别负迁移风险。优先覆盖已有有效 baseline 的 DroneVehicle、LLVIP、VEDAI；对其余 RGB–IR 数据集核查数据和模型覆盖，缺模型时明确边界。

## 设置与冻结分析协议
- 只用训练/开发/验证划分，不访问封存 test 来选择方法；不训练新 KD 方法。
- 核验模型 args、数据集、类数、输入模态、权重 SHA 与端点。不同任务模型不构成 native/KD 因果对照。
- 每数据集以固定 RNG seed=20260906 从开发配对列表抽取最多200对；先用2对 canary 验证实际显存峰值，串行加载模型/小batch推理。
- 图像：几何尺寸、亮度/对比度、梯度相似度、边缘最近邻距离、phase-correlation 位移及响应。跨模态纹理差异会污染图像配准代理，不能把低相关直接当错配。
- 标签：同类双模态框一对一匹配，报告 IoU、相对目标尺度中心偏移、数量/类别差异；模态单侧标注不等价于另一侧物理不可见。
- 特征：baseline P3/P4/P5（适用架构）通道能量图、正确中心化 linear CKA、paired与固定独立随机donor的多次null、前景/背景分开；不逐通道比较独立训练模型的语义。
- 检测：同模态GT检测误差与共同目标对上的teacher/student定位、分类、漏检互补；阈值conf=0.25/IoU=0.5为描述性诊断，AP结论以标准完整评估记录为准。
- 图像实例展示按固定ID或预先定义的亮度分位选取，不按方法收益挑图。
- 数据特性不推出方法gain；方法假设须经匹配native/same-modal/paired/shuffled、3seeds与封存test检验。

### 正式probe前工程修订（不据方法outcome调参）
- 2图canary只检查加载/shape/显存；初版20×20特征池化会抹去小目标，正式版本保留P3/P4/P5原生网格，并记录每个区域token数，少于8不计算CKA。
- 独立代码审查发现IoU总和匹配可能降低阈值命中数；改为先在同类且IoU≥0.5的边上最大化命中数，再以IoU破平局。标签对以同类IoU≥0.1做同样处理。保留原canary，修订后再canary，不将canary统计作为正式结果。

## 结果

六个模型均为YOLO11n、seed42、E200、last.pt，模型及数据身份见`dataset_model_inventory.md/json`。DroneVehicle/LLVIP正式样本各200对，VEDAI official_fold01全部121对；未混入paper80教师，未访问封存test选方法。

| 数据集 / 教师→学生 | 共同GT | 双方命中 | 教师独有 | 学生独有 | both-hit教师−学生IoU均值 / 教师更准比例 |
|---|---:|---:|---:|---:|---:|
| DroneVehicle IR→RGB | 3083 | 2505 | 360 | 78 | +0.00790 / 49.46% |
| LLVIP IR→RGB | 672 | 484 | 107 | 22 | +0.04197 / 60.95% |
| VEDAI RGB→NIR | 364 | 222 | 41 | 31 | −0.00254 / 45.50% |

命中使用conf≥0.25、同类IoU≥0.5、一对一有效匹配数优先，不是AP。独立候选复核：Drone的360个教师独有对象中118个已有学生低置信正确框，LLVIP严格对应25/107（另2个高置信分配竞争），VEDAI20/41；错误类型flag有重叠，不可相加。Drone最暗四分位包含320/360个教师独有命中，但这不是亮度门控增益证据。

前景paired−donor centered CKA：Drone P3/P4为0.582/0.575，LLVIP为0.306/0.428，VEDAI为0.466/0.496；有效图数分别200/198、200/200、93/33。VEDAI P5前景仅1图有效，不用于层级判断。三组整体对象对应可开展诊断，LLVIP/VEDAI共享标签IoU=1不是独立配准证明。

当前实验复核：Drone CMDistill−新native为−0.349±0.292 mAP百分点（3seeds）；HNEWA-inspired paired−same-modal为Drone +0.207±0.360、LLVIP +0.091±0.592（各2/3正）。主要N/L recipe实际一致，不能凭b32a2命名认定预算不同；新native分桶有重评脚本与完成日志。详见`current_results_notes.md`及来源快照。

## 结论

1. 当前RGBIR有共同对象和错误互补，但“教师更强”没有稳定转化为跨模态独特收益。
2. Drone优先研究学生已有弱线索时的前景/类别判别知识；LLVIP保留选择性定位；VEDAI关注RGB→NIR类别/排序，不能泛化成热红外路线。
3. 特征相似性和热图只提供机制筛查，不决定可学收益；教师独有命中不等于学生可达上限。
4. 最小后续验证为单一知识、四臂加同剂量随机选择、3seeds；必要时加同mask GT-only排除难例重权。当前没有启动新KD训练，设计待正式训练前冻结。

## 产物路径
- 本地：本目录。
- 94：`/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_transfer_diagnosis_20260906/`。
- 研究报告：`07_研究分析/RGBIR数据特性与蒸馏方向诊断_20260906.md`。

关键文件：

- `dronevehicle_full/`、`llvip_full/`、`vedai_full/`：summary、逐图/逐目标/逐层统计、完整预测、模型与样本manifest、激活图、energy_maps及完成回执。
- `probe_rgbir.py`、`probe_config.json`、`run_probe.sh`；实际远端v1/v2脚本、v2 canary与正式日志保存在`remote_execution/`。原始canary只作工程检查，不与正式结果混合。
- `probe_analysis.json`、`analyze_probe_outputs.py`、`candidate_flag_audit.json`、`audit_candidate_flags.py`：CPU错误候选和条件性复核。
- `registration_panels/`：三图PNG/PDF、选图元数据、CPU脚本和哈希校验回执。
- `current_results_notes.md` / `current_results_sources.json`、`dataset_model_inventory.md/json`：实验与模型身份复核。
- `observed_feature_diagnosis.md`、`probe_review.md`：独立统计/图像诊断及v1→v2代码审查。
- `evidence_and_method_notes.md`：文献边界；其中前置H_loc是假设，优先级已经由本报告和实测诊断更新。
- `minimal_kd_validation.md`：按实测收敛的后续验证设计；`delivery_verification.json`：本地交付一致性核查。

94运行：三组v2 canary成功；正式screen会话为`rgbirprobe_{drone,llvip,vedai}_full_s42`，全部完成并退出。每卡一个任务，GPU4/5/6；PyTorch peak reserved≤94MiB，NVML进程抽查约550MiB，guard按6GiB预留。所有输出位于dataset盘，无新长训。

## 局限与下一步
只诊断已有数据和模型，区分共享结构、教师额外观测、可迁移性与实际训练收益。单seed模型、开发图抽样、GT匹配代理、both-hit条件选择、低置信候选和亮度场景混杂的限制见完整报告。FLIR/M3FD尚无核实的项目双模态baseline与冻结开发划分，KAIST仍未解压，未冒用其他模型生成它们的baseline特征图。方法与长程baseline补训不由本probe替代。
