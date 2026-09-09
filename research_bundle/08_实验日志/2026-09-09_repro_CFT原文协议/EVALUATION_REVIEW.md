# CFT LLVIP 作者权重官方测试集复评审阅

结论：已完成的 3,463 图、7,931 GT 复评产物闭合，未发现空预测图漏计或 NMS 超时截断。执行结果 AP50/AP75/AP50:95 为 **97.3769069 / 72.8913483 / 63.5374776 %**；同执行环境的 CPU 复算三个值完全相等。相对论文表的 97.5 / 72.9 / 63.6，只有 AP75 在一位小数上吻合，不能写成“三项精确复现”。总体 `warn`，表示历史协议和跨环境数值限制仍需保留，不表示本次测试执行失败。

2026-09-09；审阅者 `/root/port90_code_inventory`。复用既有 agent，独立于本次执行者，非 fresh-context、非跨模型审阅；无法独立观察模型身份。使用 experiment-audit 技能，并遵循本轮明确的“不新 hash、复用 agent、仅 CPU 本地审阅”覆盖要求。本审阅未 SSH、未调用模型、未训练、未修改原产物。

**范围与结果**

本次是 LLVIP RGB+IR 双输入、person 单类 HBB 的作者已训练 CFT 权重复评。原始入口 `test.py:test`，作者 paired loader、匹配和 AP 函数保持执行路径；checkpoint 运行时记录为 206,198,710 参数、1,015 层。运行 ID `cft_author_full64`，2026-09-09 11:03:27 至 11:04:52（+08:00），完整用时回执约 90.36 秒。冻结配置为 batch 64、imgsz 1024、CUDA FP16、conf 0.001、NMS IoU 0.5、rect/pad=0.5、无 TTA，`save_hybrid=False`。这不是 RGB-only 蒸馏、DroneVehicle 评估、三 seed 增益或从头训练复现。

论文目标已直接核对 [CFT arXiv v2，Table 3（Table 1 相同）](https://arxiv.org/html/2111.00273v2#S4.T3)。所有下表数值单位为百分数；差值为百分点，原 JSON 中 AP 的单位是 0–1 比例。

| 指标 | 原执行/同环境 CPU | 本地 NumPy 2.3.5 CPU | 论文 | 原执行−论文（百分点） | 原执行一位小数 |
|---|---:|---:|---:|---:|---:|
| AP50 | 97.3769069033 | 97.3764342331 | 97.5 | -0.1230930967 | 97.4，未吻合 |
| AP75 | 72.8913482579 | 72.8943134882 | 72.9 | -0.0086517421 | 72.9，吻合 |
| AP50:95 | 63.5374776356 | 63.5370757591 | 63.6 | -0.0625223644 | 63.5，未吻合 |

本机将源码快照中的 `ap_per_class` 和 `compute_ap` 两个函数经 AST 原样提取，在已有 NPZ 上运行；没有另写 AP 公式替换作者实现。作者按置信度汇总所有图的预测，按类计算累计 TP/FP，对 101 点插值精确率曲线作梯形积分，再取 10 个 IoU 的平均；不是图均值，也未调用 pycocotools。AP75 是第 6 个阈值（0.75），AP50:95 是 0.50 至 0.95 共 10 个阈值；作者匹配条件是严格 `IoU > threshold`。

本地首次以 `1e-12` 容差核对原回执未通过，最大比例差 `2.9652302652838358e-05`，即 **0.0029652303 个百分点**；该不一致保留为 warning，未放宽容差伪装通过。原运行环境 NumPy 2.2.6，本地 2.3.5；18,275 个预测只有 5,220 个不同置信度，其中 16,090 个预测处在 3,035 个同分组中，最大组 43 个。作者 `np.argsort(-conf)` 使用非稳定默认排序，本地默认顺序与 stable 顺序确有区别；版本/平台同分排序是合理解释，但本审阅没有隔离证明具体后端原因。执行者随后提供的 `server_execution/pinned_cpu_recompute.json` 记录：同环境 2.2.6、原函数、原数组、无新推理且 CUDA 未初始化，三个 AP 对原执行差值均为 0。本审阅独立核对了这份回执的值和数组范围，未声称自己在 90 执行了它。

**完整性核对**

| 项目 | 观察与判定 |
|---|---|
| 图像覆盖 | `evaluated_roster.json` 包含 3,463 个唯一 stem，RGB/IR 顺序一致；wrapper 检查与完整 input list 集合一致；日志 seen=3463，55/55 batches，exit=0、receipt=COMPLETED。 |
| GT | `target_cls` 长度 7,931，全为 person 类 0，与 previous annotations 审计一致；RGB/IR scan 都是 3,463 found、0 missing/empty/corrupt。 |
| 预测 | 3,461 个非空 TXT，共 18,275 行；无清单外 TXT。六列均有限，类全部为 0；按 loader roster 顺序串联后，全部置信度 token 与 NPZ 经作者 `%g` 格式写出完全一致，类别顺序也完全一致。 |
| 两张无预测图 | `240113`、`240165` 没有 TXT，属于未检出而非漏评价。作者 `test.py:138-143` 先 seen+=1，再在空预测时保存该图 tcls；不会为该图创建预测 TXT。全量 GT 分母仍为 7,931。 |
| TP 数组 | bool[18275,10]，各阈值 TP 数为 7807/7700/7554/7306/6888/6292/5292/3864/2117/445；逐预测 TP 随 IoU 不增加，各列不超过 GT 数。 |
| NMS | 合并日志没有 `NMS time limit`；作者 `utils/general.py:540-541` 超时时打印并 break，实际 wrapper 的 73-75 行捕获该输出并抛错，使超时不能进入 COMPLETED。没有本次 NMS 截断证据。 |
| 指标捕获 | `executed_wrapper.py:140-144` 保存原四个参数后直接委托原 `ap_per_class`；非仅存在未运行的函数定义。 |
| 原产物保护 | 审阅前后检查全部固定输入和 3,461 个 TXT 的 size/mtime_ns 不变；没有生成摘要或修改原始结果。此检查弱于内容摘要，不作字节身份保证。 |

红外 scan 描述里再次打印 visible cache 是作者 `utils/datasets.py` 的日志变量误用：实际读取 `cache_ir_path`，显示描述使用 `cache_rgb_path`。结合配对 loader 清单检查，没有据此认定红外错误复用 RGB 的证据。

**必须随结果保留的限制**

1. 主评使用独立保存的官方 previous annotations，测试 7,931 GT；用户上传的 2023-02-21 更新版为 8,302 GT，差 371。两者不能不加说明地混比分数。转换遵循当前作者工具：1280×1024 归一化、无中心减一、六位小数、不滤 difficult；XML 预审确认全 person、整型坐标。本次独立审阅读取了审计回执和实际 AP 分母，没有重读全量 XML/图片，因此不证明与 CFT 私有历史 YOLO 标签逐字节一致。
2. 作者公开代码 revision 为 `fb591c9b163177c0e950db08e213e24ddc912d41`。旧 checkpoint 的 24 个融合子块条件重绑至当前 `myTransformerBlock`，运行回执记录原参数/buffer 对象不变、无随机参数。历史训练源无法完整确认，不将兼容运行声称为未知原训练实现数值等价。另有 torch 2.10 旧 pickle 加载与 NumPy 别名适配。
3. NPZ 已含作者生成的 TP 标志，本审阅验证其形状、范围、单调性和 AP 聚合，**没有独立重算预测框与 GT 框的 IoU 配对**；TXT 不含 GT 坐标，无法单凭这些小产物完成这一层。
4. 身份维持 `PAPER-RECONSTRUCTED checkpoint reevaluation`。官方测试已因本次作者协议复评而公开读取，结果不用于我方新方法选择，不能再称它对本研究完全未见。MR 5.40 不在已执行指标内。没有三 seed、训练收敛、跨数据集、基线增益、四臂归因或从头训练结论。

**复核产物与命令**

- `audit_author_evaluation.py`：CPU NumPy 审阅脚本；27 项通过，2 项保留跨环境精确相等 warning；不导入模型。
- `EVALUATION_RECOMPUTE.json`：逐项核查、本机 10 阈值 AP、原执行差值、同环境 CPU 回执、全部声明输入、保护检查和限制。
- `EVALUATION_REVIEW.json`：机器可读 verdict 与逐 claim 影响。
- 原证据均仍位于 `server_execution/`，包括 `official_test_attempt1/` 的 source snapshot、NPZ、TXT、plan/receipt/compatibility，以及原日志、标签审计、协议和 pinned CPU 回执。

PowerShell（已有依赖，无安装；从实验目录运行）：

```powershell
& 'C:/Users/MSI-PC/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -B './audit_author_evaluation.py'
```

脚本仅重写派生文件 `EVALUATION_RECOMPUTE.json`；遇到原产物完整性失败会报错，跨环境浮点差异明确记录为 warning。审阅范围内没有发现需撤销本次执行结果的缺陷；应按上述有限复评口径引用。
