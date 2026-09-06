# SpaceNet6 full-622 locked-test 与 W8 部署链结论

证据截止：2026-08-26 18:00 CST。

## 1. 冻结问题与完成状态

在 development 阶段完成方法筛选后，监督蒸馏方法冻结为 P3 paired optical DFL。随后使用完整 622 张训练图，对 `native`、`p3` 和 `p3_same_modal` 分别训练 detector seeds `0/42/123`，共九个 B32/A2、E300、final-epoch endpoint。九个 endpoint 全部完成后，才统一读取并评估 200 张 locked test。

本阶段没有根据 locked-test 结果修改 gate、loss、权重、seed 或 checkpoint 选择规则。

权威结果：

- `runs/p3_full622/locked_test_fp/summary.json`
- `runs/p3_full622/locked_test_fp/results.csv`
- `runs/p3_full622/locked_test_w8_summary/summary.json`
- `runs/p3_full622/locked_test_w8_summary/results.csv`
- `runs/p3_full622/locked_collect_bootstrap/isolated_seed_{0,42,123}.json`

## 2. Locked-test FP 结果

标准 Ultralytics rect validation，数值为三 seed mean ± sample SD，单位为百分数。

| 方法 | AP50 | AP75 | mAP50-95 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| native | 65.280 ± 0.450 | 23.401 ± 0.225 | 29.747 ± 0.212 | 75.730 | 57.650 |
| P3 paired optical DFL | 65.277 ± 1.388 | 23.726 ± 1.123 | 29.864 ± 0.904 | 76.387 | 57.482 |
| P3 same-modal SAR DFL | 65.207 ± 0.486 | 23.056 ± 1.133 | 29.629 ± 0.505 | 76.533 | 57.327 |

P3 paired optical DFL 相对 native 的逐 seed 差值：

| Seed | AP50 | AP75 | mAP50-95 |
|---:|---:|---:|---:|
| 0 | -1.331 | -0.731 | -0.748 |
| 42 | +0.778 | +1.593 | +0.664 |
| 123 | +0.542 | +0.113 | +0.435 |
| Mean | **-0.003** | **+0.325** | **+0.117** |

结论：development 上的稳定正收益没有在 locked test 上可靠复现。AP50 实质持平；AP75/mAP 只有很小的平均正差，均为 2/3 seeds 正向，并受到 seed0 明显反转影响。

Optical teacher premise 仍成立：locked test 上 optical teacher 为 `91.646/59.010/54.600`，SAR teacher 为 `63.886/21.490/28.682` AP50/AP75/mAP。但是教师更强并不自动推出低容量 SAR student 能从 DFL 蒸馏中获得稳定收益。

## 3. Collect-grouped 辅助诊断

为避免三个并行 evaluator 竞争同一个 Ultralytics label cache，每个 collect 现在使用独立软链接 split 和独立 cache。三 seed 均覆盖 48 个 collect、200 张图，结果无异常或缺失。

下表先对同一 collect 的三 seed 差值取平均，再对 48 个 collect 做 10,000 次 paired bootstrap。这里的 estimand 是“collect 等权的 per-collect AP 宏平均”，不是全测试集 AP；由于 AP 非线性且小 collect 样本很少，只作异质性诊断。

| 比较 | 指标 | Collect-macro 差值 | 95% CI |
|---|---|---:|---:|
| P3 − native | AP50 | -0.219 | [-0.862, +0.420] |
|  | AP75 | -1.690 | [-3.646, +0.075] |
|  | mAP50-95 | -0.439 | [-1.039, +0.098] |
| P3 − same-modal | AP50 | -0.352 | [-1.009, +0.253] |
|  | AP75 | -1.147 | [-3.342, +0.908] |
|  | mAP50-95 | -0.311 | [-1.026, +0.347] |

该诊断不支持“P3 对 collect 普遍有效”。它与 whole-test AP 的轻微正差并不矛盾：两者使用不同加权和不同 estimand。

## 4. Fixed640 FP → fake-W8 → ORT-QDQ

部署数值链固定输入为 `[1,3,640,640]`，与标准 rect validation 不同，因此只在同一 fixed640 链内比较。三 seed均值如下：

| 模式 | 方法 | AP50 | AP75 | mAP50-95 | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| Fixed640 FP | native | 65.117 | 24.576 | 30.256 | 75.781 | 56.942 |
|  | P3 | 65.234 | 24.298 | 30.237 | 75.956 | 57.815 |
|  | same-modal | 65.147 | 23.890 | 29.945 | 76.068 | 57.584 |
| PyTorch AIMET W8 fake-Q | native | 64.835 | 23.698 | 29.701 | 74.004 | 57.685 |
|  | P3 | 65.283 | 23.982 | 30.037 | 77.785 | 56.477 |
|  | same-modal | 64.863 | 23.202 | 29.554 | 75.071 | 58.096 |
| ORT QDQ | native | 64.731 | 23.652 | 29.693 | 74.333 | 57.696 |
|  | P3 | 65.078 | 23.625 | 29.919 | 76.188 | 57.196 |
|  | same-modal | 64.931 | 23.314 | 29.542 | 74.799 | 58.056 |

P3 相对 native 的平均差值：

| 模式 | AP50 | AP75 | mAP50-95 |
|---|---:|---:|---:|
| Fixed640 FP | +0.118 | -0.278 | -0.019 |
| fake-W8 | +0.449 | +0.284 | +0.337 |
| ORT QDQ | +0.347 | -0.027 | +0.226 |

Fixed640 FP 与标准 rect FP 在 AP75 方向上相反，说明小效应对评估布局敏感。不能用 fake-W8 的正差反推 FP 主方法已经稳定成立。

W8 与 ORT 相对对应 fixed640 FP 的平均量化变化如下：

| 后端 | 方法 | ΔAP50 | ΔAP75 | ΔmAP | ΔPrecision | ΔRecall |
|---|---|---:|---:|---:|---:|---:|
| fake-W8 | native | -0.282 | -0.878 | -0.555 | -1.777 | +0.742 |
|  | P3 | +0.049 | -0.316 | -0.200 | +1.830 | -1.338 |
|  | same-modal | -0.284 | -0.688 | -0.391 | -0.997 | +0.512 |
| ORT-QDQ | native | -0.386 | -0.924 | -0.563 | -1.448 | +0.754 |
|  | P3 | -0.157 | -0.673 | -0.319 | +0.233 | -0.619 |
|  | same-modal | -0.216 | -0.576 | -0.404 | -1.269 | +0.472 |

P3 在这组模型上表现出较小的量化 mAP 损失，但所有方法的平均损失都小于 1 pp，因此 `qat_needed=false`，不启动新的 QAT。Precision/recall 的变化显示阈值排序发生了明显权衡，不能只用 AP50 概括量化行为。

## 5. 数值链解释

FP PyTorch → FP ONNX 基本一致：九个 endpoint 的 raw box MAE 为 `0.0052–0.0069 px`，score MAE 为 `5.69e-6–6.25e-6`。

PyTorch fake-Q → ORT QDQ 是部署近似，不是逐元素等价：

- raw box MAE：`0.938–1.123 px`；box cosine 约 `0.999982–0.999989`；
- score MAE：`0.001007–0.001139`；score cosine `0.967–0.975`；
- score top-20% overlap：`0.852–0.887`。

第一层 input/weight QDQ 和量化输出已经验证一致，差异主要由深层浮点 kernel 的微小误差反复跨越量化边界累积。另一个语义差异是 PyTorch AIMET fake-Q 使用 FP32 bias，而 AIMET 导出会引入 INT32 bias quantization；补做 bias fake-quant 并没有改善最终一致性。因此 ORT QDQ 应作为静态部署参考，不再使用不适合深层检测图的“逐元素 1 LSB”硬门。

## 6. QCS6490 状态

九个 ORT-QDQ endpoint 均完成 QCS6490 prepare-only AOT：每个图为一个 EPContext、一个 context binary、CPU fallback disabled；源图均含 406 个 QuantizeLinear 和 493 个 DequantizeLinear，目标为 HTP v68 / SoC model 93 / VTCM 2 MB。

这只证明图可编译，不证明真实 QNN HTP AP、延迟、内存或热稳定性。seed42 P3 的正式 200-image 板端包完整保存在服务器：

`/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/artifacts/qnn/qcs6490_full622_seed42_p3_board_package`

本地保存其 manifest 镜像：`artifacts/qnn/qcs6490_full622_seed42_p3_board_package/package_manifest.json`。包内同时包含匹配的 FP raw-head ONNX；上板后会依次执行 HTP raw output、HTP W8 benchmark 和同协议 CPU FP32 benchmark。其 manifest 明确记录 `result_status=not_run_on_board`。当前本机和服务器均没有可访问的 QCS6490 板卡，真实板端验证仍是外部阻塞。

为测量真实 HTP 上的 `P3−native` 方法差，另有 seed42 native accuracy-only 包：

`/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/artifacts/qnn/qcs6490_full622_seed42_native_accuracy_package`

该包只需生成同一 200 张输入的 native HTP raw logits；正式延迟和热稳态只对 P3 最终模型测量，避免对相同推理结构重复性能测试。

## 7. Result-to-claim 裁决

Verdict：`DEFER`，且不允许根据 locked test 重新设计方法。

当前可支持的窄表述：

> 在 full-622 训练和 locked test 上，P3 paired optical DFL 与 native FP AP50 基本持平，同时具有很小的平均 AP75/mAP 正差；其平均 AP50/mAP差值在 W8 fake-Q 和 ORT-QDQ 中保持为正，并观察到较小的 W8 mAP drop。

当前不支持：

- P3 稳定提升全部 FP 指标、全部 seeds 或全部 collect；
- paired optical 在 high-IoU 指标上稳定优于 same-modal；
- W8 下全部指标均占优；
- fake-Q 与 ORT-QDQ 逐元素等价；
- 已证明真实 QNN/QCS6490 保留准确率收益或产生速度、能耗、内存、热稳定性收益；
- 跨数据集、跨架构或普适的跨模态蒸馏结论。

只有真实 QCS6490 HTP 运行完成并保持 ORT-QDQ 的 AP50/mAP方向后，才可以加入收窄的部署保持表述。若板端方向反转，则删除硬件保持主张，不再修改方法。
