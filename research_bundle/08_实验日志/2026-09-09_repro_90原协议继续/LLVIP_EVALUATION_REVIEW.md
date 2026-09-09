# LLVIP 作者两份单模态权重复评：有限范围审阅

**两次完整复评的覆盖与指标产物闭合；本机 CPU 按各次原源码重算三个 AP 和全部十个 IoU 列，与实际回执完全一致。** 两份结果接近论文 v2/v4 表3，但不能称六项都在公布精度上完全复现。总体 `warn` 仅保留历史来源、未重做框匹配和设备环境变量的范围限制，未发现使本次两份 AP 失效的执行缺陷。

2026-09-09，审阅者 `/root/port90_code_inventory`。复用 agent、与远端执行者分开；本审阅者也编写过初始包装，因此不是盲审或独立于代码作者的审阅，不是跨模型审阅。没有 SSH、模型前向、训练、阈值改变或新摘要；仅新建本报告、JSON、CPU脚本及派生复算结果。

| 项目 | visible | infrared |
|---|---:|---:|
| 实际 AP50 / AP75 / AP50:95（%） | 90.788183 / 56.342893 / 52.664239 | 96.378278 / 76.420674 / 67.014908 |
| 图像 / GT | 3463 / 7931 | 3463 / 7931 |
| 预测 TXT / 预测框数 | 3461 / 44361 | 3462 / 25061 |
| 无预测 TXT 的 stem | 210359、210382 | 210176 |
| 完成批数 / 退出码 | 109 / 0 | 109 / 0 |
| 包装计时 / 外层进程计时（秒） | 90.695608 / 94.606520 | 88.061278 / 92.390857 |

每份 roster 均有3,463个唯一stem；TXT无额外条目、无空文件。按实际loader顺序串联所有六列TXT后，置信度token与原NPZ的作者 `%g` 序列完全一致，类别同为person=0。无预测图由原 `source_snapshot/val.py:194-199` 先计入 seen，再保留该图 GT；因此少2份/1份TXT不代表漏评价。最终日志、NPZ target_cls长度和独立数据视图回执都为7,931 GT，原TP列随IoU单调且不超过GT总数。

原 `val.py`、`utils/metrics.py`、实际 `executed_wrapper.py` 的路径已经核对。包装保存四个原参数后委托作者 `ap_per_class`；本机只经AST提取该函数与 `compute_ap`，没有导入模型。NumPy 2.3.5 本地复算与 NumPy 2.2.6 运行回执三个 AP 及十列 AP 在 `1e-12` 容差内全部通过，实际差值均为0。作者原101点插值后梯形积分、单类汇总、AP75列5和十阈值平均都保留；不是逐图平均或CFT指标函数替换。

两份日志均完成109/109批，无 `NMS time limit`；实际包装在捕获这一原NMS警告时抛错，故不能静默接受NMS超时截断。冻结值都是单模态YOLOv5l、1280、B32、FP32、conf=.001、NMS IoU=.6、rect/pad=.5、无TTA/混入GT。与CFT的1024/B64/FP16/NMS .5及匹配细节不同，不能混作同条件方法增益。

**论文版本解释：保留运行前冻结的v1参照，另加v2/v4参照；没有覆写原协议或重跑。** 本审阅直接核对了 [v2 Table 3](https://arxiv.org/html/2108.10831v2#S5.T3) 与 [v4 Table 3](https://arxiv.org/html/2108.10831v4#S5.T3)，两版YOLOv5l列相同，且正文把划分更新为77.6%/22.4%。[v1 Table 3](https://arxiv.org/html/2108.10831v1#S5.T3)对应较早数据描述和数值，不能把超过旧表的差值包装成新的方法增益。

| 模态 | 冻结v1参照 AP50/AP75/AP（%） | 补充v2/v4参照（%） | 实际−v2/v4（百分点） | 一位小数吻合 |
|---|---|---|---|---|
| visible | 90.8 / 51.9 / 50.0 | 90.8 / 56.4 / 52.7 | -0.011817 / -0.057107 / -0.035761 | AP50、AP50:95；AP75未吻合 |
| infrared | 94.6 / 72.2 / 61.9 | 96.5 / 76.4 / 67.0 | -0.121722 / +0.020674 / +0.014908 | AP75、AP50:95；AP50未吻合 |

**实际执行版本是16,888字节的旧包装。** 它在第97行先调用 `set_per_process_memory_fraction(.68,0)`，第259行仍给作者入口传 `device=protocol['device']`（逻辑0），并没有使用后来新增的 `device=''` 修补。PyTorch 2.10这一分配器调用先执行CUDA初始化，见[对应官方源码](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/cuda/memory.py)。作者 `select_device('0')` 后续虽改写CUDA_VISIBLE_DEVICES字符串，本次lease及进程NVML记录均指向物理GPU3，各一个CUDA进程、峰8,908MiB，未发现误跑物理GPU0的证据。此结论仅适用于本次已核到的初始化顺序与实测记录；旧包装不应作为不会重写环境变量的通用保证。原执行快照保持不变，后续设备修补不能倒写为本次已采用。

原缓存 `get_hash` 的MD5已被**实际执行快照**替为普通path/size元组；独立baseline视图避免改写CFT缓存。主评仍为官方previous标注，现代更新版8,302 GT没有混入。身份维持 `PAPER-RECONSTRUCTED`：没有独立重读全量XML/图片、没有重新计算框与GT的IoU匹配、没有复跑模型或验证历史训练；这不是三seed增益、训练复现、MR复现或泛化结论。所有读取原件的size/mtime_ns前后不变；依用户指令不算hash，因此不宣称内容摘要级身份保证。

复核脚本 `llvip_evaluation_recompute.py`、全部逐项结果 `LLVIP_EVALUATION_RECOMPUTE.json`、机器可读短结论 `LLVIP_EVALUATION_REVIEW.json`。CPU执行命令：

```powershell
& 'C:/Users/MSI-PC/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -B './llvip_evaluation_recompute.py'
```
