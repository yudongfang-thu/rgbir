# CGA-KD W1 · native weight0 臂（seed42，DroneVehicle）

> **一句话**：已完成。协议匹配 3 seeds 对照：现有全量 KD（L）比 no-KD（N）**净负 −0.35 mAP50-95（3/3 seeds）**，且昼夜两桶都负——预注册门 1 通过，负迁移被干净证实；但负迁移不随条件集中，H3 门控机制失去基础。
> 服务器：`94` GPU4（screen `cgkd_native_s42`，经 project_resource_guard 租约，expected-vram 10000MiB / free-safety 2048MiB）。
> 输出：`94:RGBT_campaign/runs/cgkd_w1/native_rgb_s42_e200/`；日志 `94:RGBT_campaign/artifacts/cgkd_20260905/w1_native_s42.log`。

## 目的
补齐 adapted_v2 协议（b32/e200）下缺失的协议匹配 no-KD 对照——此前只有 formal_native（b32a2）版本，与蒸馏臂协议混杂无法直接对比（见昼夜分桶条目的警告）。

## 设置
- recipe 冻结自 `cmdistill_protocol_drone.yaml`：e200/b32/640/SGD lr0.01 lrf0.01 mom0.937 wd5e-4/mosaic0/translate0.1/scale0.5/fliplr0.5/hsv0/erasing0/patience0/deterministic/seed42
- 学生 RGB（rgb.data.yaml），无任何 KD；脚本 `train_native_rgbt.py`（本目录）


## 结果（2026-09-06 凌晨完成）

**三 seed 全部 200 epochs 完成，同规则评估（eval_rgbt_detector，last.pt，val 1,469 对）：**

| 模型 | s42 | s0 | s123 | mean±SD (mAP50-95) | mean±SD (AP50) |
|---|---|---|---|---|---|
| **N native（本臂）** | 53.82 | 54.36 | 53.69 | **53.954±0.356** | 76.160±0.527 |
| **L cmdistill_corrected** | 53.24 | 53.90 | 53.67 | **53.605±0.332** | 75.883±0.622 |

**N vs L（协议匹配，3 seeds）：L − N = −0.35 mAP50-95 / −0.28 AP50，逐 seed N≥L 为 3/3。**

**昼夜分桶（seed42，ultralytics val，亮度中位数桶）：**

| 桶 | N | L | L−N |
|---|---|---|---|
| day | 80.31 / 62.80 | 79.71 / 62.17 | −0.60 / −0.63 |
| night | 71.18 / 45.10 | 69.42 / 44.32 | **−1.75 / −0.78** |
| full | 76.01 / 53.82 | 75.17 / 53.26 | −0.85 / −0.55 |

## 结论
1. **预注册门 1 通过**：day 桶 mAP50-95 下降 0.63 ≥ 0.3——现有全量 KD（PCCFD/SLRD/IBCLD）在协议匹配条件下相对 no-KD 为净负，负迁移在开发集上被干净地证实。
2. **但负迁移不随条件集中**：night 桶掉点（AP50 −1.75）≥ day 桶（−0.60），尽管 IR 教师在夜间比 native 强 +11.2。**教师优势与蒸馏净收益脱钩**——审计 §5.3 的警告（夜间 IR 独有目标最难被 RGB 学生吸收）得到支持，"按亮度路由即可修复"的 H3 机制**失去实证基础**。
3. 该负结果本身可引用：一个已发表风格 KD bundle 在协议匹配、3 seeds 下净负——这是审计要求的"协议匹配基准闭合"产出。
4. 对方法设计的含义：问题不在门控（When），在于**蒸馏信号本身**（What）——G/I 组件若要成立，必须给出与"全量特征模仿"本质不同的信号，而不是在负信号上做加权。

## 局限与下一步
- 单模态标签评估的口径限制仍在（审计 §5.3）；test 未动。
- 按审计 P0：G/I 实现契约修复前不启动 W2；I 臂维持降级。
- P1 方向：共同目标错误分解（teacher 更准/student 有观测/几何可对应的区域划分）定义最小干预。


## 结论
（待结果）

## 局限与下一步
- W2 待实现代码：G（DFL 分布蒸馏）、I（可学习不变分解+配对对抗）、C（ISP 门控）
- GPU 排程：seed123 已完成释放；W2 可用 3 卡并行（4/5/6）
