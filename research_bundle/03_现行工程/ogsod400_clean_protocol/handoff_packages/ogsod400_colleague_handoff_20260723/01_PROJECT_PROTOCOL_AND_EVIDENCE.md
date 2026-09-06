# 1. 项目、协议与证据边界

## 1.1 科学问题

部署模型只能读取 SAR，训练时允许读取与 SAR 配准的 RGB。目标不是简单证明 RGB teacher 更强，而是回答：

> 在强 same-modal SAR 蒸馏锚点之上，能否筛出 RGB 中对 SAR 检测真正有增量、且不会造成跨模态负迁移的局部训练信号，使最终模型仍为 SAR-only detector？

这要求同时回答三个不同问题：

1. **Efficacy**：方法是否胜过 matched `H_S`？
2. **RGB attribution**：增量是否依赖正确配对 RGB，而不是容量、剂量、随机正则或 GT 聚焦？
3. **Interpretation**：该信号是否能被称为 shared/reachable/task-compatible？这需要额外识别假设与干预，不能由相关性、重建或单个梯度符号自动推出。

研究中最容易出现的错误，是只做 `paired > shuffled` 就声称 RGB 有净价值，或者只做 `method > SAR baseline` 就忽略了更强的 same-modal `H_S`。本项目当前统一以 `H_S` 为默认强锚点。

## 1.2 基础检测与蒸馏结构

- Student：YOLO11n，训练和推理输入均为 SAR；从 `yolo11n.pt` 初始化。
- Teacher：冻结的 RGB 和/或 SAR YOLO11n checkpoint，只在训练期使用。
- CMDistill 适配实现包含三类 loss：
  - P3/P5 上的 PCC feature distillation；
  - P5 上的 SLRD relation distillation；
  - Detect 输出上的 IBCLD output/logit distillation。
- `full RGB-CMD`：feature/relation/output 均来自 RGB teacher。
- `H_S`：feature/relation/output 均来自 SAR teacher，是强 same-modal anchor。
- `H_F`：仅 feature 来自 RGB，relation/output 来自 SAR；它是 source swap ablation。
- `H_R`：feature/output 来自 SAR，relation 来自 RGB。
- `H_O`：feature/relation 来自 SAR，output 来自 RGB。
- `H_RO`：feature 来自 SAR，relation/output 来自 RGB。

代码中的冻结映射见：

```text
comparison/runtime/mm_arcs_v2_r2a_hbb/
  src/teacher_student_decomposition_kd_hbb/hybrid_teacher.py
  src/teacher_student_decomposition_kd_hbb/loss.py
```

`HybridTeacherRoute` 将 component→teacher 映射做成固定枚举，禁止自由字符串组合；runtime 对 teacher、student init、dataset YAML、loss/trainer/router 源码做 SHA 绑定，并在 resume 时复核。

## 1.3 OGSOD 固定主协议

协议 ID：`ogsod_yolo11n_nomosaic_direct400_sgd_b64_v1`

| 项 | 固定值 |
|---|---|
| 数据 | OGSOD-1.0 legacy file split，HBB |
| 模型 | YOLO11n，`yolo11n.pt` 初始化 |
| 训练 | exact 400 epochs，patience=400 |
| 输入/批量 | imgsz=256，strict batch=64，workers=8；部分锁定 CMD exact400 runtime 为 workers=2，必须使用独立 protocol_id |
| 优化 | SGD，lr0=0.01，lrf=0.01，momentum=0.937，weight_decay=0.0005 |
| warmup | 3 epochs，momentum=0.8，bias_lr=0.1 |
| schedule | `cos_lr=false` |
| 增广 | mosaic/mixup/cutmix=0；translate=.1，scale=.5，fliplr=.5；其余固定为 0 |
| 复现 | deterministic=true；当前正式推断只允许 seeds 42/123，seed0 仅历史上下文/工程 smoke |
| endpoint | SAR-only AP50、AP50-95、best/final epoch；epoch 不是独立 replication |

任何偏离都应创建新 protocol_id，不得覆盖旧行。完整 YAML：

```text
configs/protocols/ogsod_yolo11n_nomosaic_direct400_sgd_b64_v1.yaml
```

## 1.4 SiXiang 协议和 split

原始 shipped split 存在 scene overlap，已判为 `KILL`，其结果禁止用于科学结论。当前 scene-clean coarse10 split：

| split | scenes | images |
|---|---:|---:|
| train | 110 | 2,305 |
| val | 14 | 304 |
| test | 14 | 573 |

固定开发协议为 `sixiang_yolo11n_nomosaic_direct300_sgd_b64_v1`，同样使用 YOLO11n、SAR-only evaluation、SGD、batch64、无 mosaic 的 direct300 训练。当前 val 是开发集；test 仍未读取。scene clean 解决了明显的 scene overlap，但最终 claim 仍需 acquisition/near-duplicate/group audit。

## 1.5 证据等级

本项目采用以下优先级：

```text
不可变 primary artifact/config/code
  > 经过验证的 campaign/run ledger
  > claim ledger
  > 当前 reset-v2 synthesis
  > 历史叙述、旧 handoff 与零散日志
```

状态含义：

| 标签 | 能说明什么 | 不能说明什么 |
|---|---|---|
| engineering | 入口、tensor、checkpoint、hash、队列、finite/no-op 合同可工作 | 方法有效 |
| proxy/diagnostic | 某种 predictability、reachability、replay 或局部 task signal 存在 | detector AP 增益或因果机制 |
| exploratory development | 一个冻结开发集/协议中的 outcome | paper-ready、跨数据泛化 |
| validated development | 完整矩阵、validator、controls 与 analyzer 通过 | untouched/external generalization |
| paper-ready | 独立 evaluation、至少两个数据集/组、完整 controls 与 novelty | 当前没有任何新方法达到此级别 |

所有 observation 都要经过：

```text
observation → integrity → interpretation → alternative explanation
→ supported claim → unsupported claim → next decision
```

## 1.6 本地与 L20 的职责划分

```text
本地 macOS
├── 方法源码、冻结配置、validator/analyzer、合成测试
├── registry/campaigns.csv 与 registry/claims.csv
├── reset-v2 审计和远程 manifest 镜像
└── 无 CUDA、无 ultralytics：不能运行正式训练

L20
├── /private/projects/ogsod400_clean_protocol  工程镜像、队列状态、训练日志
├── /private/results                           原始结果根、失败/重试/正式 run
└── 两张 NVIDIA L20                           正式训练执行环境
```

本地文件不是远程 raw result 的替代品；远程目录也不是自动有效的证据。二者通过 protocol、args、checkpoint/source/data SHA、registry 行和 analyzer commit artifact 对齐。

## 1.7 数据和发表边界

### OGSOD

历史 `images/test` 已反复被用于验证、选择与多轮方法开发。因此：

- 可以作为历史 benchmark 和 contaminated development evidence；
- 不能作为 untouched test；
- 不能把 optimizer seed 当成独立 dataset/group replication；
- 任何最终稿都必须明确披露这一点。

### SiXiang

- scene-clean val 可以用于当前方法开发；
- test 目前保持 untouched；
- 只有通过开发门后才能在新 freeze 下使用 test，且应补 acquisition/近重复审计；
- 不得在看到 val 结果后原地调整 q、mass、gate 或阈值并仍称预注册确认。

### DroneVehicle 与 M4-SAR

- DroneVehicle 是 RGB-thermal，不是 SAR；可作为 secondary mechanism/generalization test，不能代替 optical-SAR 主证据。
- M4-SAR 是优先外部 optical-SAR 候选；当前只完成获取/审计/readiness 基础设施，没有可用 detector outcome。

## 1.8 研究治理

- 昂贵 campaign 必须先冻结 claim、estimand、强锚点、controls、hard gates、compute ceiling 和 forbidden rescues。
- preflight 失败是工程失败，不是假设失败。
- 进行中的 run 只做低频只读监测，不按 AP 排名、不 outcome-based 停止/救援。
- B2 的结果禁读条件为：未达到 `21/21 completed + terminal validator PASS + commit-last analyzer`。
- 外部模型评审需要当前用户显式授权和脱敏；本交接没有发起新外部评审。
