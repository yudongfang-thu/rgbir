# OGSOD 实验审计：按 split、协议和 campaign 分层

## 1. 数据与总边界

OGSOD 所有现有 detector 结果使用 legacy file split：训练读 `images/train`，验证读
`images/test`。这个 test 已被长期反复用于方法开发，所以所有数值最多是 contaminated development
evidence；不能称 untouched final test，也不能单独支撑 paper-ready generalization。

L20 SAR/RGB dataset YAML SHA 分别为 `6d64e0ad...` / `e92ad9be...`。本地 YAML 只是路径本地化，
hash 不同但数据语义未发现差异。主 exact400 协议 YAML SHA 为 `4471603c...`。

## 2. 协议分区

| 分区 | 关键设置 | 允许比较 | 禁止比较 |
|---|---|---|---|
| clean exact400 | 400 epochs、imgsz256、batch64、workers8 | 同分区 seed/arm | 与 direct50/100 当同一 estimand |
| CMD exact400 matched-v2 | exact400-scheduled、workers2、22 个 primary-root cells | frozen analyzer 明确列入的同 seed contrasts | external H_S-s0 只作 context；不能偷偷加入 strict contrast |
| CMD direct50 compressed | 50 epochs，7 arms×3 seeds | 同一 21-cell panel | 与 exact400 endpoint 横拼 |
| H_F/H_S direct50 P2 | 50 epochs、workers8、仅 feature source 不同 | seeds42/123 的 formal contrast；seed0 只作历史 context | 与 MM-ARCS 局部 atom 称同一处理 |
| direct100 diagnostics | 多个不同 method/version/proxy | 仅各自冻结版本内 | 把多个失败版本当重复实验 |
| MM-ARCS exact400-scheduled | isolated runtime、workers2、custom locks | 只有 accepted analyzer 解锁后的冻结 contrasts | 手工读 CSV 或与 workers8 clean row 直接合表 |

因此“都是 400 epoch”不等于同协议；worker/runtime/source lock、root identity 与 analyzer eligibility
必须进入账本。

## 3. 已验证的结果

### 3.1 单模态 exact400 baselines

| arm | s0（context only） | s42 | s123 | 当前 formal seeds42/123 mean ± sample SD |
|---|---:|---:|---:|---:|
| SAR | 0.49003 | 0.48840 | 0.48810 | 0.488250 ± 0.000212 |
| RGB | 0.58105 | 0.58743 | 0.58768 | 0.587555 ± 0.000177 |

六个 primary CSV 都是 400 行，args 与 checkpoint 存在，checkpoint CRC PASS。但根据当前治理规则，
OGSOD 正式矩阵只统计 seeds42/123；seed0 保留作历史 context，不计为 formal replication。
这只支持 development split 上存在 RGB-SAR headroom，不支持“RGB 信息一定可蒸馏”。

### 3.2 CMD exact400 matched-v2

primary-root 22 个 run 都是 400 行且 final analyzer integrity PASS；另一个 cross-root H_S-s0 只作
context，analyzer 明确 `contrast_eligible=0`。canonical 报告 SHA 为 `a708ed...`，runs CSV SHA
为 `cbd89fd...`。

| 对比 | endpoint AP50-95 | 证据边界 |
|---|---:|---|
| full RGB-CMD | 0.50478 ± 0.00027，n=3 descriptive context | 含 seed0，已有方法/adaptation，不是新 idea |
| matched H_S | 0.50867 ± 0.00259，n=3 descriptive context | 含 cross-root seed0；不是 formal strict contrast |
| full RGB-CMD − H_S | -0.00262 ± 0.00204，strict same-seed n=2 | full RGB bundle 未胜 strong anchor |
| paired − strict shuffle | +0.03053，seed42 | 说明错误配对伤害大；不证明相对 H_S 有净价值 |
| RGB − SAR feature-only | -0.01206，seed42 | 单 seed source contrast |
| RGB − SAR relation-only | +0.00138，seed42 | 小、单 seed |
| RGB − SAR output-only | +0.00189，seed42 | 小、单 seed |

组件 omission/inclusion 存在交互，不能把 feature/relation/output 边际相加为 full method 的机制解释。

### 3.3 H_F/H_S direct50 P2

6 个 run 全为 50 行，历史三种子分析 JSON SHA `a0eea251...`。只保持 SAR relation/output，交换
P3/P5 feature teacher source：

- s42/s123 formal endpoint `H_F−H_S`：`-0.00885/-0.00978`；seed0 `-0.01359` 只作 context；
- formal mean：`-0.009315 ± 0.000658`；late10 formal mean：`-0.008422 ± 0.000824`；
- 当前 formal seeds 为 2/2 adverse；历史 seed0 方向一致，但不增加 formal replication 数。

支持的句子只有：“在 direct50 P2 的固定 SAR relation/output 条件下，raw paired RGB feature
source 比 matched SAR source 差。”它不是方法，不证明所有 RGB KD 有害，也不能直接解释 exact400。

## 4. 历史方法与工程状态

### 4.1 clean exact400 历史行

- legacy-LADD：机制身份错误，只能称 random-decomposition diagnostic。
- identity：只能称 projected-feature KD identity control。
- LCSR-v1 seed42 endpoint 0.49469，相对 plain SAR 约 +0.00629；低于既定门，且 gate/shortcut/语义
  不闭合，当前版本 KILL。
- FGD runtime 移除了 global relation，只能称 `FGD-focal-only`。
- LD 与 CMDistill 只能称 YOLO11 cross-modal adaptations，不是论文精确复现。
- FGD、LD、CMDistill 的 best/last checkpoint CRC 均失败；此前 ledger 漏标 FGD。AP 轨迹可作历史
  context，但 checkpoint-dependent 后续分析不得使用。
- component runs checkpoint PASS，但多数只有 seed42，不能升级为稳定机制。

这些行即使 outer hyperparameters 相近，也因身份、checkpoint、单 seed 或旧 runtime 问题，不能组成
“新方法稳定提升”表。

### 4.2 direct50/100 与 proxy 路线

DCR、PG-CMD、TCPAD、PTSA、CATR、RIF、SAFR 等分别在不同短协议或 offline proxy 上运行。已有版本
要么 hard gate 失败、要么只证明工程/相关性、要么没有 detector outcome。失败只杀死对应版本，
但它们都没有可进入当前论文主表的正方法证据。

## 5. MM-ARCS R2A 与 H/A/P/U

- R2A：seed42 五臂均有 results/diagnostics 400 行、args/best/last，checkpoint CRC PASS；lock
  `71835b44...`。
- confirmation：H/A/P/U × seeds{0,42,123} 共 12 个物理格，results/diagnostics 都是 400 行，
  args/best/last 和 checkpoint CRC PASS；lock `6def8613...`。当前 formal inference 只保留 seeds42/123
  的 8 格，seed0 只作历史 context。
- 旧 r1/r2/r3 final-analysis roots 含 zero-byte dose CSV，全部禁止作为 canonical。
- 当前没有 accepted final analyzer；R2A 还存在已披露的 interim outcome access，所以不能称 operator-blind。

因此现在只支持“物理训练完成、分析 DEFER”。不支持 efficacy、RGB attribution、arm 排名或 backup
winner。修 analyzer 不等于救方法；只有新 hash 的 fresh review PASS 和 commit-last bundle 才解锁。

## 6. Registry 审计

204 条物理 run 中只有约 35 条 OGSOD-ish 行、20 个唯一 primary path，主要是旧 snapshot。缺失：

- 六个 exact400 baseline；
- CCLKD、LD、CMDistill 的 canonical physical rows；
- 23-run matched panel；
- 全部 R2A 与 confirmation physical rows。

LCSR 同一路径重复约 8 行，FGD 重复约 3 行；多行仍写 running 但远端已 400 行；status 列甚至被写入
科学句子。`runs.csv` 因此只能当未完成的物理登记，不能单独作为结果 authority。

## 7. OGSOD 当前结论

**支持**：RGB-SAR headroom；direct50 条件下 raw feature-source penalty；exact400 matched panel 中
full RGB-CMD 不胜 H_S；旧版本方法的具体失败边界。

**不支持**：任何新方法已有稳定提升；MM-ARCS 正或负；RGB transferable atom 已被识别；untouched
generalization。

**决策：`DEFER`**。改变决策的最近证据是 accepted、hash-bound、commit-last 的 MM-ARCS analyzer
bundle；即使为正仍只是 contaminated development evidence。
