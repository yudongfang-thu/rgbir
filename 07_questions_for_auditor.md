# 提请外部审计员仲裁的七个问题

> 这些是我们内部无法自洽、或需要无利益冲突的第二意见的问题。每个问题附了我们目前的立场与相关证据位置，请审查立场是否成立、给出你的独立判断。

## Q1 · FreqMix 数据集错引结论是否成立
- 立场：历史总结中的 "FreqMix SiXiang +9 AP" 是把 **OGSOD 256px E400 B64** 的实验错标成 SiXiang 对照；同 recipe 下 FreqMix gray/SAR donor 相对 native **−2.6 mAP pp**、相对 H_S −3.3，整体为负结果。
- 证据：`03_freqmix_sources/`（6 臂 results.csv + OGSOD native/H_S 对照 + args.yaml + patch 脚本）。args.yaml 可确认数据路径与训练分辨率。
- 请仲裁：数值复算是否支持"错引+负结果"结论；sign 变体（−5 pp）与 gray/sar 的对称负性是否排除"实现 bug 单独致因"。

## Q2 · HNEWA 配对归因的正确口径
- 立场：历史 "+12.7pp paired 优势" 是 LLVIP seed123 precision 的误引。以独立 eval_records 重算：DroneVehicle paired−shuffled **+0.336±0.289 mAP**（3/3 正，s42 仅 +0.011）、paired−same-modal +0.207（2/3 正）；LLVIP 对应 +0.350±1.231 / +0.091（均 2/3 正）。
- 证据：`04_hnewa_eval_records/*.json`（12+ 文件）。
- 请仲裁：这一量级应解读为"存在小幅配对信号"还是"噪声"；是否需要更多 seeds/更强 baseline 才能下结论。

## Q3 · P2/P3 探针的实现与推断边界
- 立场：P2 的 CKA 未中心化、S_native checkpoint 身份错误（VEDAI 混入）、shuffled 为单一固定 null、FFT 逐通道相减对独立模型无意义——四项导致探针结论作废。P3 昼夜分桶保留为"条件差异观察"，但"负迁移被定位"撤回。
- 证据：`05_code/` 中 p2/p3 脚本副本 + `02_raw_results_dronevehicle/` 三个 summary JSON。
- 请仲裁：若重做特征级探针，什么设计才能合法回答"蒸馏把学生特征拉向教师"与"可蒸馏子空间在哪"。

## Q4 · CGA-KD 三个组件的修复方案评审
- 立场（预注册 + 偏差 D1）：G（几何）应改为**未归约逐样本 DFL 分布损失**；C（门控）应改为**未归约损失上乘 gate 再归一**（当前实现对 PCC/cosine 是空操作）；I（不变分解）因优化器缺陷、判别目标冲突及"模态不可判别≠配对不可判别"的概念问题**降级暂缓**。
- 证据：`05_code/train_cga_kd.py`（当前实现）、`01_audit_20260905/cga_code_review.md`（逐行审查）、`00_project_context/方法预注册_CGA-KD_20260905.md`。
- 请仲裁：①G 的 DFL 分布蒸馏在 ultralytics v8 loss 框架内的正确实现方式；②C 的 gate 放置位置（损失项级/位置级/通道级）哪种最有机制意义；③I 是否值得修复，还是应从方法中移除。

## Q5 · L vs N 净负结果的解读
- 立场：协议匹配 3 seeds 下，现有全量 KD（PCCFD/SLRD/IBCLD）比 no-KD **净负 −0.35 mAP50-95（3/3 seeds）**；昼夜两桶皆负（night AP50 −1.75、day −0.60）。负迁移不随条件集中——教师夜间优势 +11.2 与蒸馏收益脱钩。
- 证据：`02_raw_results_dronevehicle/N_*_metrics_record.json`、`L_*_metrics_record.json`、`native_s42_buckets.json`、`p3_daynight_summary.json`。
- 请仲裁：①此负结果可否直接归因于"特征模仿在异质模态间有害"，还是可能被 PCCFD/SLRD/IBCLD 的具体实现缺陷解释；②若要在论文中引用，还需要补哪些对照。

## Q6 · 下一步最小干预的定义
- 立场：按"共同目标错误分解"（teacher 更准 / student 有观测但漏 / 几何可对应三类）定义最小干预，而非从机制倒推。
- 请仲裁：给定现有证据（Q1–Q5），你认为**信息量最高、成本最低的下一个实验**是什么？请给出臂定义、样本量与判读门槛。

## Q7 · OS-SSL 迁移红外域的实验设计
- 立场：SpaceNet6 OS-SSL 是全项目唯一 3/3 胜全部对照的正例（+3.3±0.9 AP50），但迁移 OGSOD/SiXiang 失败（+0.20/−0.45）。计划把同一冻结流程（manifest 驱动四臂：paired/shuffled/IR-only/native，SSL 10k 步 → 注入骨干 → e200 微调）移植 DroneVehicle，3 seeds。
- 请仲裁：①以 RGB-IR 的配对质量，迁移成功的先验概率评估与理由；②实验设计的漏洞（如 SSL 数据量 17,990 对 vs SpaceNet6 的规模差、增强策略、冻结超参）；③若失败，如何把"迁移边界"写成有价值的结论。
