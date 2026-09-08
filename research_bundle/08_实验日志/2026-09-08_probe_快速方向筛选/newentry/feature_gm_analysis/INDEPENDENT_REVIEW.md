在固定三臂、单 seed FT3 的描述性回执读出范围内，当前 F-rel-GM 分析器可接受；独立复跑现有 4 项 CPU 真值全部通过，未读取新的 F AP。

审阅源码 `analyze_feature_gm.py`、合成测试及实际队列/evaluator 字段。fraction 到百分数及 pp 差值使用原浮点值；三项 AP 均检查五类宏均值，缺失/失败/错误 scope/非有限数/单位非法时拒绝。N、C1 与 F 的固定 λ 分别为 0、0.09227393550836771、14.438521129817886。共同配置、BN、完整 dev 与独立训练身份有明确检查，SD 为 null，无自动扩展或长期增益结论。

互审中补齐了候选 projection 与 F 实际 `training_configuration` 的 exact 路径绑定及配置副本字节数，原作者 attempt1 留存；controls N/C1 各自绑定配置、训练与评价回执。正式 collect 还要求新 F 队列成功、无失败记录，以及 F 完整训练前 30 批与 canary exact 的实际回执。该 projection 来自新队列对 N/C1 canary、完整训练、共同初始化 stat 与配置的逐项检查，并非仅凭同数据集默认接受。

读出只支持校准后、首次 F AP 前修订的新 `FEATURE_RELATION_GM_FT3`；旧 F-rel BLOCKED 保留。跨 scope 单 seed FT3 差值不能升级 E200、显著性、模态因果归因或原始协议全程 outcome-blind 声明。本次没有 GPU、SSH、权重加载或新 hash。独立 CPU 回执为 `INDEPENDENT_CPU.json`。
