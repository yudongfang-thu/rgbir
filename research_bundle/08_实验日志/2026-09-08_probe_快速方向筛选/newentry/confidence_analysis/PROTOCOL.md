# LLVIP 置信度两臂回执分析：新AP前冻结

只比较本轮 `LLVIP_CONFIDENCE_FT3` / `LLVIP_CONFIDENCE_FT3_LAST_EMA` 的N与C0：λ0/.1，共同visible42初始化、固定2048训练图、完整dev2406图/7879GT、单类person、seed42、独立3轮last/EMA。原L2的N即使初始化相同，也不能替换本轮N。

输入固定为 `evaluations/{N,C0}/direction_evaluation_receipt.json`；两份成功回执未齐、存在failure冲突或身份不符时拒绝出表，不填零。检查新scope/endpoint、原始metric fraction、单类macro AP一致性、共同训练子集三路径/初始化路径/完整dev YAML、实际BN证据和固定图数。回执中的这些绑定由实际训练/评价入口验证；本分析器不重新加载checkpoint、GT或图像。

保留N/C0全部原始fraction，显示百分制mAP50–95/AP50/AP75/precision/recall，固定计算C0−N的pp差与正负方向。不作SD/显著性、内容归因、跨协议比较、自动挑臂或E200准入。

```sh
python analyze_confidence.py --campaign <本轮收集目录> --output <尚不存在的新读出目录>
python test_confidence_analysis_cpu.py --receipt <新CPU回执.json>
```

输出仅summary.json、README.md和执行源码byte-exact副本。4个小真值覆盖原值/pp符号、旧N及身份拒绝、fraction/person/BN保护、固定路径与失败冲突。未读取真实新AP、未GPU、未计算新hash。
