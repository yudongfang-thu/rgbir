# LLVIP 成熟初始化的完整开发集参照

**已有原生完整 dev 读出可作为本次三臂的成熟初始化参照：mAP50–95 原值为 `0.3287840510939928`（32.87840510939928 AP）。本次只核来源，不重新评估或读取权重。**

| 指标 | 原始 fraction |
|---|---:|
| mAP50–95 | 0.3287840510939928 |
| AP50 | 0.7161396006544973 |
| AP75 | 0.24282631410771383 |
| precision | 0.7463069993744104 |
| recall | 0.7019331948389218 |

实际 [initial_evaluation_receipt.json](initial_evaluation_receipt.json) 的 status 为 `INITIAL_BASELINE_EVALUATION_COMPLETED`，scope 为 `INITIAL_BASELINE_NATIVE_RECHECK`，endpoint 为 `INITIAL_FIXED_LAST_EMA_RECHECK`。该评估独立读取原 visible seed42 成熟 `weights/last.pt`，没有 FT 训练预算，不复用本轮任一臂结果。它不是论文增益或新 FT3 训练完成回执。

原远端产物位于：

```
/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_direction_screen_20260908/initial_llvip_recheck_attempt1/evaluation
```

checkpoint 为 `/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbt_p3_causal_v1/formal_native/llvip/visible_seed42_native_b32a2/weights/last.pt`，bytes=5482010，mtime_ns=1788127857140185407。本次核出它与先前真实 DFL forward 的 S/R 初始化 stat 完全相同，三个新 L3 配置的 S/R 路径也相同；本轮正式初始化实际回执仍须事后逐臂核对。

评估为 LLVIP 完整 dev 2406 图/7879 GT，单类 person。torch `2.10.0+cu128`、ultralytics `8.4.115`；640/B32/workers4、FP32 `quantize=None`、conf=.001、NMS IoU=.7、max_det300、rect=True、augment=False、half=False。新 L3 evaluator 的固定有效参数与本次初始评估完全相同；新 `llvip_native_evaluation.yaml` 与当时实际保存配置逐字节相同。新结果的 actual profile 与实际 roster 仍必须再核，不能只凭模板宣布已匹配。

[initial_evaluation_contract.json](initial_evaluation_contract.json) 与 [development_roster.txt](development_roster.txt) 的 canonical roster 逐项相同；实际 rect loader 的顺序可能不同，但同为 2406 个唯一图且集合相同。loader 前与 capture 后的 GT 数均为 7879，person 类名和 receipt/contract checkpoint stat 一致。

本次仅从 94 已知完成目录只读补收 4 个小文件：完成回执、source_manifest、当时 requested native config、实际保存的 [initial_baseline_eval.py 源副本](remote_source/0000_initial_baseline_eval.py)。[远端收取记录](REMOTE_COLLECTION.json)保留来源 stat 与传输字节一致；原先本地三份证据另有[本地复制记录](LOCAL_COPY_RECEIPT.json)。源代码在原生 metric 返回后记录数值，保持评价前后 checkpoint stat 不变，没有 optimizer 或权重写入。

[verify_initial_reference.py](verify_initial_reference.py) 的只读身份复核已通过，见 [REFERENCE_IDENTITY.json](REFERENCE_IDENTITY.json)。原回执 `accepted_endpoint_claim=false` 如实保留：本次核验不伪装为重新从 raw predictions 复算 AP，也不新造原始评估的独立 accepted 身份。可在新三臂完成并匹配上述实际口径后计算各臂减初始化的差；不得用它替代本轮 N 对照。没有新 GPU、forward、权重读取或 hash。
