# LLVIP完整开发集旧baseline诊断

> **独立审阅通过**：[实际全量回执复核](../ap_error/llvip_independent_review/EXPERIMENT_AUDIT.md)限定接受旧baseline dev结果；四个TIDE AP另经独立COCO重算，最大差1.42e−14pp。

> 有效attempt2完成：2406图/7879个GT，旧RGB与IR baseline的mAP为32.8784/48.8529；模态差距为15.9745pp，不是KD增益，也不是新协议N结果。

## 设置与验收

固定既有grouped split dev，旧visible/infrared seed42 E200 last checkpoint；文件身份、原args、源码及数据配置均保存在采集目录。使用pinned环境、batch32/workers4、640、FP32、rect、conf=.001、NMS IoU=.7、max_det300。实际输入为544×672，原图1024×1280；不与640方形诊断直接混为同一推理协议。

每模型固定前64图分别运行原生与捕获评估，全部指标exact后才运行2406图capture。完整评估未额外重复原生前向，summary中full的native_capture_exact=false表示未运行该比较。原始标签逐图与loader的类别及归一化框array_equal，三处GT计数一致。两个模态按登记的processed模态根下唯一相对路径显式配对，逐图GT、原图尺寸、canvas尺寸exact；此项不证明物理像素配准。

## 结果

|旧baseline|mAP50–95|AP50|AP75|
|---|---:|---:|---:|
|RGB seed42|32.878405|71.613960|24.282631|
|IR seed42|48.852941|92.612756|44.076592|
|IR−RGB，pp|15.974536|20.998796|19.793960|

以上来自pinned评价器；TIDE使用不同匹配/插值口径，见[独立错误分析](../ap_error/README_LLVIP.md)，不能互相替换。只有一个旧baseline seed，结论限定为诊断优先级。IR本身高IoU定位仍有明显缺口。

## 故障与资源

第一次导出把processed软链接展开成raw图像路径，YOLO无法推导processed标签路径，产生0 GT/0 AP；该attempt已标INVALID，远端原始证据保留。attempt2保留processed图路径并逐图核对标签后全部重跑，详见[故障说明](ATTEMPT1_INVALID.md)、[代码复核](INDEPENDENT_CODE_REVIEW_ATTEMPT2.md)。另独立完成检查器首次把canary的全量label-count列表当64图列表，断言失败；修正为分别核对full_gt和expected_evaluated_gt，未涉及评估产物修改。

有效attempt2的四个任务均通过既有global lease在GPU4共享运行；NVML实测每任务峰值1370MiB，任务进程树RSS最大4103MiB；整卡最低剩余14662MiB，项目RSS监测最大149020MiB。未新增长训、未改原C1/控制任务。完整任务预约使用本路径canary实测峰值加余量，未借用训练任务峰值。

## 复核入口

- [实际完成核验](completed_verification_receipt.json)、[核验脚本](verify_completed.py)、[显式配对清单](verified_pair_manifest.jsonl.gz)。
- [采集回执](collection_receipt.json)与`remote_completed_attempt2`包含canary、full、源代码、模型身份、配置、资源/完成回执及全部小体积预测。
- 94：`/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_evidence_priority_20260907/llvip_full_eval_attempt2`。
- 新协议N/L1/L_GT尚未准入；[配置准备](../llvip_preparation/README.md)单列其阻塞，旧baseline不能代替新N。
