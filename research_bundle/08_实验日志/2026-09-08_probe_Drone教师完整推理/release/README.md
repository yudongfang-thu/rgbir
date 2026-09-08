# Drone IR42 native capture 接口

独立旧 IR42 推理，不是新训练端点。仅 root 经原 globallease 启动；producer 本身不调度。

```text
PINNED_PY export_drone_teacher.py --spec drone_teacher_spec.json --model T42 --output NEW_CANARY --canary
PINNED_PY export_drone_teacher.py --spec drone_teacher_spec.json --model T42 --output NEW_FULL
PINNED_PY test_capture_cpu.py --output NEW_CPU_RECEIPT.json
```

首32图按原 processed IR val alias 字典序 00001–00032 固定，514 GT，类计数 447/24/30/13/0。因此 canary 实际 AP 类为0–3；模型/数据必须始终五类 car/freight car/truck/bus/van。完整1469图24490 IR GT、2空GT图、类计数20588/918/1470/789/725。不沿用 RGB 22462 GT，不将两模态独立标签数组视为相同，不做跨模态GT配对。

固定 `FP32 / quantize=None / half=False / imgsz640 / B32 / workers4 / rect=True / conf=.001 / NMS IoU=.7 / max_det300 / agnostic=False / single_cls=False / augment=False / val`。原图W640/H512；捕获原生实际canvas，不重设框坐标。保留全部GT和预测（包括空数组），原捕获发生于native update_metrics之后。阈值是低阈值完整 native NMS 缓存，不是先截断 .25。

canary 用两个 fresh YOLO 的原生 evaluator 与同 helper capture evaluator 验证所有总体及出现类指标 exact、effective kwargs/loader顺序 exact；完整dev只运行 capture一次。使用旧完整E200 IR42 last，通过 YOLO 标准 EMA/model加载路径并记录实际来源。模型仅由远端正式推理加载，准备与CPU测试不读checkpoint；不写权重。原 checkpoint path/stat前后检查，正式训练args的200/42/IR数据和模型类序核对。

成功 `summary.json`：status=`completed`，scope=`DRONE_IR42_FULL_DEV_CAPTURE`，dataset=`dronevehicle`，model=`T42`，images=32或1469，gt_count=514或24490，full_population=1469，full_gt=24490，canary布尔，native_capture_exact仅canary为true，metrics fraction，resources为原legacy全局lease资源结构，gpu_allocated/reserved_peak_mib、seconds。另有 `capture/objects.jsonl.gz`、`capture_contract.json`、`capture_metrics.json`、`model_identity.json`、`population.json`、源码副本；canary另存native contract/metrics。native_capture_exact=false的full表示未重复原生对照，不表示canary失败。异常在已建立attempt后写 `failure_receipt.json`；前置输入拒绝由外部queue记录。输出已存在则拒绝，不自动重试。

不调用原formal evaluate run/publish/binding；仅复用 `make_evidence_validator/capture_contract/verify_population`、`dev_roster/metric_record/assert_equal_metrics`。未计算新hash。CPU truth仅输入/指标合同，不能代替真实native canary或峰值实测。实现由上一级 accepted_llvip_export_full_dev_source.py 独立派生；原文件与旧attempt未改。
