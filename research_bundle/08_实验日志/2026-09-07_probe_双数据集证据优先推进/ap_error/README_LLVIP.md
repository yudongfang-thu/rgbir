# LLVIP完整dev AP错误诊断（有效attempt2，CPU）

> **独立审阅PASS**：[审阅报告](llvip_independent_review/EXPERIMENT_AUDIT.md)验证两模型全量GT、明确相对路径配对、TIDE oracle与COCO AP。完整full只运行capture；64图canary才是双路径对照。

> 在六类main error中，旧visible baseline的AP50/AP75均以定位oracle贡献最大；AP50仍有较大的漏检与背景贡献。IR检测器的较高AP与定位oracle仅构成待验证的知识线索，不能视为可达KD收益或L1准入。

## 目的、输入和模型身份

使用完整2406图/7879个person GT的低阈值post-NMS预测，补充先前200dev对象子集分析。只读取根任务认可的 `../llvip_full_eval/remote_completed_attempt2/{N42,T42}_full_attempt1/`；内部full_attempt1是有效第二轮campaign的任务编号，不是先前GT丢失的无效campaign。

模型为旧历史visible/infrared seed42 native E200 last.pt，训练batch32、imgsz640、SGD、lr0=.01、weight_decay=.0005、AMP、workers8。评估FP32。两份加载回执为checkpoint_epoch=-1、ema_present=false、loaded_source=model（已strip的模型）；不伪称另有可检查的EMA字段。**N42是旧baseline标签，不是新协议N，也不是LLVIP三seed KD结果。**

|代号|模态|checkpoint|
|---|---|---|
|LLVIP_N42|visible|/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbt_p3_causal_v1/formal_native/llvip/visible_seed42_native_b32a2/weights/last.pt|
|LLVIP_T42|infrared|/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbt_p3_causal_v1/formal_native/llvip/infrared_seed42_native_b32a2/weights/last.pt|

## 评价设置与覆盖

原预测conf=.001、NMS IoU=.7、multi-label、max_det300、FP32、native rect，实际画布544×672。GT及预测为该画布中的绝对xyxy；适配器仅改为xywh，不缩放、不裁剪。官方TIDE正IoU=.50/.75、背景IoU=.10、max_dets300，全部GT保留，无ignore/crowd元数据。

逐图processed alias通过各自contract的显式alias_to_canonical双射与canonical roster匹配；期望、loader、捕获三个GT计数均为7879。两模态根路径不同；本入口只分别核完整性，不按basename擅自建立跨模态配对。`summary.json`的same_gt_population=false表示保存的image路径不相同，不能解读为两个数据集GT数不一致。跨模态配对和标签归属由根任务另行核验。

|端点|图|GT|输出框|空预测图|无GT图|最大单图框数|达到300框图数|
|---|---:|---:|---:|---:|---:|---:|---:|
|LLVIP_N42|2406|7879|28406|1|0|62|0|
|LLVIP_T42|2406|7879|12929|0|0|27|0|

两端点均无max_det额外截断。原conf=.001之下的响应仍不可见，因此“完整”指固定后处理协议的全部输出。

## 原pinned指标与TIDE指标（均为百分数）

|端点|pinned AP50|pinned AP75|pinned mAP50–95|TIDE AP50|TIDE AP75|
|---|---:|---:|---:|---:|---:|
|LLVIP_N42|71.613960|24.282631|32.878405|71.427586|24.537259|
|LLVIP_T42|92.612756|44.076592|48.852941|92.229627|44.135239|

pinned使用IoU候选优先去重与插值梯形积分；TIDE使用分数优先匹配和101点recall均值。二者不混称同口径。本报告只解释AP50/AP75的TIDE错误，没有重定义原pinned mAP50–95。

## 官方独立oracle贡献（dAP，百分点）

|端点|IoU|Cls|Loc|Both|Dupe|Bkg|Miss|special FP|special FN|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|LLVIP_N42|0.5|0.000000|8.754009|0.000000|0.254040|5.533965|7.684950|10.750632|15.043994|
|LLVIP_N42|0.75|0.000000|56.373148|0.000000|0.001133|1.131363|2.524979|17.046900|34.023572|
|LLVIP_T42|0.5|0.000000|3.615429|0.000000|0.146799|0.641480|2.751826|1.829779|5.487398|
|LLVIP_T42|0.75|0.000000|51.892331|0.000000|0.001079|0.211308|1.350379|16.260800|28.646785|

这里“定位最大”限定六类main error。special FP/FN覆盖范围不同，不能放在同一互斥类别排序中；例如N AP50 special FN=15.0440pp，大于Loc=8.7540pp。独立oracle不能相加；Miss/FN还改变漏检分母，不等于新生成可学出的正确预测。

LLVIP只有person一个前景类，因此Cls/Both为结构性的0；人/背景区分仍体现在Bkg、Miss及分数排序。不能将零Cls解释为“分类任务已完美”或将其与Drone五类Cls数值直接排序。

## 全低阈值匹配计数与分数区间

|端点|IoU|TP|FP|FN|score≥.25 TP|score≥.25 FP|其中Bkg|
|---|---:|---:|---:|---:|---:|---:|---:|
|LLVIP_N42|0.5|6513|21893|1366|5591|1996|1268|
|LLVIP_N42|0.75|3262|25144|4617|2917|4670|1268|
|LLVIP_T42|0.5|7469|5460|410|6991|514|190|
|LLVIP_T42|0.75|4771|8158|3108|4555|2950|190|

score≥.25计数沿用同一低阈值TIDE分数优先匹配后筛分，不重新运行NMS或挑阈值；不是AP分母或独立最佳F1指标。全部固定分数区间和分位数在端点JSON内。

## 发现与局限

1. N从AP50到AP75明显下降，同时Loc dAP由8.7540升至56.3731；在该评价下，高IoU定位质量是明确瓶颈。IR也有51.8923pp的AP75 Loc残余，不能将IR定位当成完美目标。
2. N的AP50还受Miss7.6850、Bkg5.5340pp影响；单类检测前景/背景质量与覆盖仍须检验，不能因为定位优先就忽略它们。
3. 旧IR模型AP较高且错误结构不同，结合既有对象/DFL证据可支持继续检查定位知识的可用性；本次未评估目标几何认证、当前学生可学性或新KD训练收益。
4. 两模型只有seed42，输入模态与训练身份不同；没有匹配新协议N三seed和四臂控制。AP差不是蒸馏收益，无效attempt也不是模型AP0。
5. 本次只读dev，不触碰官方test、训练参数或准入；原L1几何合同保持不变。

## 验证与复核入口

官方TIDE与独立NumPy IoU/score-greedy/AP101对两模型×两个IoU逐类AP、GT/TP/FN完全交叉一致（容差1e-9pp）；已知真值错误类别、空图、score ties、单调分数变换及alias绑定测试均通过。独立NumPy核验不证明oracle可达性。

- `llvip_results_v1/summary.json`、`LLVIP_N42.json`、`LLVIP_T42.json`、`tide_endpoints.csv`：原始统计。
- `llvip_results_v1/runner_source.py`、`summary_runner_source.py`、`LLVIP_PROTOCOL.md`：本次运行源和协议副本。
- `llvip_results_v1/source_binding_stats.json`及`LLVIP_*_source/`：来源path/size/mtime与原population/model_identity/metrics/summary字节副本；未新增hash。
- `llvip_results_v1/synthetic_validation.json`：真值检查；`llvip_manifest.json`：输入完整绑定。
- CPU入口：`run_tide_audit.py --manifest llvip_manifest.json --output <new_directory>`。官方TIDE源commit49a5d2a4aeb56795e93a3ed7cc7e6d25757bb4c1，来源为[作者仓库](https://github.com/dbolya/tide)及[原论文](https://arxiv.org/abs/2008.08115)。
