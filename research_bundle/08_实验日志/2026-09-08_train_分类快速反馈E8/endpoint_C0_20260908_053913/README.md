# C0／seed42 E8 完整端点

**C0完成独立E8日程，完整dev mAP50–95=43.920584；限定回执复核通过。这里仅报告C0原值，待C1完成后再统一形成三臂比较，不作正式蒸馏增益判断。**

## 设置和完成身份

Drone、seed42、原通用YOLO11n初始化；共同8轮与8轮LR日程、warmup3、640、batch32/nbs64/workers4，原增强及SGD配置。原C0判别分支、实际分类系数0.1、定位系数0；未使用selected-only候选。配置、实际日志和原冻结来源的核对见 [ENDPOINT_REVIEW.md](ENDPOINT_REVIEW.md)。

固定E8 last/EMA。4504批（8×563）、2667次成功optimizer更新、7次AMP跳步，2674次attempt和EMA计数。训练总耗时1939.103秒（32.318分钟）；CSV累计1928.220秒是另一计时口径，2–8轮每轮约237–240秒。当前共享负载与N运行时段不同，不能把耗时差解释为方法速度优势。

## 完整dev独立评价

1469图、22462个GT，原pinned native口径；评价耗时15.685秒。JSON原值单位为0–1，下表乘100。

|指标|百分制数值|
|---|---:|
|mAP50–95|43.920584|
|AP50|65.071726|
|AP75|51.113261|
|Precision|66.400280|
|Recall|64.271871|

|类别|mAP50–95|AP50|AP75|
|---|---:|---:|---:|
|car|62.198344|92.657154|71.690398|
|freight car|22.732459|38.033400|23.759022|
|truck|40.864936|58.977444|48.444923|
|bus|66.132742|91.497675|79.789760|
|van|27.674440|44.192958|31.882201|

单seed、短日程，不报告跨seed SD，不与旧E200直接相减，也不解除C0原有REVIEW_REQUIRED状态或触发新增长训。保持现有C1日程与系数。

## 证据与局限

训练实测NVML7630MiB/RSS28884MiB，评估1370/4027MiB；两阶段queue均完成且无资源错误。18项限定交叉核对通过，源为实际配置、日志、训练/评价/资源回执，未重新训练、评价、加载大权重或新计算hash。

训练`results.csv`表头与前7轮15列、末轮8列，末尾三个数为LR；禁用训练内评价的零值及末行错位均不能当AP。唯一AP来源是 `evaluations/C0/short_evaluation_receipt.json`，独立评价合同及checkpoint stat已闭合，原CSV保留不改。

`collection_receipt.json`记录94原路径、stat、已收小产物和只留远端的大文件。`runs/C0/`、`evaluations/C0/`、`queue/`分别保存训练、独立评价、阶段完成及资源依据；审阅与简单复算见 `ENDPOINT_REVIEW.md`、`endpoint_review_receipt.json` 和 `review_receipts.py`。
