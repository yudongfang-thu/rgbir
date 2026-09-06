# Task-Conditional D1/D2 实际机会诊断

> 已完成四组D1/D2及单帧verified诊断：未核验几何时Drone/LLVIP train分别选中1.96%/3.77%的RGB GT；唯一已接纳几何帧的两个对象均不在证据覆盖区域内，实际D2为0，当前不能授权正式L长训。

## 目的
区分对象级最佳预测机会D1与冻结R实际预选anchor机会D2，复用已有特征图、不伪造raw分布。

## 设置
Drone/LLVIP各固定train2048张与development200张；train按真实来源组比例分层，组内均匀；清单在读图前保存。输入640，冻结seed42 T/R，批8；先8张GPU短测测量实际峰值，统一资源lease。亮度78.283为既有低/高亮度代理阈值；附加区间明确次级。无test读取，不据AP挑清单/阈值。

## 结果
完整原始证据已于2026-09-07 03:04收齐，独立重算与bound summary一致。详见[独立结果与局限](INDEPENDENT_RESULTS.md)。

| 划分 | RGB GT | D2基础E | D2选中 | 选中/GT |
|---|---:|---:|---:|---:|
| Drone train | 31931 | 23511 | 626 | 1.96% |
| Drone dev | 3084 | 2177 | 112 | 3.63% |
| LLVIP train | 5592 | 5492 | 211 | 3.77% |
| LLVIP dev | 643 | 559 | 131 | 20.37% |
| LLVIP verified单帧 | 2 | 0 | 0 | 0% |

前四行均为未核验几何的机会统计。selected教师相对R的GT-DFL CE均值降低0.47–0.86；这支持继续验证定位内容，不能替代CL/CGT效果证据。D1与D2选预测方式不同，两集合不嵌套；D2不代表当前学生已检出但定位不足。

## 结论
诊断通过不等于蒸馏收益；geometry_verified=false的D2不能作为正式L准入。

## 产物路径
代码：`03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_task_conditional_v1/diagnose_opportunities.py`。
94：`/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_task_conditional_v1_20260907/d2_{drone,llvip}_{train,val}_attempt1/`及`d2_llvip_verified_attempt1/`。

本目录五个dataset_split子目录保留原始JSONL与gzip、summary、配置及receipt；[独立统计与过滤表](analysis_v1/summary_recomputed.json)、[全部分组CSV](analysis_v1/group_comparison.csv)、[机会和过滤图](analysis_v1/opportunity_and_filters.png)、[DFL内容图](analysis_v1/selected_dfl_content.png)均由`analyze_diagnostics.py`复算。图同时保存PDF。`collect_diagnostics.py`、`compress_evidence.py`与collection_receipts保留采集入口。

15份JSONL共102,254,198字节，gzip共12,890,622字节；全部直接验证解压bytes等于原bytes，见`gzip_evidence_manifest.json`。GitHub发布gzip及小产物，本地保留原文；分析器自动支持仅gzip复算，不制造巨型纯文本diff。

## 局限与下一步
只读冻结模型，不计作新训练；正式teacher内容价值由CL/CGT完整端点判断。训练集实际监督机会明显低于LLVIP开发集；正式增强与几何mask还会改变覆盖率。单帧零覆盖不能推论整个数据集不可蒸馏。每次attempt独立目录、保留失败记录。
