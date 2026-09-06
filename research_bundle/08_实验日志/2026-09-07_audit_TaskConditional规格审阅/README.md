# Task-Conditional KD 实施规格审阅（2026-09-07）

> **审阅完成：总体可采用，D1/D2拆分和C冻结合理，附录DFL内核14项CPU检查通过；实施前需补可执行几何契约、冻结R与原生owner冲突处理及GT-DFL温度定义。**建议接入最新D1证据并将同mask GT控制前置到pilot。原规格、原OEv1及在跑训练未改，没有新GPU实验。

## 目的与范围
检查研究前提、对象/anchor选择、定位分布与GT控制、CPU内核、工程接口和新增文献身份。原文件：`07_研究分析/RGBIR_Task_Conditional_KD_Codex_Spec_20260907.md`，现场大小81033字节，DRAFT-v1。

## 结果

完整总评：[SPEC_REVIEW.md](SPEC_REVIEW.md)。

| 核查面 | 判断 |
|---|---|
| 研究与归因 | H1/H2/H3、D1/D2、C冻结、L/CL对照逻辑成立；选点与几何机会仍需实测 |
| 几何 | verified布尔字段尚缺残差单位/容差/适用对象与区域范围；DFL-first有条件成立 |
| anchor归属 | 冻结R唯一owner仍可能与动态native TAL owner冲突；需记录same-object/background/other-object，不能暗加S依赖mask |
| GT-DFL | 非整数距离的两bin GT概率若直接配T>1学生softmax，会改变原生最优分布；对照温度必须明确 |
| DFL内核 | 本地PyTorch1.8，14项CPU检查通过；现代CPU autocast不可用，未验收GPU AMP/真实trainer |
| 数据衔接 | 最新241/200/117统计提供D1性质证据，定义不完全等同于新规格D1，更不能直接当D2 |
| 新增文献 | R1/R2/R8/R9均找到一手来源，主要概括成立；publication全文/verification checkpoint限制仍须保留 |

CPU合成反例也说明：期望框/距离更准不保证全DFL分布对GT监督更有益。这是方法的可证伪风险，不是附录KL实现错误；支持把GT控制和梯度诊断提前。

## 产物与局限

- [SPEC_REVIEW.md](SPEC_REVIEW.md)：root综合判断、原文行号、具体补充建议。
- [SCIENCE_REVIEW.md](SCIENCE_REVIEW.md)：独立科学/协议审阅。
- [KERNEL_REVIEW.md](KERNEL_REVIEW.md)、[kernel_validation_local.json](kernel_validation_local.json)：原样内核与14项CPU验证、反例。
- [verbatim_appendix_a.py](verbatim_appendix_a.py)、[verify_reference_kernel.py](verify_reference_kernel.py)：原文内核副本和可复核脚本。
- [CITATION_CHECK.md](CITATION_CHECK.md)：四篇新增一手来源核查及机制边界。

原规格共1296行，未改写。没有现场GPU诊断/完整trainer实现/新定位结果；CPU环境不同于原文自报2.10.0+cpu，不能称其完整环境已重现。设计选择、待补契约和可复现的数值陷阱分开报告，不把未实现模块说成已发现训练bug。
