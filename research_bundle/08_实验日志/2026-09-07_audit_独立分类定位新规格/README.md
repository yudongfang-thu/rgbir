# 独立分类/定位新规格审阅与下一步计划（2026-09-07）

**结论：采用C0保留、C1与L1独立验证、暂不融合的方向；新增分类首先3seed，定位先证明有限几何审核能达到自然训练流覆盖，再校准/长训。计划补齐了旧校准器不兼容、E_C索引风险、梯度剂量措辞与最终四臂预算。**

## 目的与材料

用户要求阅读 `07_研究分析/RGBIR_Independent_Class_Loc_Formal_Spec_20260907.md` 与同名 `_Package.zip`，结合上一轮native/载体/门控/文献审计制定下一步计划。使用experiment-plan技能，三条独立审阅分别覆盖C1、L1、矩阵；本轮不实施trainer、不启动GPU。

## 本轮实际完成

- MD与ZIP内MD逐字节一致，68094 bytes；8个包内文件，18个测试函数。已静态阅读代码与提供方18 passed回执，**未本地重跑pytest**。检查脚本 `inspect_package.py` 与 `package_review_receipt.json` 已保存，没有执行外部包源码。
- [C1_REVIEW.md](C1_REVIEW.md)：C1 Bernoulli相对证据定义可采用；matched→E_C索引必须明确，前景重叠对象需诊断；C1_y删非目标项不保证共享参数梯度范数降低。
- [L1_REVIEW.md](L1_REVIEW.md)：DFL/GT数学可采用；旧校准器实际eval()且不逐批恢复BN，不满足新合同；当前几何全图象限规则与仅局部6点不同，不能隐式混用。
- [MATRIX_REVIEW.md](MATRIX_REVIEW.md)：N/C0优先复用但新wrapper须兼容；保持现有data generator；12核心实验不含胜出分支四臂，最高情形24次，不应一次铺开。
- 新增只读CPU roster计数：Drone258条只有238个唯一图，LLVIP42个；已有D2基础表交集里含selected的20/4图。未覆盖记录不能判负。LLVIP若只授权42图，简化均匀抽样64×B32命中期望8.36批，提示16/64信号门与有限审核可能不协调，**不是实际流的硬上界**。脚本及JSON见 `l1_geometry_coverage_audit.*`。

## 最终计划

- [完整计划](../../refine-logs/EXPERIMENT_PLAN.md)，不可变初版副本为 `EXPERIMENT_PLAN_20260907_123709.md`。
- [实验跟踪表](../../refine-logs/EXPERIMENT_TRACKER.md)，不可变初版副本为 `EXPERIMENT_TRACKER_20260907_123709.md`。
- 与原材料的显式补充：L准入后将L_GT42前置与L1_42同批；仍无条件收齐技术有效的L1三seed，不拿该单seed改变门/λ。
- 先C1×3；L准入再L1×3+GT42；有价值再C1_y×3+GT0/123。核心最多12；胜出方法四臂额外补齐，先闭合类别赢家。
- 保留L1的strict07窄假设，.80敏感性是探索，不静默改门。先CPU重放固定64批，检查拟审核图像能命中的确切上界；有限点审不能达到覆盖则当日给出BLOCKED_COVERAGE_BUDGET，类别继续。
- 当前不实现局部特征/dense C2/融合/动态路由；OS-SSL、VEDAI仍暂停。原CL/CGT未启动扩展按新计划登记延期，不触碰在跑C0控制。

## 结果与局限

本轮没有新AP。基线采用11:19快照：C0−N=+0.266655±0.144373pp，三个seed同向；仍须四臂归因。旧native与N的0.469pp观察差未因果分解；不据此重训一整套旧基线，也不在新方法中改变增强种子后继续复用旧N。

计划不是已经通过的geometry/calibration/canary回执。未来每次GPU启动须现场核验资源并走共享lease；原始结果、失败attempt和检查点不覆盖。阶段代码/证据按已有授权同步GitHub；**本次计划目前只落盘本地，未发布、未启动新训练。**

## 产物路径

本目录：三份审阅、包清单脚本/JSON、几何计数脚本/JSON；`refine-logs/`：完整计划、tracker及时间戳版本。本轮没有新服务器训练输出目录；依据的94原始结果路径见此前native审计与11:19快照。
