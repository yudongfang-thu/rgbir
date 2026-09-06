# 判别与定位蒸馏可行性复核（2026-09-07）

> **支持试验有条件的定位分支，LLVIP证据更强，Drone有局部机会；教师自身定位更准不等于能直接监督RGB。** 新CPU再分析得到Drone/LLVIP定位细化候选241/3083与117/672；Drone双方命中对象若原样复制IR框，相对RGB框平均IoU差为−0.04564。未后处理检出不等于不可蒸馏；需要分别确认实例身份、坐标传递、任务优势与学生可学性。未修改已冻结OEv1，未启动新的GPU长训。

## 目的
回应用户提出的对称任务选择：判别不足时学判别、定位不足时学定位；区分可学性、对象身份、坐标对应和教师相对优势。

## 设置
重读2026-09-06真实六baseline、521对开发图像的matched_objects和prediction_records及生成脚本；明确所有新阈值为探索性诊断，不追溯称为预注册。核对现行OEv1源码和原冻结协议，检索相关定位/任务解耦蒸馏原始论文。

## 结果

完整论证与试验草案见[FEASIBILITY_AND_PILOT.md](FEASIBILITY_AND_PILOT.md)。

| 指标 | DroneVehicle | LLVIP |
|---|---:|---:|
| 固定开发图像 / 共同对象 | 200 / 3083 | 200 / 672 |
| 双方命中对象IR相对各自GT的IoU优势均值 | +0.00790 | +0.04197 |
| 双方命中对象中IR框直接放到RGB后的优势均值 | **−0.04564** | +0.04197 |
| 类别正确、conf≥.25、RGB IoU[.5,.75)、IR自身IoU领先≥.1的候选 | **241/3083（7.82%）** | **117/672（17.41%）** |
| 上述候选中原样搬框反而变差 | 30/241 | 0/117 |
| 改用RGB目标坐标优势的事后oracle候选 | 200/3083 | 117/672 |

这些是单seed固定baseline的对象级描述，不是AP或定位KD收益；LLVIP共享/复制标签使自身与目标坐标数值相同，不能独立证明像素配准。重算采用全图一对一预测分配，保护原TP后才分配剩余粗候选；root独立标量IoU复核2523/506行通过。

## 结论

1. 当前OEv1筛选使用冻结RGB reference的宽松粗候选，不是动态学生定位准确才学类别；新定位分支需要独立定义质量条件。
2. “类别/置信度好但定位差”有真实候选。Drone该机会主要在已检出对象的精修，LLVIP比例更大。
3. 没有后NMS检测框仍可有dense候选与GT assignment；完全无可靠身份或可用几何时，第一版跳过额外定位KD、保留原生GT监督。
4. 对象身份和坐标对齐是两件事。教师在IR自身GT上高IoU不保证它在RGB坐标更准，不能简单复制IR框或同grid DFL KL。
5. 文献已有分类/定位任务分离及质量权重的明确先例；增加定位loss本身不能算第二原创点。具体可研究的是按实例、任务及坐标可传递性选择或拒绝监督。

## 产物路径
- [DATA_FEASIBILITY.md](DATA_FEASIBILITY.md)：分子分母、候选与几何统计、阈值敏感性、匹配局限。
- [reanalyze_localization.py](reanalyze_localization.py)、[summary.json](summary.json)、两数据集`*_localization_objects.csv`：可复现CPU结果。
- [verify_localization_reanalysis.py](verify_localization_reanalysis.py)、[ROOT_RECHECK.json](ROOT_RECHECK.json)：独立标量IoU复核及目标坐标oracle描述。
- [CODE_FEASIBILITY.md](CODE_FEASIBILITY.md)：现行源码、候选/assignment、可微框解码、坐标及GT-only边界。
- [PRIOR_ART.md](PRIOR_ART.md)：LD/TBD/CoLD/GaLD/CrossKD/AR-CNN六篇原始文献与新颖性边界。
- [FEASIBILITY_AND_PILOT.md](FEASIBILITY_AND_PILOT.md)：综合论证与尚未冻结的最小试验路线。

源证据在`../2026-09-06_probe_RGBIR数据特性与可迁移知识/`；大文件仍在原94路径，本次没有服务器写入。未访问official test、未动原始数据与权重。

## 局限与下一步
既有探针为固定baseline和开发集抽样，保存后NMS预测，不含raw anchor/DFL；不能当作当前动态学生、训练集候选覆盖率或新分支效果证据。新定位训练参数尚未冻结，后续先做训练raw候选/身份/几何覆盖与梯度canary，再比较OEv1、OEv1+定位和同mask的GT定位重权；完整结论仍需三seed与same-modal/shuffled/random控制。
