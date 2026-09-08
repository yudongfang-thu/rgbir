# Drone 完整开发集缓存 CPU 读出

实际输出在 `output_attempt2/`，19.6716071 秒，待独立验收。共1469配对图像、21568对GT；RGB 894与IR 2922未配对GT分别报告，四桶分母不含它们。冻结范围及不存在旧val映射的勘误见上一级 `PLAN_CPU_PAIRING.md`。

8项合成CPU真值通过（CPU_attempt1.json），覆盖五类/异类、最大基数优先而非greedy、双方未配对、全空、原None图像关联分支与重复stem拒绝、原生detector实际排序和严格score>.25边界。它们不读取真实实验缓存。实际完整统计另运行一次有效attempt。

`output_attempt1` 在完成输入校验与新映射生成后，复制源码快照时把导入产生的 `frozen_sources/__pycache__` 目录当文件而失败；尚未进入检测统计。失败回执、原源码 `analyze_drone_dev_attempt1_source.py` 与其已有产物保留。修订仅将源码清单限制为文件；GT配对/检测匹配/四组/阈值未变。有效结果另写output_attempt2，未覆盖旧raw。

运行入口（本机CPU）：

```text
D:/Anaconda/envs/KGJ_proj/python.exe cpu_analysis/test_drone_cpu.py --output NEW_TEST.json
D:/Anaconda/envs/KGJ_proj/python.exe cpu_analysis/analyze_drone_dev.py --n-root N_s42_attempt1 --t-root evidence_1401/T42_full_attempt1 --output NEW_RESULT
```

缺省图像映射通过实际执行原 PairDataset 构造器的 `strong_by_weak=None` 分支产生。远端canonical身份从已接受capture合同查表，不能在Windows上resolve远端路径。新完整映射及来源为 output_attempt2/image_mapping.json 与 image_mapping_provenance.json；不是声称找到了不存在的旧val JSON。

GT配对执行源副本中的 `_match_objects`，同类IoU>=.5、最大基数后最大IoU。native每模态始终在ownGT独立评价；其四组匹配调用已接受的原函数并捕获实际GT/pred关联。源码副本在frozen_sources与输出source_copies内，不计算hash。

summary提供配对四桶、五类及未配对、整体TP/FP/FN、低阈值到>.25转移、.5/.75联合状态。paired_objects为配对分母，all_gt_objects完整记录双方ownGT（无伙伴则partner_correct=null），prediction_matches保留原predID，frames保留空图及实际canvas。真实原生NMS框可越canvas，不另裁剪。没有AP重算、训练选择门推断、GPU、checkpoint加载或KD收益主张。
