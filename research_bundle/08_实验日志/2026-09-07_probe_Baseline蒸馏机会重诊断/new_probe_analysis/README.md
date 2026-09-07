# CPU baseline 额外信息分析器（2026-09-07）

**已完成两数据集真实分析；结论见[KEY_FINDINGS.md](KEY_FINDINGS.md)。Drone固定ridge对少数类明显欠拟合，IR特征未稳定胜过独立RGB，不能认定feature载体优越；LLVIP的对象/DFL定位目标线索更一致。分析器23项真值测试和独立5项反例通过。**

## 目的

比较 RGB baseline、IR teacher 与独立 RGB baseline 中可以解码的类别、定位及局部特征信息。对象检测机会、线性读出和DFL头部方向分别报告，不混成AP或蒸馏训练增益。

## 固定设置

- 每行RGB GT，另有固定annotation-background行；对象机会和DFL全部排除背景。对象关联 paired_gt_iou≥0.5，检测正确为conf≥0.25、类别正确、RGB坐标IoU≥0.5。所有RGB GT保留在修复/损伤分母中。
- train拟合、val评价；ridge目标为均值one-hot平方误差＋α‖W‖²，α=1，截距不正则化。每个模型/层以固定seed20260907高斯投影128维，投影后逐列只拟合train均值/标准差。P3/P4及联合层全列示，无dev调参或选层。
- 固定3×3 anchor patch→2×2池化特征为前景/背景主读出。GT宽高驱动ROI为辅助，只使用所有已导出模型×P3/P4共同regions.valid集合，保留同集合N_logits基线；无有效性字段时ROI读出停用。两类cohort不跨分母直接作差。
- 教师shuffle分别在每split内固定置换；GT-ROI置换只在共同有效集合，anchor置换覆盖其完整集合。各层共用置换，未按类别置换。N0同模态臂仅在有真实导出时出现。
- metadata-only的输入为log(ROI宽、高、stride)，揭示GT/背景采样或ROI尺度混杂，不视为视觉信息优势。
- 在查看真实dev输出前固定加入教师logits/feature载体对照：N_logits+T_logits、N_logits+N_anchor_feature+T_logits以及共同有效ROI的N_region_logits+T_region_logits，对应比较同cohort的T feature输入；N0 logits/region对照在导出存在时列示。输入维数和有效正则不同，差异不是KD载体带宽的因果效果。
- 按类别、source_group、scale_bin、train拟合的亮度三分位、anchor候选/GT中心回退及对象配对状态分别列示。前景macro recall和背景recall单独报告。
- DFL需两模型同一RGB参考anchor/stride/输入画布，四边GT距离在[0,14.99]，对齐pinned native的reg_max−1−0.01，不裁剪；数学原语允许精确15的边界测试，但该值不进入native主诊断。GTCE用T=1，教师→native KL用T=2乘T²，比较raw native DFL-logit梯度cosine。主表要求paired＋有native粗候选，GT中心回退另表。

## 产物与运行

`analyze_baseline_probe.py`只依赖Python/NumPy，不执行SSH、不使用GPU、不改trainer、不计算hash。输入为原始导出目录：objects.jsonl、features.npz、logits.npz、summary.json及可用model_identity.json/frozen_roster.json。若存在metadata.json则优先读取。原始文件仅只读，输出目录必须尚不存在，失败尝试不得覆盖重跑。

```powershell
python analyze_baseline_probe.py --input '原始数据目录' --output '新的分析结果目录'
python -m unittest test_analyze_baseline_probe -v
```

每个结果目录含README.md、summary.json、probe_predictions.npz、probe_train_transforms.npz和dfl_per_object.csv。JSON保留所有类/组分母、输入身份、原始文件路径与字节数，无hash。真值测试回执保存为synthetic_test_run_v*.txt；合成测试输出使用临时目录，不纳入正式实验证据。

## 局限

固定patch避免使用GT宽高定义局部区域，但选anchor仍有GT关联特权，fallback更直接使用GT中心。背景排除使用两模态标注，只是annotation-background代理。主分类表中的未配对GT位置不能称同对象IR信息，paired_gt子组才限定标签对象身份；即使已配对也不证明像素配准。probe评价时输入IR信息是双模态读出，不代表RGB-only推理或KD增益。头部局部梯度不等于骨干/共享特征参数更新方向，也不满足冻结L1几何准入。

## 下一步

真实结果已写入dronevehicle_analysis_v1与llvip_analysis_v1。另有direct_head_baseline_v1自然head sanity，未改冻结probe。正式研究主张仍需匹配native、至少三seed和四臂归因；本脚本只提供诊断线索。
