# 单批DFL原值生产协议（新结果前冻结）

执行上一条 `RAW_DFL_INFORMATION_PLAN.md`，仅原LLVIP自然首批32图/80GT的一次新零更新前向。旧本批raw DFL缓存缺失已由root的有界清点确认。旧机会、C/L2选择、native框及anchor名单全部固定，不重跑NMS/匹配/旧机会分析。

S与R各导出历史R候选anchor、各自历史native IoU.5 anchor。T导出历史R同索引、历史native T anchor，以及历史L2确已selected的独立T anchor。角色缺失保留null；不替代为其他点。按model/image/anchor去重分布，同时保留每GT每角色ownGT语义，T始终使用IR ownGT。

独立源码审阅后、真实前向前明确“相同索引”的范围：T同时保留 `same_S_native_iou50_index` 与 `same_R_native_iou50_index` 两角色，覆盖S/R native点与历史R点不同的原对象；实际同索引分布仍去重，缺失native仍null。不改变任何anchor选择或GT语义。

原运行driver的loader/GT tracing、初始化、S train+BN冻结、R/T eval、AMP及零更新检查复用。只在本批实际前向前注册模型forward计数，要求S/R/T各一次；不统计setup内部dummy。前后核学生与R/T所有参数/buffer exact、无grad、optimizer state空且更新0。新first_batch_stream和完整GT identity_contract必须与旧probe exact；明确这不是历史同一次forward，也未保存/比较历史像素张量。

输出真实raw logits四边×16bin，无损转为JSON float；另存原dtype。FP32 softmax按原OEv1全张量原维度计算、保留完整概率，并核期望解码与同次原 `_decode_boxes` exact。通过当前pinned Detect._inference(existing_raw)原头解码，hook原DFL conv输入取得实际native概率，hook原DFL输出取得实际native距离；重放原decode_bboxes及strides核native框exact。native概率期望与native conv/AMP舍入差单列，不用FP32框冒称native。仅重复head decode，无额外backbone。

每分布记录模型、frame、anchor、level/stride/center、bins/rawdtype、两套概率/期望、原native与FP32框。每object-role保存ownGT及未clamp LTRB/stride，严格0<=d<15才是相邻bin可插值范围；不截断GT、不截断/重归一概率、不跨anchor KL/重映射，不添加任何学生loss或梯度。

producer仅保存自身小源副本及依赖path/stat身份，不复制整个runtime。root唯一lease预约并测本路径实际峰值，只有一个batch，无额外full。成功status=RAW_DFL_SINGLE_BATCH_COMPLETED。合成真值先覆盖同均值异分布、布局和距离范围、空/缺anchor、角色去重；实际读出待独立验收，不由非零熵推断教师更可靠或可学。

## Pre-execution setup amendment

Root authorized a probe-specific bypass of generic setup check_amp: reuse cfg AMP only after exact equality to the prior accepted actual_amp and restore the native module binding in finally. No unrelated model download or check_amp forward is performed. The actual requested batch still uses original autocast, with first-batch stream and full GT identity equality checked before its S/R/T forwards. This is not a claim of complete setup equivalence; any stream/identity mismatch fails without changing the batch or relaxing identity. Portable CPU tests cover successful and failed setup restoration.
