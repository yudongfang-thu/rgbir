# 冻结的 Drone 完整 dev CPU 配对读出

scope=`DRONE_FULL_DEV_CACHED_NATIVE_PAIRED_OPPORTUNITIES`。仅复用完整开发集 N42 与 T42 的 native post-NMS 低阈值缓存；不读test、不加载checkpoint、不运行推理/NMS/训练、不计算hash，也不根据新AP调方法。实际 IR 完整缓存到达、CPU小真值接受之后才运行完整统计。

N输入固定为 `2026-09-07_train_IndependentKD实施/legacy_diagnostics_snapshot_20260907_161459/N_s42_attempt1` 的 predictions/objects.jsonl.gz、evaluation_contract/evaluation_val/reevaluation_receipt；模型是 full_weight0_s42_attempt1，不是旧RGB reference。T输入为本条目即将收集的 `remote_attempt1/T42_full_attempt1`（实际下载目录由root提供CLI），对应远端 `artifacts/rgbir_drone_teacher_capture_20260908/attempt1/T42_full_attempt1`；只有completed/full1469/IR24490/固定IR42及完整contract被接受。N有22462 GT、T有24490独立GT；禁止要求GT数组相等。

图像关联必须来自原工程明确开发集映射：`94:/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbt_p3_causal_v1/prepared/dronevehicle/mappings/rgb_to_infrared_val.json`，当前待收集，不替代为train映射或新stem猜测。原 tools/prepare_rgbt_experiment.py 的 prepared_selection_mapping按两模态val相同stem集合生成。原 `legacy_oev1/paired_rgbir_data.py` 对显式mapping两端canonicalize、核对teacher索引完整。分析将原mapping每个key/value分别经已记录alias→canonical映射绑定到两份完整native实际roster，要求双向1469唯一覆盖；未知/重复/缺失即拒绝。

同帧两模态必须拥有相同original shape和native canvas shape，均以各自缓存的native canvas xyxy GT直接关联，不猜坐标缩放；若不一致则停在缺口，另定协议。图像相同尺寸不证明物理配准。

GT关联严格执行原 `legacy_oev1/object_evidence_loss.py` 的 `_iou` 与 `_match_objects`，同类且IoU>=.5。实际是 eligible边加 `min(Ngt,Tgt)+1` 的基数奖金再 Hungarian：先最大配对数量，再最大总IoU；不是普通greedy，也不是原生detector的GT/pred匹配。保留原RGB/IR GT行序、各自框/类/ID、配对IoU和两个未配对集合。关联只算一次，四组沿用同一固定GT关联。源副本与原文件逐字节比对，不算hash。

检测四组固定为原conf>.001低阈值NMS缓存 / 原缓存保序过滤score>.25，各自独立执行native IoU=.5/.75匹配。真实 DetectionValidator._process_batch/BaseValidator.match_predictions 复用已接受的 native_cached_match；捕获原函数返回时的实际GT/pred配对并核对完整TP位，不能换近似greedy。原预测ID、过滤后ID和对应own-GT均保存，不再次排序，不裁剪合法越canvas预测框。

T检测正确性始终相对IR ownGT；仅对已配对GT将这个布尔映回其RGB伙伴。四桶为双方正确/仅T正确/仅N正确/双方未匹配，**分母仅配对GT数**。所有RGB和IR未配对GT另外按每组正确/未匹配、五类、图像数报告；它们没有对侧正确性，值为null，不能补false或塞进四桶。全模态TP/FP/FN以各自完整GT分母报告，与配对桶分母分开。

输出所有五类（car/freight car/truck/bus/van）的配对和未配对计数、四桶；score>.25的N@.5/N@.75/T@.5/T@.75共16格联合表，以及低阈值→>.25四桶转移。缺失分母比例为null，不假设高IoU正确GT一定为低IoU子集。全空图仍在frame manifest；原pred和GT空数组保留。

CLI预定 `analyze_drone_dev.py --n-root ... --t-root ... --image-mapping ... --output NEW`；CPU测试覆盖五类、异类拒绝、重叠竞争/最大基数、双方未配对、全空、原生匹配顺序、>.25边界与错误映射拒绝。统计只说明成熟N/T在该dev与既定关联中的检测互补，不是KD可达收益、IR几何正确性、当前学生训练选择覆盖或单seed方法增益。所有输出新目录、原raw不改。

## 输入勘误及原规则分支冻结（完整CPU结果之前）

root在94实际读取上述预期旧val JSON得到FileNotFoundError；该路径仅由现行准备器源码推导，不能声称旧文件已存在或已绑定。root已明确授权执行原 DualLabelRGBIRDataset.__init__ 的 `strong_by_weak=None` 分支。分析使用原完整源码的私有模块，构造只承载已接受两模态alias/canonical清单的CPU载体；不读图片、不执行变换。原 `_canonical` 的远端解析事实由capture合同逐项查表提供，禁止Windows本地resolve远端路径；除此之外原构造器分支不改。两模态canonical唯一、canonical stem唯一、1469项、双射/无复用/全覆盖均硬检。原teacher唯一stem分支查找得到mapping，随后按原显式mapping的两端canonical规则再次闭合到完整dev清单。输出 image_mapping.json + image_mapping_provenance.json，明确为本次执行原fallback生成的新manifest，非不存在的旧文件。两模态各自完整root严格限制由其capture合同保障。

真实IR已由root确认 `evidence_1401/T42_full_attempt1` 及queue完成；仅在CPU小例通过后读取其输入身份与完整缓存。CLI省略 `--image-mapping` 即执行此已授权原分支；传显式文件仍须相同完整identity核对。此前输入路径待落实的原文保留作为勘误前记录，不改变四组、GT关联、分母或结果口径。
