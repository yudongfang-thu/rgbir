# F-rel-GM：校准后、任何 F AP 前的独立协议修订

这是明确的新探索协议 `FEATURE_RELATION_GM_FT3`，endpoint=`FEATURE_RELATION_GM_FT3_LAST_EMA`。原 F-rel 因 λ>1 被阻塞的回执原样保留，复制件为 `original_blocked_calibration_receipt.json`；不把原候选改为准入，不宣称本修订先于已经完成的校准，也不称正式准入。

只准备 F-rel-GM 一份配置：`configs/drone_F-rel-GM_s42_FT3.yaml`。固定 λ=14.438521129817886，来自已完成8批的目标 C1 加权梯度范数/单位 F 梯度范数比值中位数，不新校准、不扫参、不修改损失单位。原 F-rel 3×3 ROI、channel normalize eps=1e-6、72 off-diagonal cosine-Gram MSE、P3/P4共同有效层平均、原 selected/base 分母完全不变；`direction_losses.py` 与原源逐字节相等。

## 梯度剂量读数

`CALIBRATION_RATIO_READOUT.json` 保存原8批的 native norm、单位 B∇C1/B∇F norm、目标 BλC1∇C1 norm、固定系数后的 BλF∇F norm及比值表；均为原 calibration 小记录的纯计算，未读取 F AP。

- C1 固定系数为 0.09227393550836771。
- 原逐批目标/单位 F 系数比：6.283914、16.254979、13.099523、29.428592、8.588623、12.742012、16.843059、15.777520；其中位数为固定的 14.438521129817886。
- 在此固定系数下，实际 F/C1 加权梯度 norm 比最小 **0.4906290218**，中位数 **1.0086749286**，最大 **2.2976954163**，8批均有限非零。

偶数样本的比值中位数与倒数比值中位数不互逆，所以不写“实际比值中位数严格=1”，也不为此改 λ。安全/剂量依据是这组实测梯度与新 canary 的数值有限性和资源峰值；不同 loss 的数字 λ≤1 不提供普遍的同剂量约束。八批范数匹配不能保证后续每批同剂量或方向有益。

## 主 N 的明确跨 scope 对照投影

模型、T/R、2048 Drone subset/mapping、seed42、B32/nbs64/workers4、640、增强/evidence、SGD/AMP、lr=1e-4 constant、warmup0、BN running freeze、3轮192批均保持主 direction 匹配 N 的配置。这里的 N 分支代码保留原行为；仅新 arm 的 `request` 显式映射到既有 `F-rel`。CPU 使用实际旧/新 `make_type` 函数和明确 toy API/native 依赖，确认 N 的 loss、parameter gradient、selection 调用与诊断 exact；真实整流身份还须由新 F canary 与主 N 的前30批、完整 warmstart initialization stat 核对。

只复用主 direction 的完整 N，不把 N 回执改为新 scope。统一队列另落 `matched_control_projection`，闭合主 N full completion 和两边配置、首30批、initial checkpoint stat；评估回执标记 projection_required=true。若这些不匹配，不把现有 N 当作可比对照。`feature_gm_config.yaml` 和完整训练回执副本会随 eval 保存，供分析器检查 recipe。

## 入口与接口

- Canary：`train_feature_gm.py --reference-dir PINNED --config CFG --output NEW_CANARY --canary`，至少24次成功更新并有 selected 非零 KD 梯度；测真实资源峰值。
- 训练：`train_feature_gm.py --reference-dir PINNED --config CFG --output NEW_RUN --canary-receipt CANARY/canary.json`，严格3轮192批。
- 评估：`evaluate_feature_gm.py --reference-dir PINNED --config CFG --checkpoint RUN/weights/last.pt --output NEW_EVAL --native-config NATIVE_CONFIG --native-profile-binding ACCEPTED_DRONE_BINDING`，原 FP32/640/B32/workers4/full1469图22462 GT。

训练保存 `feature_gm_config.yaml`；`canary.json` status 保留通用 `DIRECTION_CANARY_COMPLETED`，`completion_receipt.json` status 保留 `DIRECTION_TRAINING_COMPLETED`，均严格新 scope/endpoint。失败为 `feature_gm_failure.json` / `FEATURE_GM_TRAINING_FAILED`。eval 保存通用 `direction_evaluation_receipt.json` / `DIRECTION_EVALUATION_COMPLETED` 或 `direction_evaluation_failure.json` / `DIRECTION_EVALUATION_FAILED`，scope/endpoint 为新协议，完整 AP fraction/逐类、training_subset_identity、expected_train_images=2048 与主对照 scope 声明保留。

## 已完成与限制

`CPU_CHECKS_attempt1.json` 六组通过：算子字节一致与唯一 arm alias；实际 N 函数的合成 loss/梯度/选择等价；完整原 recipe；固定系数与错误 scope/L 拒绝；原 BLOCKED 校准及实际比值；eval 完整 dev 身份和源码编译。此处仅源码与 CPU 准备，无 GPU、无新 hash、无 F AP。根队列等待置信度队列成功结束后，使用原 globallease 安排新 F canary、训练、评估，不由本文件自动准入或升级正式多 seed 证据。
