# 技术执行证据绑定实现

**实现了实际配置、三个数据小文件、定位实测输入、引用名单和实际加载源码的阶段绑定；22项本地CPU测试通过，L实现agent独立复核并接受该辅助工具范围。回执见`evidence_binding_review_v2/review_receipt.json`。该工具不签发训练许可，不产生GPU结果；根agent已接入真实receipt/admission。**

## 产物与接口

- `03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_independent_kd_v2/evidence_bindings.py`
- 同目录`test_admission.py`

```python
binding_path = capture_execution_binding(output, cfg, stage, HERE)
verified_binding = validate_execution_binding(binding_path, cfg, stage, HERE)
```

stage为`compatibility`、`calibration`、`canary`。capture必须在真实阶段成功、懒加载计算路径已经运行后调用；目录为`output/execution_binding_<stage>/`，排他创建，不能覆盖旧证据。实际completion/calibration/compatibility receipt应保存返回路径为`execution_binding`。验证器返回已核验的binding或抛异常，没有`ACCEPTED`快捷开关。

## 绑定范围

保存完整原始`effective_configuration`、按阶段规范化的`normalized_configuration`、实际arm/source/family、原路径和三个小文件的原字节副本：student YAML、privileged YAML、paired train mapping。原YAML和路径保持原样；不删除或改写test字段，实际train-only检查仍由执行入口负责。

源码从当时`sys.modules`中读取：位于当前release根目录、具有实际`.py`文件的所有模块，包括`__main__`。最低清单必须包含当前stage入口、runtime、train_independent、independent_criterion、evidence_bindings及冻结legacy trainer/双标签loader。记录模块名、原路径和字节副本。验证时逐个核对记录的实际依赖，不要求formal进程和旧校准进程加载列表完全相同；没有被加载的评价脚本后续变更不会使已测兼容无效。

独立审阅发现仅保留geometry/D2路径不足以证明校准时使用的门，已补：calibration/canary对非空`geometry_contract`、`d2_receipt`和实际为JSON/YAML文件的`source_geometry_scope`保存并核验原字节；同路径内容修改会使旧校准/canary失效。描述性`source_geometry_scope`仍按配置字符串绑定。

两份原数据YAML中train/val引用的`.txt`名单也保存原字节并核验；不遍历test条目。目录型split本版没有新增图像/标签内容认证，仍依赖既有数据治理及执行器的实际完整roster检查。不得把三个小文件字节一致扩大表述为整个数据集逐字节不变。

拒绝缺少关键源码、重复条目、源码/数据改变、路径逃出绑定目录、用原文件自身冒充副本、stage和arm重命名等情况。此机制没有hash，也不绑定大权重的文件内容；模型路径身份和生命周期仍由其他检查负责。

## 配置规则

共同忽略seed、method_id、description、readiness_receipt、protocol_status、formal_training_authorized、calibration_receipt、canary_acceptance及两个KD系数字段。被忽略的值仍完整保存，供admission独立核验。

| 阶段 | 额外允许的差异 | 仍严格绑定 |
|---|---|---|
| compatibility | arm/source及classification/localization/geometry/D2/evaluation_contract/calibration新分支字段 | native recipe、数据路径、模型、T/R、evidence门、全部增强 |
| calibration | 同family的arm；C1 family的classification.off_target_weight | source、温度、clip、evidence、增强、定位参数/几何及其他配置 |
| canary | 无 | arm/source与除共同忽略项外的全部配置 |

compatibility capture必须实际为paired N/C0；calibration capture必须实际为C1或L1。C1可服务C1_y，L1可服务L_GT，但跨family被拒绝。

## admission接入要求

在读取兼容、canary、校准receipt后调用相应stage验证，现有成功更新数、64批、非零梯度、几何覆盖、独立接受及资源检查继续保留。尤其：

1. 兼容receipt的seed/arm必须与binding完整配置一致，而formal目标seed/arm按预定共享规则比较。
2. canary receipt的arm/source、T/R、两个系数必须既等于binding完整配置，又等于formal cfg；否则手改receipt系数可能绕过单独的λ检查。
3. 校准receipt的实际family需与binding相同；已测`lambda_C1/lambda_L1`等于formal系数。校准输入系数可能为None，不要求校准前None等于输出λ。
4. binding只证明提交的文件仍对应已记录配置/源码；实际成功执行、独立复核和信任链由对应真实receipt及admission承担。

## 验证

本地环境`D:/Anaconda/envs/KGJ_proj/python.exe`，无Torch/SSH/GPU。执行：

```text
python -m unittest -v test_admission
Ran 19 tests in 1.230s
OK
```

测试用显式synthetic临时文件和模块夹具，不把夹具当作真实训练成功。覆盖错增强、错C门、错source/arm、只允许预定C1_y载体差异、L_GT共享、跨family拒绝、旧N/C0共享规则、同路径数据变化、已加载源码变化与未加载eval允许变化、`__main__`、关键源码缺失、原副本同指、不可覆盖和配置无副作用。λ测试明确证明binding保留实际原系数供admission比较，**不宣称binding自身核验了λ**。

后续需要对根agent集成后的admission做端到端检查；本次没有修改admission.py或trainer/calibrator。

补强后执行`python -m unittest -q test_admission`：`Ran 22 tests in 1.686s / OK`。新增3项涵盖calibration/canary各三种辅助文件同路径变更、缺少几何或aux记录、两模态train/val四份文本名单变更。测试夹具为合成数据，不是GPU准入证据。
