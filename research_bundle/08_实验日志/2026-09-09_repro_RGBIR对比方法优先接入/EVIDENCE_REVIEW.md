# 90 复现接入证据范围审阅（2026-09-09）

**三项狭义完成事实有回执支持：CFT 作者 checkpoint 经显式兼容适配后完成3对 fit 图像前向/NMS；BCDL 分类算子通过6项CPU检查；CMD端口通过原22项CPU测试及`--help`。没有新AP、完整训练或author-exact复现证据。** 总体记为 `warn`，表示必须保留范围与兼容性限定，不是否认已执行的CPU结果。

本审阅由复用代理 `/root/port90_protocol_assets` 完成，其之前参与LLVIP资料检索、准备脚本和CFT静态核查；不是fresh-context审计，也不声称跨模型审阅。此次仅读本地原回执、日志与源码，不SSH、不跑模型或测试、不解包重算预测、不计算新hash。遵用户明确nohash要求，替代技能默认摘要/rehash步骤；文件身份依赖路径、原来源引用及现有回执，不提供内容不可变证明。结构化记录见 [EVIDENCE_REVIEW.json](EVIDENCE_REVIEW.json)。

| 项目 | 已核证据与可支持结论 | 不可支持的扩大结论 |
|---|---|---|
| CFT | [attempt2回执](cft_forward90/cft_forward_attempt2/receipt.json)、三个inference JSON和raw输出文件存在；[作者来源](cft_forward90/source_downloaded.json)为`fb591c9b163177c0e950db08e213e24ddc912d41`，[下载回执](cft_forward90/weight_downloaded.json)为413453231字节。CPU4线程、Torch2.10；020001/100118/180408三对fit图，实际两流输入均`[1,3,832,1024]`，原预测`[1,52416,6]`且回执有限性为true，NMS分别2/1/2框。支持**PROTOCOL-ADAPTED作者权重执行恢复**。 | 不能称历史训练代码数值等价、author-exact、论文表复现、速度benchmark、精度正确性或RGB-only KD；未评价AP，也没有dev/test比较。 |
| BCDL | [90算子回执](existing_assets/bckd_operator/CPU_RECEIPT_90_attempt1.json)为6/6 PASS；读过[检查程序](existing_assets/bckd_operator/check_operator_cpu.py)、[提取函数](existing_assets/bckd_operator/extracted_novel_kd_loss.py)及[独立表达式](existing_assets/bckd_operator/bcdl_independent.py)。作者`novel_kd_loss`函数体AST保留，移除两个MMCV装饰器，在小合成张量检查损失、学生梯度、教师detach、beta/实际温度及错误替代式。 | 不是完整BCKD，不覆盖MMCV weighted reduction、检测头样本选择、回归蒸馏、真实教师、两数据集训练或指标。`independent`仅指算式表达不同，不是独立实验或独立审稿人。 |
| CMDistill-corrected | [90接线回执](existing_assets/cmd90_port/cmd_cpu_receipt_attempt1.json)两命令returncode=0；[测试日志](existing_assets/cmd90_port/cmd_tests_attempt1.log)22 passed in 1.68s；[help日志](existing_assets/cmd90_port/cmd_help_attempt1.log)显示必填`--teacher-weights`；[合并来源](existing_assets/cmd90_port/cmd_source_merged.json)明确新源目录、PROTOCOL-ADAPTED、teacher TO_BIND。原测试使用合成张量/假教师。 | 只证明CPU导入、CLI和被覆盖算式/接口测试通过；没有真实IR教师绑定、真实数据训练接线准入、GPU可运行/显存/时长保证，也无新AP。两数据集模板存在不代表已运行两数据集。 |

## CFT 的适配及失败保留

[attempt1失败](cft_forward90/cft_forward_attempt1/failure.json)明确为`TransformerBlock`缺失`conv`，原失败保留。成功attempt2由[新包装](run_cft_author_cpu_v2.py)调用[cft_checkpoint_compat.py](cft_checkpoint_compat.py)：仅将子模块名匹配`ln_input/ln_output/sa/mlp`且处于`.trans_blocks.`的旧`TransformerBlock`重新绑定为当前作者`myTransformerBlock`。回执列出24个块，记录参数/缓冲对象未变、未创建随机参数、未改原仓库源码。

该检查支持“显式适配后跑通”。参数对象不变不能推出forward与不可得历史训练代码数值等价；原`attempt_load`还按作者入口执行FP32与fuse，不能将兼容函数内的对象不变断言扩大为整个加载过程完全未变。此次只核适配调用路径和现有兼容回执，类内部结构的进一步独立核对由并行Drone代理负责，不预先宣称其结论通过。

三个stem与本项目冻结fit名单一致，非dev前缀01/04/07/12/25；本次实际读取真实图像但未使用GT计算精度。因此类型为真实输入上的机械执行检查，**不是real_gt检测评价**。作者official train若覆盖本项目fit+dev，其checkpoint见过我方dev，后续dev推理不能作为未见数据泛化比较。融合RGB+IR结果不进入RGB-only KD胜负表。

## 一项归档警告与未审范围

`cft_forward90/AUTHOR_ARCHIVE_RECEIPT.json`实际内容是`M2D-LIF_master_author.zip`及`Zhao-Tian-yi/M2D-LIF`来源，属于混放的另一方法源码清单；**排除为CFT来源证据**。不改、不覆盖、不移动原件。CFT来源使用正确的`source_downloaded.json`与`weight_downloaded.json`即可，故此定位问题不推翻attempt2执行事实。

没有检查远端checkpoint完整内容、历史训练源、图像/标签版本逐字节一致性或全量预测。raw NPZ只核存在，未重新载入计算；结果数字以已保存JSON及实际命令日志为依据。此审阅没有统计检验、多seed、归因四臂、accepted full-dev analyzer，也不是全论文接受或正式训练准入。
