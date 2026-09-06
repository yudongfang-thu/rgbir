# criterion 等价验证 attempt1 类型适配修复

> attempt1 在读取原生 loss items 时失败，尚未产生任何 GPU 等价通过证据；已修复验证器的递归快照/比较，4 项真实结构 CPU fixture 通过，等待 root 启动新 attempt。

## 实际失败

94 `artifacts/rgbir_task_conditional_v1_20260907/criterion_equivalence_s42_attempt1/failure_receipt.json`：`AttributeError: 'dict' object has no attribute 'detach'`，位于验证脚本旧第150行。零 optimizer 更新、未访问测试集。原失败 attempt 与 release_v6 保留。

现场核对 pinned `ultralytics/utils/loss.py`：第357行命名 `box_loss / cls_loss / dfl_loss`，第463行通过 `dict(zip(self.loss_names, loss.detach()))` 生成 items，第486行返回 loss 张量与该 dict。错误在验证器假定 items 是 tensor，不是被测损失代码。

## 修复范围

- 新 `criterion_items.py`：递归 detach/clone tensor 叶子，保留 dict/list/tuple 结构；精确比较键、容器类型、张量 dtype/shape/value，报告逐层差异。
- `verify_criterion_equivalence.py` 仅替换两处 items 快照与两处 items 比较，并在运行 receipt 中保存新 helper 源码。
- 没有修改 legacy criterion、新 TaskCriterion、预测、批次、native 实例、权重或梯度比较规则。
- `TaskCriterion` 是普通 Python 对象；现有模型状态检查使用 `nn.Module.state_dict()` 的注册张量，临时 criterion 的统计字段不伪装为模型参数状态。状态快照仍在唯一学生 forward 之后取得。

## 实际验收

94 CPU 运行 `test_criterion_items.py`，4项通过：真实三项标量Tensor字典、嵌套dict/list/tuple、键/形状/dtype/容器不等、快照不改变原预测计算图及梯度。环境 Torch2.10.0+cu128、Python3.10.20，`CUDA_VISIBLE_DEVICES=''`，`torch.cuda.is_initialized()==False`。

本机默认 `D:/Anaconda/python.exe` 没有 torch，因此本机 unittest 未执行成功；有效测试证据是94 CPU结果，见 `criterion_items_cpu_fix_receipt.json`。此前CPU语法检查不能替代此测试或GPU等价检查。

## 下一步

新 release 必须同时包含 `verify_criterion_equivalence.py` 与 `criterion_items.py`，由 root 通过统一 lease 调度新的完整B32 attempt。此次修复不自动升级 `legacy_C_N_equivalence`。
