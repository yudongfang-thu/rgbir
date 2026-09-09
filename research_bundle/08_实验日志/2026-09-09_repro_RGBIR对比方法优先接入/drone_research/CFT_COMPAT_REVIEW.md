# CFT checkpoint 类绑定兼容审阅（2026-09-09）

结论：**当前 `cft_checkpoint_compat.py` 的条件 `__class__` 重绑适配合理，未发现错误结构匹配或偏离当前作者 forward。**这支持恢复已存参数的执行路径，不证明恢复了未公开的 checkpoint 训练源码，也不证明与该未知训练源码数值等价。

## 历史核对

直接读取作者 [a3e9f5b 的 common.py](https://raw.githubusercontent.com/DocF/multispectral-object-detection/a3e9f5b/models/common.py)、[fdffff4 的 common.py](https://raw.githubusercontent.com/DocF/multispectral-object-detection/fdffff4/models/common.py)，与本地 `llvip_research/CFT_source/models/common.py` 做 Python AST 比较；去掉注释、docstring 和源码行号，不导入或执行作者模块，不计算 hash。

| 类 | a3e9f5b → 当前 | fdffff4 → 当前 |
|---|---|---|
| TransformerBlock | 完整类 AST 相同 | 完整类 AST 相同 |
| myTransformerBlock | 完整类 AST 相同 | 完整类 AST 相同 |
| SelfAttention | 完整类 AST 相同 | 完整类 AST 相同 |
| GPT | 完整类 AST 相同 | 完整类 AST 相同 |

两个公开版本**均已同时包含两个不同的类**，不能从这两个 commit 声称“仅发生重命名”：

- `TransformerBlock`（本地第 70 行）：标准视觉 Transformer，使用 `conv/linear/tr/c2`。
- `myTransformerBlock`（本地第 516 行）：CFT 融合块，注册 `ln_input/ln_output/sa/mlp`。forward 先执行 `x + sa(ln_input(x))`，再执行 `x + mlp(ln_output(x))`，返回相同 token 结构。
- `GPT`（本地第 549 行）构造的 `trans_blocks` 使用 `myTransformerBlock`。SelfAttention 及 GPT 本身的公开历史 forward 也没有 AST 差异。

checkpoint 中融合块被 pickle 为 `models.common.TransformerBlock`，而其存储子模块是另一种融合结构；公开类名和 checkpoint 类名不一致已经有明确结构证据。确切保存时源码仍不可得。

## 对当前小适配的审阅

审阅对象：[cft_checkpoint_compat.py](E:/SHARE/光sar/08_实验日志/2026-09-09_repro_RGBIR对比方法优先接入/cft_checkpoint_compat.py)。实现仅对以下条件同时成立的实例执行 `module.__class__ = myTransformerBlock`：

1. `type(module) is TransformerBlock`，且注册子模块集合恰好为 `ln_input/ln_output/sa/mlp`；
2. 实例路径包含 `.trans_blocks.`，无 `conv/linear/tr/c2` 属性，本级无直接 parameters/buffers；
3. 重绑前后整个模型的参数名→对象 id、buffer 名→对象 id 完全相同；未找到可修复实例时明确失败。

重绑不调用 `__init__`、不重新创建层、不初始化参数、不进行宽松 `load_state_dict`，因此直接复用原有 LayerNorm、SelfAttention、MLP、参数和 buffer；执行的是 AST 已核验的当前作者 CFT forward。正常标准 Transformer 的子模块签名不同，会保持原类，不应全局把 `models.common.TransformerBlock` 设为别名。路径检查是 `.trans_blocks.` 子串限制，不是额外的父类类型判定；对本次已知 GPT 结构和精确子模块签名，没有发现由此导致的错误匹配。

本次适配无需为修复缺 `conv` 而添加 `conv=None`，那样会继续落入错误的标准 Transformer forward。保留现有精确重绑方案即可。参数/buffer 对象不变的断言仅覆盖重绑这一操作；作者 `attempt_load` 已有的 `.float().fuse().eval()` 是前置加载流程，不能把此断言扩展成“整个加载流程从未改变参数表达”。

## 验证边界

本审阅只读本地适配与作者历史代码；未 SSH、未运行 checkpoint、未修改主任务运行文件。真实 CPU forward 成败以主任务 v2 执行回执为准。本结论应标记为“作者 checkpoint 的显式运行时兼容恢复”，保持 `PROTOCOL-ADAPTED` 和“未知原训练源码不作数值等价声明”的现有表述。
