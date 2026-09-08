# LLVIP 初始化端点原生复验入口

**入口就绪，未在本任务运行 GPU；它仅复验初始 visible seed42 checkpoint，不创建训练或 FT3 身份。**

[initial_baseline_eval.py](initial_baseline_eval.py) CLI 为 `--reference-dir --config --checkpoint --output`。输入固定为当前 LLVIP full native 配置声明的 model 路径，只允许 `weights/last.pt`，不查看 best；不要求或制造新的 training receipt。继承配置中的 epochs、arm 等训练字段不被用于声明本次训练，回执显式记录仅消费原生评价字段。

固定 dev2406 图/7879 GT、单类；imgsz640/B32/workers4/quantize=None（FP32）、conf.001/iou.7/maxdet300 与当前 direction eval 相同。复用 pinned capture_contract、verify_population 和只读 object capture；实际 loader、GT 数、类名、FP32 kwargs 及评价前后 checkpoint stat 必须闭合。只写小回执、合同、原生对象读出和源码副本，不写 checkpoint、创建 optimizer 或更改七臂矩阵。

成功输出 `initial_evaluation_receipt.json`，status `INITIAL_BASELINE_EVALUATION_COMPLETED`，scope `INITIAL_BASELINE_NATIVE_RECHECK`，endpoint `INITIAL_FIXED_LAST_EMA_RECHECK`。它是新实际读出，不自称已有 LLVIP profile 绑定或被独立接受的端点。

[轻量入口检查](INITIAL_EVAL_ENTRY_CHECK.json)：语法/import、实际 full native 配置和 3 个错误输入反例通过，CLI --help 通过；未导入 torch、读 AP、使用 GPU/SSH 或新算 hash。实际推理由根任务通过已有 globallease 调度，结果不用于修改已冻结七臂计划、参数或门。
