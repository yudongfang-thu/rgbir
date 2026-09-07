# 原生/扩展评价探针与双开资源验证入口

结论：evaluator_profile.py 和相应 dispatcher 支持已实现，8 项评价探针 CPU 测试与 28 项资源测试独立运行，**36/36 通过**。这里只交付代码和可运行入口，没有执行 GPU 评价、没有产生真实 AP 等价或双开通过结论。

## 固定 N42 评价探针

唯一基线输入为已完成的旧 weight0/N seed42 E200 fixed last。程序只读其 completion_receipt.json 和 weights/last.pt，核对身份后，在新 artifact 目录依次运行原生 DetectionValidator 与 make_evidence_validator(DetectionValidator)。不调用新训练 load_run_configuration，不创建任何训练完成回执，不向旧 run 内写文件。

固定读取 Drone 的完整 1469 张 dev；拒绝 test YAML/路径与不足/重复名单。两个调用使用相同 YOLO checkpoint、相同 kwargs，并分别重置本独立探针的随机种子。每边从实际 validator 获取 loader 顺序、seen、实际参数和量化精度。扩展一侧还核验 objects.jsonl.gz 的唯一完整样本集合。

通过条件为 AP50、AP75、mAP50–95、precision、recall 的浮点结果精确相等，逐类 AP 也相等，实际 kwargs 与 loader 顺序相同。没有根据新方法 AP 改阈值或选择通过样本。

新输出包含 native/evidence 的 metric/contract、真实对象捕获、旧 N42 completion 的只读副本、dev 名单、实际评价器/原生指标/NMS/loader 源码副本、资源回执和最终 evaluator_profile_receipt.json。NVML 峰值与完整进程树 RSS 来自原共享 guard，dispatcher 同时监控整卡和项目资源。资源 profile 不冒充 accepted analyzer 的性能主张，也不自动替代旧结果评价桥接。

## 独立评价资源身份

dispatcher 新增 stage=evaluation_profile，可在没有历史评价 profile 时进行受限 bootstrap：16,000 MiB 预约和硬上限、没有其他本项目任务、允许余量足够的其他用户共享，所有显存/RSS/GPU数量规则继续有效。

后续 stage=evaluation 使用相同 profile_key 和 requires_profile 指向上述 dispatcher 的资源 JSON。身份来自实际配置中影响评价的字段、dev1469 当前名单、数据 YAML 字节、实际评价器和 loader 源文件字节；不使用哈希、不信任人为写入的 profile_identity。训练的 arm/类别系数不同不会改变评价资源身份，但模型基型、类别数、imgsz、batch、workers、软件版本或代码/data/roster 变化会拒绝复用。

eval profile 使用独立 binding，不走 C1/L1 校准的训练 execution binding。它只为 Drone 当前评价器配置提供资源依据，LLVIP 要有自己的实际评价 profile。

## 真正双开 canary

require_project_companion=true 仅允许已具同路径资源 profile、非 bootstrap 的 canary。候选 GPU 必须恰有一个本项目实际 CUDA 任务且仅一个有效预约 slot；仍满足所有显存及主机内存余量。

每秒样本新增整卡 used/free、项目 actual VRAM 和 actual CUDA 任务数。本任务自身已有 CUDA 后，全部采样区间都必须观察到两个项目 CUDA 任务，且 canary 实际完成至少 24 次 optimizer update，资源 profile 才写 companion_measurement.verified=true。若伴随任务提前结束导致条件不足，保留真实测量但不宣称双开通过。

仅有单测 profile 的正式训练会选择没有其他本项目任务的卡，其他用户共享仍按余量允许；已有匹配双开 canary profile 才允许与另一个本项目正式训练共卡。没有足够双开余量不阻塞合法单开。采样证明限于实际测量时段，不宣称连续每个微秒都被 NVML 观测。

## 入口

评价探针命令：python evaluator_profile.py --config <Drone配置> --checkpoint <旧N42/weights/last.pt> --output <全新artifact目录>。

dispatcher job 使用 stage=evaluation_profile、kind=eval、bootstrap_profile=true、vram_mib=16000、rss_mib=32768，result_receipt 指向新输出下 evaluator_profile_receipt.json。未来 evaluation 使用同 profile_key 与 requires_profile。

伴随短测使用已有 C1 单测资源 profile、相同 C1 lambda 与计算路径，stage=canary、require_project_companion=true。未生成或复制 C1_y 正式矩阵。

## 文件与审阅范围

新增 evaluator_profile.py、test_evaluator_profile.py；修改 resource_dispatch.py、test_resource_dispatch.py。未修改 C/cal/compat、criterion 或 evaluate_independent.py。本作者不自签独立接受，由根执行者与其他审阅者审核后部署/启动。源码副本在 evaluator_resource_probe_implementation_v1/；所有 CPU 夹具明确标为合成，测试通过不表示 GPU 已运行。

