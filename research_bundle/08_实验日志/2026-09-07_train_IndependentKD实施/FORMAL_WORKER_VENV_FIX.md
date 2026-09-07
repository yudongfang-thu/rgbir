# 正式worker解释器符号链接修复

**已修复LOG调度器对虚拟环境入口的错误resolve；94仅CPU验证通过，无GPU初始化。** NEW训练release和科学验收未改动，不需要重跑科学准入。

根报告原 `/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_independent_kd_v2_20260907/formal_C1_gpu5` 的seed42正式启动在导入ultralytics时失败，尚未创建run目录或执行训练batch。失败日志与campaign保留；下一次使用根指定的新 `formal_C1_gpu5_attempt2`，本工具修复任务未启动它。

`formal_campaign.py.initialize` 仅两处将 `Path(args.python).resolve()` 换为 `.absolute()`，分别用于build_jobs的train/eval argv和manifest/coordinator解释器。release/config/脚本自身路径仍可resolve，未扩展其它逻辑。

新增一个CPU用例验证initialize的manifest及六个stage argv都保留venv入口；Windows本机无创建symlink权限，因此用模拟的resolve行为隔离该语义，16/16测试通过（0.246s）。另在94以真实symlink入口执行一次CPU探针，确认真实路径行为，结果见 `formal_worker_venv_cpu_verification.json`。

实际配置入口为 `.../environments/sn6-int8-kd/bin/python`：`sys.executable`保持该入口，`sys.prefix`为sn6-int8-kd，Ultralytics为8.4.115。该入口是symlink，resolve会变成 `.../environments/sn6-otd-r0/bin/python3.10`，解释了此前screen落入错误环境。探针显式设置空CUDA_VISIBLE_DEVICES，`torch.cuda.is_initialized()`为false，exit0。

探针首次通过PowerShell向bash送heredoc时，终止行受换行格式影响被解释成Python标识符PY，打印了环境但以NameError退出；此失败没有GPU操作。改用SSH直接将纯Python标准输入传给指定venv解释器后exit0。未将首次包装失败隐去或视为训练失败。

代码接受由独立C1 reviewer另行复核，本文件只记录作者修改与实际CPU结果。
