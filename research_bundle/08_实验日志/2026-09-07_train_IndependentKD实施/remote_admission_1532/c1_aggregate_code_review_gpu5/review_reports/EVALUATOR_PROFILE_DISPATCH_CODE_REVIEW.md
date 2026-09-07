# Evaluator profile / companion dispatcher 代码范围审阅

2026-09-07，独立审阅 `/root/review_matrix_spec`，未修改被审源码。已复跑 `python -m unittest test_evaluator_profile test_resource_dispatch -v`，8+28=36项 CPU 测试通过，0.339s。

读核了 evaluator_profile 原生/扩展的同 kwargs、完整 dev1469 名单/实际 loader 顺序/seen、五指标与逐类 AP 精确比较、旧 N42 completion/last 只读及新 probe 输出；还核对 evaluation_profile 独立资源身份、配置/数据/实际源码字节绑定，和 companion canary 的已有一个实际项目 CUDA、全部采样活跃区间两任务、24更新后才允许正式共卡。

**本次静态和合成测试未发现阻塞当时 release4 路径的问题，但不签真实 GPU/AP/峰值接受。** 根随后报告真实 GPU probe 暴露回调 `v.model` 接口问题并正在修复。该实际失败优先于此代码审阅，必须保留失败 attempt、修复后重验；本记录不覆盖修订版或声称 probe 成功。CPU fixture 未执行真实 validator 回调，因此没有涵盖这个实际接口。
