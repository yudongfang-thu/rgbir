# 开发检查失败记录

- 首轮 17 项 CPU 测试中 16 项通过，`test_original_unchanged` 误将整个 unittest 对象（含输出流）深复制，报 `TypeError: cannot pickle '_io.TextIOWrapper' object`。已改为仅复制七份实际输入数据，第二轮 17 项通过；未改变分析逻辑。
- 首轮六端点真实接入在读第一个端点时退出：`ValueError: Observed identity differs: seed`。原生执行回执 seed 位于顶层，其他方法字段位于 inputs；适配器原先错误地统一从 inputs 读取。已按实际 schema 修复，保留双向 seed 核验；未产生首轮分析结果文件、未修改原始证据、未启动 GPU。
