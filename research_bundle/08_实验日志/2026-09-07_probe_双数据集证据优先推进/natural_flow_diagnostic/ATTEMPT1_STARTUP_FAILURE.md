# 第一次真实canary启动失败

`natural_flow_attempt1`在首个数据batch前失败：runtime把legacy/reference模块目录放到sys.path前方，它们也含prepare_configs.py；简单按模块名导入了旧配置生成器，访问llvip_C1报KeyError。

该attempt未产生真实自然流结果，没有训练或参数更新。原部署源码、独立CPU审阅、screen日志、lease与失败状态保存在本地remote_failed_attempt1和原94目录，均不覆盖。该故障说明CPU的选择器真值测试未覆盖真实模块加载顺序。

修复仅按明确的release_gpu5/prepare_configs.py文件路径加载独立v2生成器，使用唯一模块名，不改输入配置、采样、增强、教师、参考、门控或统计。另新建natural_flow_attempt2，仍从固定流首批运行各数据集2批canary和64批完整诊断。

首个失败短测的GPU1由既有global lease动态分配；启动检查记录项目占卡为[1,2,4,5]、使用后完全空卡为[6,7]，满足仍余≥2张空卡，明确援引AGENTS.md §2.1第四卡放宽条款。原始admission和资源记录保留。后续每次启动仍重新检查，不据这次分配硬编码GPU。
