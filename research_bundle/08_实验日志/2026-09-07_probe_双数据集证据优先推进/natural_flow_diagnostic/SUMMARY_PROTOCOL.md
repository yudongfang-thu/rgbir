# 完成后只读汇总合同

冻结于完整结果下载之前。只读取 remote_completed_attempt2 的 llvip_full_attempt1 与 drone_full_attempt1；两者均须 status=COMPLETED、batches=64、canary_only=false、seed20260907、无几何/训练准入。任何缺文件、未完成、2批canary、批序/原trace不一致均拒绝；两组全部验证后才新建汇总输出。

从selection_batches.jsonl与natural_batches.jsonl重算，每批B32，64批共2048不重复源图；natural需与该full保存的original_natural_batches完全相同。C base/eligible/selected记录及类别计数、L整批/逐图gate计数和全局summary的对象/图/来源组逐项核对。原始JSONL保持不变。

- 每gate损失为上一累计gate对象数减本级对象数；损失比例分母是上一gate对象数，零分母为NA。teacher_gt_count是配对上下文，不混入RGB→matched累计损失链。
- C按完整names字典报告每类base份额、eligible/base、selected/base、selected/eligible与selected在全选中对象中的份额；零对象类别保留。C0/C1共享选择集合，不拆为两项独立实验。
- C有效尺度指共同有效的P3/P4层，分别报告base/eligible/selected中的层有效对象数/比例及平均有效层数；两层可以同时有效，不把它们当互斥物体大小档。L另报base/selected anchor的stride8/16分布。
- 图/来源组集中度按各阶段对象贡献计算top1/top5占比。主表突出selected阶段，原直方图完整保存；未知组单列并保留在分母中，不伪称来源多样性或标定组。
- selected批次数只表示至少一个selected对象，不代表非零或有益梯度。所有结果保持UNVERIFIED_GEOMETRY_DIAGNOSTIC，不称正式L1数量严格上界、实际KD剂量、训练收益或AP。

入口 `summarize_completed.py --input <完整下载目录> --output <新汇总目录>`；默认输出completed_readout_attempt2，已存在则拒绝覆盖。CPU真值在summary_cpu_tests.json，真实结果须由独立代理再核验。
