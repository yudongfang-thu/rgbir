"""Copy this completed phase's small evidence into the already-authorized review branch.

No credentials, weights, dataset images, new hashes, or git mutations are handled here.
"""
from pathlib import Path
import json
import shutil

phase = Path(__file__).resolve().parent
workspace = phase.parent.parent
repo = Path('C:/Users/MSI-PC/rgbir_review_worktrees/evidence-20260906')
target = repo / 'research_bundle' / '08_实验日志' / phase.name
allowed = {'.md', '.py', '.json', '.jsonl', '.txt', '.log', '.yaml', '.npz', '.exit', '.sh'}
copied = []
for src in sorted(phase.rglob('*')):
    if not src.is_file() or (src.suffix not in allowed and src.name != 'LICENSE'):
        continue
    if src.stat().st_size > 2_000_000:
        raise RuntimeError(f'Unexpected large file: {src.name}')
    dst = target / src.relative_to(phase)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append((str(src.relative_to(phase)), src.stat().st_size))

for rel in ['README.md', '08_实验日志/README.md', '99_整理回执/20260909_CFT原文协议复现.md']:
    dst = repo / 'research_bundle' / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(workspace / rel, dst)

entry = '''# 原文协议优先：CFT／LLVIP 实际复现结果（2026-09-09）

**原作者权重复评已基本对齐；从头训练未准入。新增迁移到我方配置的实验暂停。**

先在原模型、原划分、原标注和原评估条件下验证作者方法，再讨论迁移效果。否则，迁移负结果会混淆原方法可复现性、实现错误与适用条件变化，不能用于判断论文是否造假。

| 指标 | 论文（%） | 本次（%） | 本次−论文（pp） |
|---|---:|---:|---:|
| AP50 | 97.5 | 97.376907 | −0.123093 |
| AP75 | 72.9 | 72.891348 | −0.008652 |
| AP50–95 | 63.6 | 63.537478 | −0.062522 |

90服务器上，CFT作者双流融合模型及权重、官方previous标注、完整LLVIP test 3463对/7931GT、1024输入、作者评估入口，完整进程90.36秒，exit=0。覆盖和指标聚合经有限范围独立复核，同环境CPU重算三个AP完全一致。论文数值来自[CFT Table 3](https://arxiv.org/html/2111.00273v2#S4.T3)，模型源码来自[作者仓库](https://github.com/DocF/multispectral-object-detection)。

身份为PAPER-RECONSTRUCTED checkpoint reevaluation：保留旧checkpoint运行时类兼容、现代环境与历史标签代码无法逐字节确认等限制。没有三项精确吻合、从头训练、多seed或跨模态蒸馏增益结论。CFT推理需要RGB+IR，不能直接充当我方RGB-only部署的公平对手。

原B32/1024训练的单卡及作者双卡DataParallel三次资源检查均在首批前向OOM，0次成功optimizer更新；没有取消项目每卡显存低于70%的限制。E200未启动，没有测得其整程时长，全部本轮lease已释放。保留三次失败原件；这不是方法负结果，也不证明完整24GB或其他硬件不能训练。

当前用户明确授权原论文协议评估，官方test暴露单列记录；我方grouped train/dev原件不变，此结果不用于我方方法选择或调阈值。新旧官方标签test分别7931/8302GT，不能混比。

- [完整记录、实际设置、训练失败与下一步](research_bundle/08_实验日志/2026-09-09_repro_CFT原文协议/README.md)
- [协议和兼容差异](research_bundle/08_实验日志/2026-09-09_repro_CFT原文协议/PROTOCOL_REVIEW.md)
- [独立指标复核与局限](research_bundle/08_实验日志/2026-09-09_repro_CFT原文协议/EVALUATION_REVIEW.md)
- [原执行完成回执](research_bundle/08_实验日志/2026-09-09_repro_CFT原文协议/server_execution/official_test_attempt1/receipt.json)
- [训练资源检查汇总](research_bundle/08_实验日志/2026-09-09_repro_CFT原文协议/server_training/training_feasibility.json)
- [Drone原文方法队列：尚缺作者权重/标签，未启动](research_bundle/08_实验日志/2026-09-09_repro_CFT原文协议/DRONE_ORIGINAL_PROTOCOL_QUEUE.md)

本次公开小原始预测、指标数组、实际脚本和失败日志；不包含模型权重、数据集原图或凭据。作者源码快照附原LICENSE。之前REPRODUCTION_STATUS_20260909.md保留为本轮早期接入状态，不再代表最新结果或排程。
'''
(repo / 'AUTHOR_PROTOCOL_STATUS_20260909.md').write_text(entry, encoding='utf-8')
banner = '> **2026-09-09 最新原文复现：**[CFT／LLVIP 作者权重复评与训练准入结果](AUTHOR_PROTOCOL_STATUS_20260909.md)。完整3463对test的AP50/AP75/mAP为97.376907/72.891348/63.537478%，距论文均≤0.13pp；覆盖/指标复核完成。原B32训练三次资源检查首批OOM，E200未启动。新增迁移版暂停。下方均为历史阶段记录。\n\n'
for name in ['README.md', 'LATEST_RESULTS.md']:
    path = repo / name
    content = path.read_text(encoding='utf-8-sig')
    if banner not in content:
        path.write_text(banner + content, encoding='utf-8')
print(json.dumps({'copied_files': len(copied), 'bytes': sum(n for _, n in copied), 'excluded': 'compressed duplicate archives; weights; images; credentials', 'target': str(target)}, ensure_ascii=False))
