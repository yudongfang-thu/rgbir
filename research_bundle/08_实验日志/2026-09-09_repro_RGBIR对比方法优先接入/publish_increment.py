"""Stage a small review copy; original evidence and large assets stay in place."""
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path('E:/SHARE/光sar')
REPRO = ROOT / '08_实验日志/2026-09-09_repro_RGBIR对比方法优先接入'
REPO = Path('C:/Users/MSI-PC/rgbir_review_worktrees/evidence-20260906')
BRANCH = 'research/full-evidence-20260906'
assert subprocess.check_output(['git', '-C', str(REPO), 'branch', '--show-current'], text=True).strip() == BRANCH
assert not subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain'], text=True).strip()
allowed = {'.md', '.json', '.py', '.yaml', '.yml', '.log', '.txt', '.npz', '.tsv', '.csv'}
excluded_dirs = {'local_papers', 'primary', 'public_sources', 'CFT_source', 'AMFD', 'CFT',
                 'CrossFusionKD', 'ICAFusion', 'M2D-LIF', '__pycache__', '.git'}
copied, excluded = [], []
for folder in (REPRO, ROOT / '08_实验日志/2026-09-09_ops_90迁移与训练接入'):
    for src in sorted(folder.rglob('*')):
        if not src.is_file():
            continue
        rel = src.relative_to(ROOT)
        if any(p in excluded_dirs for p in rel.parts) or src.suffix.lower() not in allowed or src.stat().st_size > 2_000_000:
            excluded.append({'path': rel.as_posix(), 'bytes': src.stat().st_size, 'reason': 'external full source/text/archive or non-small artifact; kept in local/remote evidence'})
            continue
        dest = REPO / 'research_bundle' / rel
        assert not dest.exists(), str(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied.append({'path': dest.relative_to(REPO).as_posix(), 'bytes': dest.stat().st_size})

# Stage one new dated index; retain the repository's historical indexes intact.
index = REPO / 'REPRODUCTION_STATUS_20260909.md'
assert not index.exists()
index.write_text('''# 2026-09-09：90 上的实际复现接入

CFT 作者 checkpoint 经显式兼容适配完成三对 LLVIP fit 前向；BCDL 分类算子 6/6、CMD 原 22 项 CPU 测试通过。没有新增 AP、完整训练或论文整表复现。

- [结果、方法优先级与限制](research_bundle/08_实验日志/2026-09-09_repro_RGBIR对比方法优先接入/README.md)
- [证据范围审阅](research_bundle/08_实验日志/2026-09-09_repro_RGBIR对比方法优先接入/EVIDENCE_REVIEW.md)
- [90 数据与环境接入](research_bundle/08_实验日志/2026-09-09_ops_90迁移与训练接入/README.md)

94 仍故障且旧任务未恢复。本轮无 GPU 长训。CMD 真实 IR 教师仍待绑定。融合模型和 OBB 结果不能充当 RGB-only/HBB 的公平主表对比。

本增量保留报告、脚本、失败/成功回执与小原始预测；不上传权重、数据集图像、第三方论文全文和大源码归档。详细排除清单见 REPRODUCTION_INCREMENT_20260909.json；完整资产位置见报告。本增量不刷新旧总 BUNDLE_MANIFEST 的历史范围，不计算新文件摘要。
''', encoding='utf-8')
for name in ('README.md', 'LATEST_RESULTS.md'):
    dest = REPO / name
    old = dest.read_text(encoding='utf-8')
    dest.write_text('> **2026-09-09 最新：**[90 实际复现接入与范围](REPRODUCTION_STATUS_20260909.md)：CFT 真实权重前向、BCDL 算子、CMD 22 tests 已执行；无新 AP。以下时间点为历史状态。\n\n' + old, encoding='utf-8')
manifest = dict(scope='2026-09-09 reproduction evidence and migration context', copied=copied,
                excluded=excluded, total_copied_bytes=sum(i['bytes'] for i in copied),
                new_digest_calculated=False, weights_or_dataset_images_included=False,
                full_paper_text_included=False, prior_manifest_is_historical=True)
(REPO / 'REPRODUCTION_INCREMENT_20260909.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
(REPRO / 'PUBLICATION_STAGING.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

# A small mirrored folder for navigable remote review, with identical layout.
archive = REPRO / 'review_snapshot_20260909.tar.gz'
assert not archive.exists()
with tarfile.open(archive, 'w:gz') as tf:
    for item in copied:
        src = REPO / item['path']
        tf.add(src, arcname=item['path'], recursive=False)
print(json.dumps({'copied_files': len(copied), 'bytes': manifest['total_copied_bytes'],
                  'excluded_files': len(excluded), 'archive_bytes': archive.stat().st_size}))
