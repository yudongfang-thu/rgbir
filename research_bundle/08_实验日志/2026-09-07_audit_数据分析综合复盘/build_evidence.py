"""Snapshot existing small evidence and plot the accepted class-oracle values.

No inference, training, new hashes, or changes to original evidence.
"""
import csv
import datetime
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
EARLY = ROOT / '08_实验日志/2026-09-06_probe_RGBIR数据特性与可迁移知识'
BASE = ROOT / '08_实验日志/2026-09-07_probe_Baseline蒸馏机会重诊断'
NEW = ROOT / '08_实验日志/2026-09-07_probe_双数据集证据优先推进'
sources = {
    'early_probe_analysis.json': EARLY / 'probe_analysis.json',
    'baseline_compact_findings.json': BASE / 'new_probe_analysis/compact_real_findings.json',
    'class_oracle_summary.json': NEW / 'ap_error/class_oracle_v1/summary.json',
    'drone_ap_summary.json': NEW / 'ap_error/drone_results_v1/summary.json',
    'llvip_ap_summary.json': NEW / 'ap_error/llvip_results_v1/summary.json',
    'natural_flow_summary.json': NEW / 'natural_flow_diagnostic/completed_readout_attempt2/summary.json',
}
snapshot = OUT / 'source_snapshots'
snapshot.mkdir(exist_ok=True)
records = []
for name, source in sources.items():
    raw = source.read_bytes()
    assert len(raw) < 5_000_000, str(source)
    target = snapshot / name
    with target.open('xb') as stream:
        stream.write(raw)
    assert target.read_bytes() == raw
    stat = source.stat()
    records.append(dict(source=str(source), copy=str(target),
                        size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                        byte_comparison=True))

data = json.loads((snapshot / 'class_oracle_summary.json').read_text(encoding='utf-8-sig'))
seeds = [0, 42, 123]
classes = [data['N_s0']['0.5']['per_class'][str(i)]['name'] for i in range(5)]
gt = np.array([data['N_s0']['0.5']['per_class'][str(i)]['gt'] for i in range(5)])
delta = np.array([[data['N_s'+str(s)]['0.5']['per_class'][str(i)]['Cls_oracle_dAP']
                   for i in range(5)] for s in seeds])
mean = delta.mean(axis=0)
sd = delta.std(axis=0, ddof=1)
gt_share = 100 * gt / gt.sum()
oracle_share = 100 * mean / mean.sum()
assert int(gt.sum()) == 22462
assert abs(mean.mean() - 10.359657861) < 1e-7
for s in seeds:
    assert [data['N_s'+str(s)]['0.5']['per_class'][str(i)]['gt'] for i in range(5)] == gt.tolist()
minority = [1, 2, 4]
table = []
for i, name in enumerate(classes):
    table.append(dict(class_name=name, gt=int(gt[i]), gt_share_pct=float(gt_share[i]),
                      cls_dap_seed0_pp=float(delta[0, i]),
                      cls_dap_seed42_pp=float(delta[1, i]),
                      cls_dap_seed123_pp=float(delta[2, i]),
                      cls_dap_mean_pp=float(mean[i]), cls_dap_sample_sd_pp=float(sd[i]),
                      cls_oracle_share_pct=float(oracle_share[i])))
with (OUT / 'class_oracle_readout.csv').open('x', encoding='utf-8', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(table[0]))
    writer.writeheader()
    writer.writerows(table)
fp = Path('C:/Windows/Fonts/msyh.ttc')
font = FontProperties(fname=str(fp)) if fp.exists() else FontProperties()
fig, ax = plt.subplots(figsize=(11.5, 5.6), dpi=180)
x = np.arange(5)
width = .35
bars1 = ax.bar(x-width/2, gt_share, width, color='#31688e', label='GT对象数量占比')
bars2 = ax.bar(x+width/2, oracle_share, width, color='#d66b32', label='分类oracle贡献占比')
for bars in (bars1, bars2):
    ax.bar_label(bars, labels=[f'{b.get_height():.1f}%' for b in bars],
                 padding=4, fontsize=10)
ax.set_xticks(x, classes, fontsize=11)
ax.set_ylim(0, 98)
ax.set_ylabel('各自分母内的占比（%）', fontproperties=font)
ax.set_title('Drone：对象数量多，不等于对宏平均 AP 的分类瓶颈贡献大',
             fontproperties=font, fontsize=15, pad=17)
ax.legend(prop=font, frameon=False, loc='upper right')
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', alpha=.16)
ax.set_axisbelow(True)
note = ('完整 dev：1469 图 / 22462 GT；N seeds 0、42、123。\n'
        '橙色为各类 Cls dAP 三 seed 均值占五类总和的比例；不是全部 AP 错误，更不是 KD 收益。')
fig.text(.08, .015, note, fontproperties=font, fontsize=10, color='#444444')
fig.tight_layout(rect=(0, .09, 1, 1))
fig.savefig(OUT / 'drone_class_counts_vs_oracle.png')
fig.savefig(OUT / 'drone_class_counts_vs_oracle.svg')
plt.close(fig)
receipt = dict(created_at=datetime.datetime.now().astimezone().isoformat(),
               scope='Existing accepted evidence synthesis and arithmetic-only visualization',
               source_records=records, new_gpu_tasks=0, new_hashes=False,
               minority_gt_share_pct=float(gt_share[minority].sum()),
               minority_class_oracle_share_pct=float(oracle_share[minority].sum()),
               macro_class_oracle_pp=float(mean.mean()),
               all_classes=table)
with (OUT / 'evidence_receipt.json').open('x', encoding='utf-8') as stream:
    json.dump(receipt, stream, ensure_ascii=False, indent=2)
print(json.dumps({k:v for k,v in receipt.items() if k not in ('source_records','all_classes')}))
