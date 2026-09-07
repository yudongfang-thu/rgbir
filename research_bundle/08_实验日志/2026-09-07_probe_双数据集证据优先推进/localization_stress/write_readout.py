"""Render frozen stress outputs without fitting, reselection or new metrics."""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs_attempt2'


def read_csv(name):
    with (OUT / name).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


rows = read_csv('all_metrics.csv')
worst = read_csv('all_worst_case.csv')
small_fields = ['dataset','split','fixed_stratum','condition_index','dx_input_px','dy_input_px','linf_px','l2_px','fixed_n',
                'teacher_unclamped_support_n','iou_advantage_mean','teacher_minus_native_ce_mean','teacher_ce_better_fraction',
                'kl_t2_mean4_mean','cosine_mean','positive_cosine_fraction_fixed','original_quality_gate_n',
                'original_quality_gate_survived_n','original_quality_gate_survival_fraction','new_quality_gate_members_n',
                'lost_mass_t1_mean4_mean','lost_mass_t2_mean4_mean','conditional_vs_exact_box_max_abs_px_mean',
                'conditional_vs_exact_box_max_abs_px_max']
with (OUT / 'compact_metrics.csv').open('w',encoding='utf-8-sig',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=small_fields); writer.writeheader()
    writer.writerows({k:r[k] for k in small_fields} for r in rows)
display = {'llvip':'LLVIP','dronevehicle':'Drone'}
layer = {'base':'全部基础对象','n_iou_lt_070':'仅N IoU<.70','quality_gate070':'固定quality_gate070'}
def v(r,k): return float(r[k])
def pct(r,k): return f'{v(r,k)*100:.2f}%'

lines = ['# 固定 anchor 定位目标平移压力诊断（2026-09-07，attempt2修正）','',
'旧attempt1的worst汇总`all_directions_gate_n`被统计分母覆盖，旧README表格误写61/61、47/47；该字段不能作为存活数引用。旧原始attempt与源码/报告快照完整保留于`outputs_attempt1/`和`attempt1_source_snapshot/`。attempt2用`success_n`/`denominator_n`区分成功计数和分母；原逐对象数组、比例与正文4px的34/61、9/47未受影响。修复未改协议、cohort、anchor、阈值或扰动集合。','',
'> **已执行两数据集的固定 25 条件 CPU 诊断，等待独立审阅。LLVIP 的条件定位质量在本压力测试中比 Drone 更稳定；其 dev 固定质量门的 61 对象在四输入像素八方向下仍有 55/61 的教师 DFL CE 始终更低，但 gate 仅 34/61 全方向存活，训练快照只有 23 个门内对象。此结论不能替代原 L1 准入或实际定位训练。**','',
'## 目的与设置','',
'检查同一教师定位目标在人为坐标扰动下的质量和检测头局部监督方向。先冻结 [PROTOCOL.md](PROTOCOL.md)，执行 10 项已知真值检查，然后按 LLVIP→Drone 顺序读取上一阶段 raw DFL/logits 和对象 JSONL；只用 CPU，完整执行约14秒，没有新推理、拟合、GPU、test、训练或旧产物修改。','',
'25 条件为零扰动及 a=1/2/4 时的四轴向与四对角方向。a 是 L∞ 幅度，对角 L2=a√2。原 RGB/IR GT、N42、对象和 anchor 均固定；平移教师期望框，不裁画布、不 clamp、不按扰动后 gate 重新选择。所有主表保留零扰动固定分母。','',
'基础 cohort 与上一阶段定位 readout 的 source_row/object_id/顺序完全一致：非背景、paired GT IoU≥.5、有N参考候选、RGB GT四边未 clamp 距离全在[0,14.99]。quality_gate070 沿用旧质量条件：N/T类正确且conf≥.25，N IoU<.70，T对RGB GT IoU≥.60、对IR GT IoU≥.50、T−N>.05。它只是本共同anchor上的描述性分层，不等于 L1/D2 selected；未把pair≥.8、唯一owner或独立几何证据补造成已通过。','',
'## 固定分母与训练覆盖上下文','',
'|数据集/划分|全部RGB GT|基础cohort|N IoU<.70|quality_gate070|门内唯一图|门内唯一owner=1|',
'|---|---:|---:|---:|---:|---:|---:|']
for dataset in ('llvip','dronevehicle'):
    s=json.loads((OUT/dataset/'summary.json').read_text(encoding='utf-8'))
    for split in ('train','val'):
        c={r['fixed_stratum']:r for r in s['stratum_context'] if r['split']==split}
        lines.append(f"|{display[dataset]} {split}|{s['filter_counts'][split]['real_rgb_gt']}|{c['base']['n']}|{c['n_iou_lt_070']['n']}|{c['quality_gate070']['n']}|{c['quality_gate070']['unique_images']}|{c['quality_gate070']['native_owner_count_eq1_n']}|")
lines += ['', 'LLVIP 门内 train 23/2728=0.84%基础对象、dev 61/548=11.13%；Drone相应87/15019=0.58%、47/2821=1.67%。这是已训练端点和固定letterbox的静态快照，不能估计真实自然增强64批的有效监督量。LLVIP train门内来自10组且全为stride16，dev来自5组；Drone dev的source_group为`unavailable:val`，不能当作有来源多样性证据。','',
'## 零扰动 dev','',
'CE差定义为 teacher−native，负数表示教师对固定RGB GT的DFL目标更好。框IoU优势为 teacher−native，正数表示教师更好。','',
'|数据集|固定层|n|框IoU优势均值|DFL CE差均值|教师CE更低|KD/GT cosine>0|',
'|---|---|---:|---:|---:|---:|---:|']
for r in rows:
    if r['split']=='val' and r['condition_index']=='0':
        lines.append(f"|{display[r['dataset']]}|{layer[r['fixed_stratum']]}|{r['fixed_n']}|{v(r,'iou_advantage_mean'):+.5f}|{v(r,'teacher_minus_native_ce_mean'):+.5f}|{pct(r,'teacher_ce_better_fraction')}|{pct(r,'positive_cosine_fraction_fixed')}|")
lines += ['', '同anchor框优势与上一阶段“各自候选框”的对象级IoU优势不是同一指标。零扰动CE、KL与cosine逐对象重算与上一阶段定义最大差均为0。','',
'## dev固定质量门：八方向最差情况','',
'每个对象先在给定幅度的八方向取最差，再对原固定集合求均值；“CE始终更低”等比例要求八方向全部满足。gate存活是另报的覆盖量，下面损失/IoU均没有改用幸存者分母。','',
'|数据集|a输入px|固定n|最小IoU优势均值|最大CE差均值|IoU始终更好|CE始终更低|cosine始终>0|gate全方向存活|',
'|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in worst:
    if r['split']=='val' and r['fixed_stratum']=='quality_gate070':
        lines.append(f"|{display[r['dataset']]}|{r['linf_px']}|{r['fixed_n']}|{v(r,'min_iou_advantage_mean'):+.5f}|{v(r,'max_teacher_minus_native_ce_mean'):+.5f}|{pct(r,'all_directions_iou_better_fraction')}|{pct(r,'all_directions_ce_better_fraction')}|{pct(r,'all_directions_positive_cosine_fraction')}|{r['all_directions_gate_success_n']}/{r['all_directions_gate_denominator_n']}|")
lines += ['', '这支持继续检查LLVIP的条件定位目标，不能推出整个LLVIP全对象稳健：基础cohort在a=1/2/4的逐对象最差IoU优势均值为+.01471/−.01745/−.08695，最大CE差均值为−.11324/−.06483/+.05158。Drone基础对象零扰动已是平均IoU/CE质量不利；其固定条件层仍有局部机会，但a=4时47对象的最差IoU优势均值转为−.02310、仅9/47全方向保住quality gate。','',
'## DFL近似、支持域与梯度边界','',
'DFL平移是离散概率质量重采样近似：每个原bin质量移动到相邻整数bin，小于0/大于15的部分另记underflow/overflow，不clamp到边界。T=1与T=2概率分别搬运；用于CE/KL/梯度的是保留质量的条件分布。搬运与温度化不交换，这不是可反推的真实新网络logits。零扰动概率逐元素exact，边界流失为0。','',
'a=4八方向下，固定质量门的逐对象最大T1边界流失（四边均值）再平均为LLVIP1.743%、Drone0.998%；均值小不代表所有边都小。作为透明示例，预定(+4,+4)条件的单边最大流失分别19.65%/25.04%，条件分布解码框偏离直接平移框最大3.778/3.654输入像素。因此不能把损失CE在压力下较好等同于精确坐标已稳健。所有25条件与每对象每边质量保存于原始产物，没有仅挑该示例作为主结果。','',
'同样在(+4,+4)时，LLVIP/Drone门内未clamp教师期望距离仍全在[0,14.99]的对象为59/61、46/47；域外对象保留在主表。RGB GT支持域固定，IR GT支持域与owner上下文单列。分布的边界质量流失和期望距离是否越界是两种不同读数。','',
'KD为KL(pT_shift,T=2 || pN,T=2)×4、四边平均；head梯度为2(pN2−pT2)/4，相比GT梯度(pN1−qGT)/4，逐对象64维cosine/dot/norm都保存。这里只读DFL raw logits局部梯度，没有完整native/CIoU/TAL/共享特征梯度或实际学习更新。两数据集全部对象×25条件的CE/KL/cosine非有限数均为0；不能把cosine正读作训练可学性保证。','',
'## 产物、验证与下一步','',
'- [执行脚本](localization_stress.py)、[冻结协议](PROTOCOL.md)、[报告生成脚本](write_readout.py)。',
'- [精简逐条件原始表](outputs_attempt1/compact_metrics.csv)、[完整分布统计](outputs_attempt1/all_metrics.csv)、[八方向最差原始表](outputs_attempt1/all_worst_case.csv)。',
'- 每数据集 `cohort.csv` 保存全部源行和固定层；`per_object.npz` 保存全部基础对象×25条件指标，以及T1/T2逐边underflow/overflow、支持域与gate布尔表；`summary.json` 保存过滤链、模型身份路径、源文件stat及零扰动回验。',
'- [CPU真值回执](outputs_attempt1/known_truth_tests.json)：10项通过；64个KD与GT梯度均以中心有限差分核验，另含质量守恒/无clamp/零扰动exact/严格.70/固定mask等。',
'- [保存产物自检](saved_output_self_check.json)及[复核脚本](verify_saved.py)：两数据集零扰动逐对象对旧dfl_per_object.csv的CE/KL/cosine最大差0，全部保存表均值/固定mask/gate存活回算最大差0，源文件size/mtime保持不变。这是执行者自检，不能代替独立接受。',
'- [完成回执](outputs_attempt1/completion_receipt.json)：`COMPLETED_PENDING_INDEPENDENT_REVIEW`。',
'', '建议下一步按既有方案核对LLVIP匹配N、自然训练流的覆盖与同mask GT控制，并保留Drone局部条件机会；不得据本结果放宽.70门或宣布L1通过。此处没有置信区间、独立seed干预、真实配准误差测量或AP/KD增益；LLVIP共享标签也不是物理配准真值。']
(HERE/'README.md').write_text(('\n'.join(lines)+'\n').replace('outputs_attempt1/compact_metrics','outputs_attempt2/compact_metrics').replace('outputs_attempt1/all_metrics','outputs_attempt2/all_metrics').replace('outputs_attempt1/all_worst_case','outputs_attempt2/all_worst_case').replace('outputs_attempt1/known_truth_tests','outputs_attempt2/known_truth_tests').replace('outputs_attempt1/completion_receipt','outputs_attempt2/completion_receipt').replace('执行 10 项','执行 11 项').replace('10项通过','11项通过'),encoding='utf-8')
print('Wrote README.md and compact_metrics.csv; no analysis re-fit or re-selection.')
