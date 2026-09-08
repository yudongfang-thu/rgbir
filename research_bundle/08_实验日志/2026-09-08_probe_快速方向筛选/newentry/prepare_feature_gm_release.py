"""Separate post-calibration, pre-feature-AP protocol; never changes old release."""
import json
from pathlib import Path
import shutil
import statistics
import math
import yaml

HERE=Path(__file__).resolve().parent;SOURCE=HERE/'release';DEST=HERE/'feature_gm_release'
DEST.mkdir(exist_ok=True)
SCOPE='FEATURE_RELATION_GM_FT3';ENDPOINT=SCOPE+'_LAST_EMA';COEF=14.438521129817886


def put(name,text):
    with (DEST/name).open('x',encoding='utf-8') as f:f.write(text)


def change(text,old,new):
    if old not in text:raise ValueError('Expected source absent: '+old[:100])
    return text.replace(old,new)


common=(SOURCE/'direction_common.py').read_text(encoding='utf-8')
common=change(common,'DIRECTION_FT3_BNFROZEN_LAST_EMA',ENDPOINT)
common=change(common,"if c.get('dataset') not in ('drone','llvip'):raise ValueError('Dataset not admitted')", "if c.get('dataset')!='drone':raise ValueError('Only Drone admitted')")
common=change(common,"arms=('N','C1','C2','F-rel') if c['dataset']=='drone' else ('N','L2-box','L2-GT')", "arms=('N','F-rel-GM')")
common=change(common,'DIRECTION_FT3_BNFROZEN',SCOPE)
common=change(common,"if not 0<=c['kd_coefficient']<=1:raise ValueError('Invalid fixed coefficient')\n    if c['arm']=='N' and c['kd_coefficient']!=0:raise ValueError('N must have zero dose')", "expected=0. if c['arm']=='N' else 14.438521129817886\n    if c.get('kd_coefficient')!=expected or c.get('classification_coefficient')!=expected:raise ValueError('Fixed feature-GM coefficient differs')\n    if c.get('localization_coefficient')!=0.:raise ValueError('No localization branch allowed')\n    if c.get('expected_val_images')!=1469 or c.get('expected_nc')!=5:raise ValueError('Full Drone identity differs')")
put('feature_gm_common.py',common)

criterion=(SOURCE/'direction_criterion.py').read_text(encoding='utf-8')
criterion=change(criterion,"            request='L2-box' if arm=='N' and self.cfg['dataset']=='llvip' else arm", "            request='L2-box' if arm=='N' and self.cfg['dataset']=='llvip' else arm\n            if arm=='F-rel-GM':request='F-rel'  # Explicit alias; identical frozen operator.")
put('feature_gm_criterion.py',criterion)
loss_source=SOURCE/'direction_losses.py';loss_target=DEST/'direction_losses.py'
if loss_target.exists():raise FileExistsError(loss_target)
shutil.copyfile(loss_source,loss_target)
if loss_source.read_bytes()!=loss_target.read_bytes():raise ValueError('Feature operator copy differs')

train=(SOURCE/'train_direction.py').read_text(encoding='utf-8')
train=change(train,'from direction_common import','from feature_gm_common import')
train=change(train,'from direction_criterion import make_type','from feature_gm_criterion import make_type')
train=change(train,'DIRECTION_FT3_BNFROZEN',SCOPE)
train=change(train,'direction_config.yaml','feature_gm_config.yaml')
train=change(train,'direction_failure.json','feature_gm_failure.json')
train=change(train,"status='HOURLY_SCREEN_FAILED'","status='FEATURE_GM_TRAINING_FAILED',scope='FEATURE_RELATION_GM_FT3'")
train=change(train,"c.get('arm')!=cfg['arm'] or c.get('successful_updates',0)<24", "c.get('arm')!=cfg['arm'] or c.get('successful_updates',0)<24 or c.get('scope')!='FEATURE_RELATION_GM_FT3' or c.get('dataset')!='drone' or c.get('bn_running_buffers_unchanged') is not True")
put('train_feature_gm.py',train)

ev=(SOURCE/'evaluate_direction.py').read_text(encoding='utf-8')
ev=change(ev,'DIRECTION_FT3_BNFROZEN_LAST_EMA',ENDPOINT)
ev=change(ev,'DIRECTION_FT3_BNFROZEN',SCOPE)
ev=change(ev,"ARMS = ('N', 'C1', 'C2', 'F-rel', 'L2-box', 'L2-GT')","ARMS = ('N', 'F-rel-GM')")
ev=change(ev,"DATASET_ARMS = {'drone': ('N', 'C1', 'C2', 'F-rel'), 'llvip': ('N', 'L2-box', 'L2-GT')}","DATASET_ARMS = {'drone': ('N', 'F-rel-GM')}")
ev=change(ev,"POPULATIONS = {'drone': (1469, 22462, 5), 'llvip': (2406, 7879, 1)}","POPULATIONS = {'drone': (1469, 22462, 5)}")
ev=change(ev,"    cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8-sig'))", "    from feature_gm_common import load_config as fixed_feature_gm_config\n    cfg = fixed_feature_gm_config(path)")
ev=change(ev,'direction_config.yaml','feature_gm_config.yaml')
ev=change(ev,"training_completion=str(completion_path), evaluation_identity_projection=projection,", "training_completion=str(completion_path), evaluation_identity_projection=projection,\n            expected_train_images=cfg['expected_train_images'],\n            training_subset_identity={key:cfg['paths'][key] for key in ('student_data_yaml','privileged_data_yaml','paired_train_mapping')},\n            matched_control_projection_required=True, matched_control_scope='DIRECTION_FT3_BNFROZEN',")
put('evaluate_feature_gm.py',ev)

cfg=yaml.safe_load((SOURCE/'configs/drone_F-rel_s42_FT3.yaml').read_text(encoding='utf-8'))
cfg.update(arm='F-rel-GM',scope=SCOPE,method_id='FEATURE-RELATION-GM-FT3',method_identity=SCOPE,
    description='Separate post-calibration and pre-feature-AP gradient-matched feature screen',
    kd_coefficient=COEF,classification_coefficient=COEF,localization_coefficient=0.,
    protocol_status='POST_CALIBRATION_PRE_FEATURE_AP_REVISION')
cfg['direction_screen'].update(scope=SCOPE,endpoint=ENDPOINT,comparison_arms=['matched-main-N','F-rel-GM'],
    coefficient_recalibrated=False,existing_fixed8_coefficient_reused=True,
    old_f_rel_blocked_record_unchanged=True,matched_control_scope='DIRECTION_FT3_BNFROZEN',
    matched_control_projection_required=True,formal_admission=False)
(DEST/'configs').mkdir(exist_ok=True)
with (DEST/'configs/drone_F-rel-GM_s42_FT3.yaml').open('x',encoding='utf-8') as f:yaml.safe_dump(cfg,f,sort_keys=False,allow_unicode=True)

caldir=HERE.parent/'results_1156_partial/calibration/drone'
receipt=json.loads((caldir/'calibration_receipt.json').read_text(encoding='utf-8'))
batches=[json.loads(line) for line in (caldir/'calibration_batches.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
assert receipt['details']['F-rel']['raw_median']==COEF and len(batches)==8
assert receipt['blocked']['F-rel']=='Unclipped coefficient outside (0,1]'
rows=[]
for r in batches:
    norms=r['unit_B_kd_norms'];target=.09227393550836771*norms['C1'];raw_ratio=target/norms['F-rel']
    values=[r['native_norm'],norms['C1'],target,norms['F-rel'],raw_ratio,COEF*norms['F-rel'],COEF*norms['F-rel']/target]
    assert all(math.isfinite(v) and v>0 for v in values)
    rows.append(dict(batch=r['batch'],native_norm=values[0],unit_B_C1_norm=values[1],target_B_lambda_C1_norm=values[2],
        unit_B_F_norm=values[3],raw_coefficient_ratio=values[4],fixed_B_lambda_F_norm=values[5],actual_F_over_C1_norm_ratio=values[6]))
ratios=[r['actual_F_over_C1_norm_ratio'] for r in rows]
result=dict(protocol_revision_scope=SCOPE,fixed_coefficient=COEF,old_status='BLOCKED',old_record_unchanged=True,
    calibration_scope=receipt['scope'],batches=8,existing_rows_reused=True,new_calibration=False,new_AP_read=False,
    raw_coefficient_median=statistics.median(r['raw_coefficient_ratio'] for r in rows),
    actual_ratio_min=min(ratios),actual_ratio_median=statistics.median(ratios),actual_ratio_max=max(ratios),
    ratio_definition='fixed_lambda_F * ||B grad F|| / (lambda_C1 * ||B grad C1||)',
    calibration_design='Median of per-batch target/unit-F coefficient ratios; actual norm-ratio median is reported, not forced to 1',
    finite_all_eight=True,formal_admission=False,new_hash_computed=False,rows=rows,
    sources=[dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in (caldir/'calibration_receipt.json',caldir/'calibration_batches.jsonl')])
put('CALIBRATION_RATIO_READOUT.json',json.dumps(result,ensure_ascii=False,indent=2))
shutil.copyfile(caldir/'calibration_receipt.json',DEST/'original_blocked_calibration_receipt.json')
print(json.dumps({k:result[k] for k in ('fixed_coefficient','actual_ratio_min','actual_ratio_median','actual_ratio_max')}))
