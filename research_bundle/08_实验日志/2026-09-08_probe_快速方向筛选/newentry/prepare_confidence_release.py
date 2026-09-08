"""One-time source derivation; never changes the running direction release."""
from pathlib import Path
import shutil
import yaml

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'release'
DEST=HERE/'confidence_release'
DEST.mkdir(exist_ok=True)
SCOPE='LLVIP_CONFIDENCE_FT3'
ENDPOINT='LLVIP_CONFIDENCE_FT3_LAST_EMA'


def put(name,text):
    with (DEST/name).open('x',encoding='utf-8') as f:f.write(text)


def replace(text,old,new):
    if old not in text:raise ValueError('Expected source fragment absent: '+old[:70])
    return text.replace(old,new)


common=(SOURCE/'direction_common.py').read_text(encoding='utf-8')
common=replace(common,'DIRECTION_FT3_BNFROZEN_LAST_EMA',ENDPOINT)
common=replace(common,"if c.get('dataset') not in ('drone','llvip'):raise ValueError('Dataset not admitted')", "if c.get('dataset')!='llvip':raise ValueError('Only LLVIP admitted')")
common=replace(common,"arms=('N','C1','C2','F-rel') if c['dataset']=='drone' else ('N','L2-box','L2-GT')", "arms=('N','C0')")
common=replace(common,'DIRECTION_FT3_BNFROZEN',SCOPE)
common=replace(common,"if not 0<=c['kd_coefficient']<=1:raise ValueError('Invalid fixed coefficient')\n    if c['arm']=='N' and c['kd_coefficient']!=0:raise ValueError('N must have zero dose')", "coefficient=0. if c['arm']=='N' else .1\n    for key in ('kd_coefficient','kd_weight','classification_coefficient'):\n        if c.get(key)!=coefficient:raise ValueError('Original C0 fixed coefficient differs: '+key)\n    if c.get('localization_coefficient')!=0.:raise ValueError('No localization loss permitted')\n    if c.get('expected_val_images')!=2406 or c.get('expected_nc')!=1:raise ValueError('Full LLVIP dev identity differs')")
put('confidence_common.py',common)

train=(SOURCE/'train_direction.py').read_text(encoding='utf-8')
train=replace(train,'from direction_common import','from confidence_common import')
start=train.index('def build_private(');end=train.index('\ndef install_bn_freeze',start)
train=train[:start]+'''def build_private(runtime,cfg,args):
    from confidence_criterion import make_type
    legacy=runtime.legacy
    ScreenCriterion=make_type(runtime.ORIGINAL_CRITERION)
    def frozen(path,expected_data,names):
        key='privileged_data_yaml' if str(path)==cfg['teacher'] else 'student_data_yaml'
        if str(path) not in (cfg['teacher'],cfg['reference']) or str(expected_data)!=cfg['paths'][key]:
            raise ValueError('Unexpected frozen binding')
        return legacy.load_frozen(path,cfg['auxiliary_data_identity'][key],names)
    builder=cloned(legacy.build_trainer,EvidenceCriterion=ScreenCriterion,
        DualLabelRGBIRDataset=runtime.TrackedDualLabelRGBIRDataset,load_frozen=frozen)
    return builder,[]

''' + train[end:]
start=train.index('    import independent_criterion as criterion');end=train.index('    legacy=runtime.legacy',start)
train=train[:start]+"    if Path(runtime.__file__).resolve()!=ref/'runtime.py':raise ValueError('Wrong pinned runtime')\n"+train[end:]
train=replace(train,'build_private(runtime,criterion,selection,classification,cfg,args)','build_private(runtime,cfg,args)')
train=replace(train,'DIRECTION_CANARY_COMPLETED','CONFIDENCE_CANARY_COMPLETED')
train=replace(train,'DIRECTION_TRAINING_COMPLETED','CONFIDENCE_TRAINING_COMPLETED')
train=replace(train,'DIRECTION_FT3_BNFROZEN',SCOPE)
train=replace(train,'direction_config.yaml','confidence_config.yaml')
train=replace(train,'direction_failure.json','confidence_failure.json')
train=replace(train,"status='HOURLY_SCREEN_FAILED'","status='CONFIDENCE_TRAINING_FAILED',scope='LLVIP_CONFIDENCE_FT3'")
train=replace(train,"c.get('arm')!=cfg['arm'] or c.get('successful_updates',0)<24", "c.get('arm')!=cfg['arm'] or c.get('successful_updates',0)<24 or c.get('scope')!='LLVIP_CONFIDENCE_FT3' or c.get('dataset')!='llvip' or c.get('bn_running_buffers_unchanged') is not True")
train=replace(train,'Warm-start N/C0/C1','Warm-start LLVIP N/C0 original object-evidence')
put('train_confidence.py',train)

ev=(SOURCE/'evaluate_direction.py').read_text(encoding='utf-8')
ev=replace(ev,'DIRECTION_FT3_BNFROZEN_LAST_EMA',ENDPOINT)
ev=replace(ev,'DIRECTION_FT3_BNFROZEN',SCOPE)
ev=replace(ev,"ARMS = ('N', 'C1', 'C2', 'F-rel', 'L2-box', 'L2-GT')","ARMS = ('N', 'C0')")
ev=replace(ev,"DATASET_ARMS = {'drone': ('N', 'C1', 'C2', 'F-rel'), 'llvip': ('N', 'L2-box', 'L2-GT')}","DATASET_ARMS = {'llvip': ('N', 'C0')}")
ev=replace(ev,"POPULATIONS = {'drone': (1469, 22462, 5), 'llvip': (2406, 7879, 1)}","POPULATIONS = {'llvip': (2406, 7879, 1)}")
ev=replace(ev,"    cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8-sig'))", "    from confidence_common import load_config as fixed_confidence_config\n    cfg = fixed_confidence_config(path)")
ev=replace(ev,"cfg.get('direction_screen', {})","cfg.get('confidence_screen', {})")
ev=replace(ev,'DIRECTION_TRAINING_COMPLETED','CONFIDENCE_TRAINING_COMPLETED')
ev=replace(ev,'direction_config.yaml','confidence_config.yaml')
ev=replace(ev,"training_completion=str(completion_path), evaluation_identity_projection=projection,", "training_completion=str(completion_path), evaluation_identity_projection=projection,\n            training_subset_identity={key:cfg['paths'][key] for key in ('student_data_yaml','privileged_data_yaml','paired_train_mapping')},")
put('evaluate_confidence.py',ev)

(DEST/'configs').mkdir(exist_ok=True)
for arm in ('N','C0'):
    cfg=yaml.safe_load((SOURCE/'configs/llvip_N_s42_FT3.yaml').read_text(encoding='utf-8'))
    cfg.update(method_id='LLVIP-CONFIDENCE-'+arm,method_identity=SCOPE,description='Fixed LLVIP original C0 confidence FT3 screen',
        scope=SCOPE,arm=arm,kd_coefficient=0. if arm=='N' else .1,kd_weight=0. if arm=='N' else .1,
        classification_coefficient=0. if arm=='N' else .1,localization_coefficient=0.,protocol_status='FROZEN_EXPLORATORY_CONFIDENCE')
    meta=cfg.pop('direction_screen');meta.update(scope=SCOPE,endpoint=ENDPOINT,comparison_arms=['N','C0'],
        initialization='Completed LLVIP visible42 last/EMA',coefficient_recalibrated=False)
    meta.pop('candidate_source',None);meta.pop('approved_candidate_source',None)
    cfg['confidence_screen']=meta
    for key in ('classification','localization','calibration'):
        cfg.pop(key,None)
    with (DEST/'configs'/('llvip_'+arm+'_s42_FT3.yaml')).open('x',encoding='utf-8') as f:yaml.safe_dump(cfg,f,sort_keys=False,allow_unicode=True)
native=DEST/'configs/llvip_native_evaluation.yaml'
if native.exists():raise FileExistsError(native)
shutil.copyfile(SOURCE/'configs/llvip_native_evaluation.yaml',native)
print('CONFIDENCE_SOURCE_PREPARED_NO_EXECUTION')
