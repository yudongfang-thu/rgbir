"""Create a distinct frozen-BN training release from previously executed plumbing."""
from pathlib import Path
import yaml,copy
root=Path(__file__).parent;release=root/'newentry/release'
if (release/'train_direction.py').exists():
    raise FileExistsError('Historical one-time generator; preserve the executed release and use a new directory')
old=root.parent/'2026-09-08_ops_小时级筛选重构/release'
common=(old/'hourly_common.py').read_text(encoding='utf-8')
start=common.index('def load_config(');end=common.index('\ndef data_output',start)
common=common[:start]+'''def load_config(path):
    c=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if c.get('dataset') not in ('drone','llvip'):raise ValueError('Dataset not admitted')
    arms=('N','C1','C2','F-rel') if c['dataset']=='drone' else ('N','L2-box','L2-GT')
    if c.get('arm') not in arms:raise ValueError('Arm not admitted')
    fixed=dict(seed=42,epochs=3,imgsz=640,batch=32,nbs=64,workers=4,amp=True,
        source='paired',optimizer='SGD',lr0=.0001,lrf=1.,warmup_epochs=0.,
        expected_train_images=2048,freeze_bn_running_statistics=True,scope='DIRECTION_FT3_BNFROZEN')
    for k,v in fixed.items():
        if c.get(k)!=v:raise ValueError('Direction config differs: '+k)
    if not 0<=c['kd_coefficient']<=1:raise ValueError('Invalid fixed coefficient')
    if c['arm']=='N' and c['kd_coefficient']!=0:raise ValueError('N must have zero dose')
    return c

'''+common[end:]
common=common.replace("ENDPOINT='HOURLY_SCREEN_FT_E3_LAST_EMA'","ENDPOINT='DIRECTION_FT3_BNFROZEN_LAST_EMA'")
(release/'direction_common.py').write_text(common,encoding='utf-8')
s=(old/'train_hourly.py').read_text(encoding='utf-8').replace('from hourly_common import','from direction_common import')
a=s.index('def build_private(');b=s.index('\ndef run(args):',a)
s=s[:a]+'''def build_private(runtime,criterion,selection,classification,cfg,args):
    legacy=runtime.legacy
    p=Path(cfg['direction_screen']['candidate_source'])
    if p.read_bytes()!=Path(cfg['direction_screen']['approved_candidate_source']).read_bytes():
        raise ValueError('Approved selected-only source differs')
    m=explicit('_direction_selected_only',p)
    from direction_criterion import make_type
    ScreenCriterion=make_type(m.make_api(selection,classification))
    def frozen(path,expected_data,names):
        key='privileged_data_yaml' if str(path)==cfg['teacher'] else 'student_data_yaml'
        if str(path) not in (cfg['teacher'],cfg['reference']) or str(expected_data)!=cfg['paths'][key]:
            raise ValueError('Unexpected frozen binding')
        return legacy.load_frozen(path,cfg['auxiliary_data_identity'][key],names)
    builder=cloned(legacy.build_trainer,EvidenceCriterion=ScreenCriterion,
        DualLabelRGBIRDataset=runtime.TrackedDualLabelRGBIRDataset,load_frozen=frozen)
    return builder,[p]

def install_bn_freeze(trainer):
    import torch
    original=trainer._model_train
    def model_train():
        original()
        for m in trainer.model.modules():
            if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):m.eval()
    trainer._model_train=model_train
    return model_train

def bn_state(model):
    import torch
    return {n+'.'+k:v.detach().cpu().clone() for n,m in model.named_modules()
            if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)
            for k,v in m.named_buffers(recurse=False) if v is not None}

'''+s[b:]
s=s.replace("cfg=load_config(args.config);output=data_output(args.output)","cfg=load_config(args.config);cfg['canary_execution']=bool(args.canary);output=data_output(args.output)")
s=s.replace('HOURLY_SCREEN_CANARY_COMPLETED','DIRECTION_CANARY_COMPLETED').replace('HOURLY_SCREEN_TRAINING_COMPLETED','DIRECTION_TRAINING_COMPLETED').replace('HOURLY_SCREEN_FT','DIRECTION_FT3_BNFROZEN')
s=s.replace('hourly_config.yaml','direction_config.yaml').replace('hourly_training_receipt.json','completion_receipt.json').replace('hourly_failure.json','direction_failure.json')
s=s.replace("dataset='dronevehicle'","dataset=cfg['dataset']")
s=s.replace("(2048,1469,64,32,4)","(2048,cfg['expected_val_images'],64,32,4)")
s=s.replace("trainer._setup_train=setup","trainer._setup_train=setup\n        install_bn_freeze(trainer)\n        frozen_bn={}")
s=s.replace("del payload,initial,expected,current","frozen_bn.update(bn_state(trainer.model))\n            del payload,initial,expected,current")
s=s.replace("torch.cuda.reset_peak_memory_stats();trainer.train()","torch.cuda.reset_peak_memory_stats();trainer.train()\n        bn_after=bn_state(trainer.model)\n        if set(bn_after)!=set(frozen_bn) or any(not torch.equal(bn_after[k],frozen_bn[k]) for k in frozen_bn):\n            raise AssertionError('Frozen BN buffers changed')")
s=s.replace("localization_coefficient=0.,","localization_coefficient=cfg['localization_coefficient'],kd_coefficient=cfg['kd_coefficient'],\n            bn_running_buffers_unchanged=True,bn_buffer_count=len(frozen_bn),bn_affine_trainable=True,")
s=s.replace("if cfg['arm']=='C1':","if False:")
(release/'train_direction.py').write_text(s,encoding='utf-8')
template=yaml.safe_load((old/'configs/drone_C1_s42_FT3.yaml').read_text(encoding='utf-8'))
configs=release/'configs';configs.mkdir(exist_ok=True)
B='/mnt/dataset/yudongfang/projects/RGBT_campaign'
ident=yaml.safe_load((root/'subset_llvip/remote_subset_llvip_v1/checkpoint_identity.json').read_text())
for dataset,arms in [('llvip',('N','L2-box','L2-GT')),('drone',('N','C1','C2','F-rel'))]:
    for arm in arms:
        c=copy.deepcopy(template);c['dataset']=dataset;c['arm']=arm
        c.update(method_id='RGBIR-DIRECTION-'+dataset+'-'+arm,method_identity='DIRECTION_FT3_BNFROZEN',scope='DIRECTION_FT3_BNFROZEN',lr0=.0001,lrf=1.,freeze_bn_running_statistics=True,protocol_status='FROZEN_EXPLORATORY_DIRECTION',calibration_receipt=None,calibration=dict(batches=8,min_nonzero_batches=4,seed=42))
        c['direction_screen']=c.pop('hourly_screen');c['direction_screen'].update(scope=c['scope'],endpoint='DIRECTION_FT3_BNFROZEN_LAST_EMA',comparison_arms=list(arms),coefficient_recalibrated=arm not in ('N','C1'))
        coef=0. if arm=='N' else .09227393550836771 if arm=='C1' else 1.
        c.update(kd_coefficient=coef,classification_coefficient=coef if arm in ('C1','C2','F-rel') else 0.,localization_coefficient=coef if arm.startswith('L2') else 0.)
        if dataset=='llvip':
            base=B+'/artifacts/rgbir_direction_screen_20260908/subset_llvip_v1/'
            c.update(model=ident['visible']['checkpoint']['path'],reference=ident['visible']['checkpoint']['path'],teacher=ident['infrared']['checkpoint']['path'],expected_nc=1,expected_val_images=2406,student_modality='visible')
            c['paths']=dict(student_data_yaml=base+'data_visible.yaml',privileged_data_yaml=base+'data_infrared.yaml',paired_train_mapping=base+'visible_to_infrared_train.json')
            c['auxiliary_data_identity']=dict(student_data_yaml=ident['visible']['data_yaml']['path'],privileged_data_yaml=ident['infrared']['data_yaml']['path'])
            c['evaluation_contract']['expected_val_images']=2406
            c['native_contract_config']=None
        (configs/(dataset+'_'+arm+'_s42_FT3.yaml')).write_text(yaml.safe_dump(c,sort_keys=False),encoding='utf-8')
print('Created new trainer/common and seven frozen template configurations')
