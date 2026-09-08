"""One-time local source adaptation; no import/forward of a model."""
import argparse,json,shutil
from pathlib import Path
import yaml

def replace(text,old,new):
    if old not in text:raise AssertionError('Missing source fragment: '+old[:100])
    return text.replace(old,new)

def run(source,dest):
    dest.mkdir(exist_ok=True)
    mappings={'direction_common.py':'object_dfl_common.py','train_direction.py':'train_object_dfl.py',
              'calibrate_direction.py':'calibrate_object_dfl.py','evaluate_direction.py':'evaluate_object_dfl.py'}
    rows=[]
    for before,after in mappings.items():
        p=source/before;t=p.read_text(encoding='utf-8')
        for old,new in [('DIRECTION_','OBJECT_DFL_'),('direction_common','object_dfl_common'),
                        ('direction_criterion','object_dfl_criterion'),('train_direction','train_object_dfl'),
                        ('EXPLORATORY_FIXED8_BNFROZEN','OBJECT_DFL_FIXED8_BNFROZEN')]:t=t.replace(old,new)
        if before=='direction_common.py':
            t=replace(t,"if c.get('dataset') not in ('drone','llvip'):","if c.get('dataset') != 'llvip':")
            t=replace(t,"arms=('N','C1','C2','F-rel') if c['dataset']=='drone' else ('N','L2-box','L2-GT')","arms=('N','L3-DFL','L3-GT')")
            t=replace(t,"    return c\n","    validate_method(c)\n    return c\n")
            t+='''\nMETHOD_IDENTITY='OBJECT_DFL_FT3_BNFROZEN'\nMETHOD=dict(temperature=2.0, teacher_anchor='same_reference_index',\n    normalization='coarse_reference_base_before_quality', teacher_transport='object_coordinate_full_mass',\n    gt_target='sqrt_adjacent_bins_normalized', full_support='reject_any_positive_mass_outside_0_15',\n    native_loss_replaced=False, physical_registration_claim=False)\n\ndef validate_method(cfg):\n    if cfg.get('method_identity')!=METHOD_IDENTITY or cfg.get('object_dfl')!=METHOD:\n        raise ValueError('Frozen object DFL method identity differs')\n    if cfg.get('classification_coefficient')!=0:\n        raise ValueError('Object DFL has no classification KD')\n    if cfg.get('localization_coefficient')!=cfg['kd_coefficient']:\n        raise ValueError('Single localization coefficient must equal outer KD dose')\n    if cfg.get('direction_screen',{}).get('scope')!=METHOD_IDENTITY:\n        raise ValueError('Nested method scope differs')\n'''
        elif before=='train_direction.py':
            t=replace(t,"Warm-start N/C0/C1 on a fixed 2048-image subset", "Warm-start N/L3-DFL/L3-GT on a fixed 2048-image subset")
            t=replace(t,"scope='OBJECT_DFL_FT3_BNFROZEN',single_seed=True", "scope='OBJECT_DFL_FT3_BNFROZEN',method_identity=cfg['method_identity'],single_seed=True")
            t=replace(t,"status='HOURLY_SCREEN_FAILED'", "status='OBJECT_DFL_TRAINING_FAILED'")
        elif before=='calibrate_direction.py':
            t=replace(t,"rows=[];arms=['C1','C2','F-rel'] if cfg['dataset']=='drone' else ['L2-box','L2-GT']", "rows=[];arms=['L3-DFL','L3-GT']")
            t=replace(t,"ratios={a:[] for a in arms};coef_c1=.09227393550836771", "ratios={a:[] for a in arms}")
            t=replace(t,"target=coef_c1*norms['C1'] if cfg['dataset']=='drone' and norms['C1'] is not None else .1*nn if cfg['dataset']=='llvip' and nn is not None else None", "target=.1*nn if nn is not None else None")
            start=t.index("        coefficients={'N':0.};blocked={};details={}")
            end=t.index("        write_new(output/'calibration_receipt.json'",start)
            t=t[:start]+'''        coefficients,blocked,details=coefficient_plan(ratios)\n'''+t[end:]
            t=replace(t,"dataset=cfg['dataset'],coefficients=coefficients", "dataset=cfg['dataset'],method_identity=cfg['method_identity'],coefficients=coefficients")
            t=replace(t,"    legacy=runtime.legacy\n", "    legacy=runtime.legacy\n    for module,name in ((runtime,'runtime.py'),(independent_criterion,'independent_criterion.py'),(selection_adapter,'selection_adapter.py'),(classification_logit,'classification_logit.py')):\n        if Path(module.__file__).resolve()!=args.reference_dir.resolve()/name:raise ValueError('Wrong pinned calibration source: '+name)\n    if str(torch.__version__)!=cfg['torch_version'] or str(legacy.ultralytics.__version__)!=cfg['ultralytics_version']:raise ValueError('Pinned calibration runtime differs')\n")
            t=replace(t,"def run(args):",'''def coefficient_plan(ratios):
    details={a:dict(finite_nonzero_batches=len(r),raw_median=statistics.median(r) if r else None,ratios=r) for a,r in ratios.items()}
    r=ratios['L3-DFL'];blocked={}
    if len(r)<4:
        dose=None
        blocked={a:'Fewer than 4/8 finite nonzero DFL gradient ratios' for a in ('L3-DFL','L3-GT')}
    else:
        median=statistics.median(r)
        if not math.isfinite(median) or median<=0:raise ValueError('Invalid DFL median dose')
        dose=min(1.,median)
    return {'N':0.,'L3-DFL':dose,'L3-GT':dose},blocked,details

def run(args):''')
        elif before=='evaluate_direction.py':
            t=replace(t,"ARMS = ('N', 'C1', 'C2', 'F-rel', 'L2-box', 'L2-GT')", "ARMS = ('N', 'L3-DFL', 'L3-GT')")
            t=replace(t,"DATASET_ARMS = {'drone': ('N', 'C1', 'C2', 'F-rel'), 'llvip': ('N', 'L2-box', 'L2-GT')}", "DATASET_ARMS = {'llvip': ARMS}")
            t=replace(t,"POPULATIONS = {'drone': (1469, 22462, 5), 'llvip': (2406, 7879, 1)}", "POPULATIONS = {'llvip': (2406, 7879, 1)}")
            t=replace(t,"    return cfg\n", "    from object_dfl_common import validate_method\n    validate_method(cfg)\n    return cfg\n")
            t=replace(t,"for key in ('arm', 'seed', 'dataset', 'model', 'teacher', 'reference',", "for key in ('arm', 'seed', 'dataset', 'model', 'teacher', 'reference', 'method_identity',")
            t=replace(t,"seed=42, arm=cfg['arm'], dataset=cfg['dataset'], single_seed=True, epochs=3,", "seed=42, arm=cfg['arm'], dataset=cfg['dataset'], method_identity=cfg['method_identity'], single_seed=True, epochs=3,")
        out=dest/after
        with out.open('x',encoding='utf-8',newline='\n') as f:f.write(t)
        rows.append(dict(source=str(p),source_bytes=p.stat().st_size,adapted=after,adapted_bytes=out.stat().st_size,byte_identity=False,adaptation='Independent OBJECT_DFL scope and L3-only branch; see source diff'))
    p=source/'localization_box_v2.py';out=dest/p.name
    if out.exists():raise FileExistsError(out)
    shutil.copyfile(p,out);assert p.read_bytes()==out.read_bytes()
    rows.append(dict(source=str(p),source_bytes=p.stat().st_size,adapted=out.name,byte_identity=True))
    cfgdir=dest/'configs';cfgdir.mkdir(exist_ok=True)
    template=yaml.safe_load((source/'configs/llvip_N_s42_FT3.yaml').read_text(encoding='utf-8'))
    import importlib.util
    spec=importlib.util.spec_from_file_location('_l3_common_config',dest/'object_dfl_common.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for arm in ('N','L3-DFL','L3-GT'):
        c=json.loads(json.dumps(template));dose=0. if arm=='N' else 1.
        c.update(arm=arm,method_id='RGBIR-OBJECT-DFL-llvip-'+arm,method_identity=module.METHOD_IDENTITY,
            description='Fixed prospective LLVIP object-coordinate DFL FT3 screen',scope=module.METHOD_IDENTITY,
            object_dfl=module.METHOD,kd_coefficient=dose,localization_coefficient=dose,classification_coefficient=0.,
            protocol_status='FROZEN_OBJECT_DFL_PROSPECTIVE',calibration_receipt=None)
        c['direction_screen'].update(scope=module.METHOD_IDENTITY,endpoint='OBJECT_DFL_FT3_BNFROZEN_LAST_EMA',comparison_arms=['N','L3-DFL','L3-GT'])
        c['native_contract_config']='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_object_dfl_screen_20260908/release_v1/configs/llvip_native_evaluation.yaml'
        p=cfgdir/('llvip_'+arm+'_s42_FT3.yaml')
        with p.open('x',encoding='utf-8') as f:yaml.safe_dump(c,f,allow_unicode=True,sort_keys=False)
    p=source/'configs/llvip_native_evaluation.yaml';out=cfgdir/p.name
    with out.open('xb') as f:f.write(p.read_bytes())
    with (dest/'WRAPPER_SOURCE_ADAPTATION.json').open('x',encoding='utf-8') as f:
        json.dump(dict(status='PREPARED_NOT_RUN',files=rows,new_GPU=False,new_hash_computed=False,old_sources_modified=False),f,indent=2)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.source,a.output)
