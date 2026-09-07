"""Executed on 94 by root via pinned Python stdin; no GPU initialized."""
from pathlib import Path
import sys,importlib.util,json,yaml
r=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_independent_kd_v2_20260907/release_gpu5')
sys.path.insert(0,str(r))
import coverage_probe,runtime,selection_adapter,localization_loss,diagnose_opportunities,torch
s=importlib.util.spec_from_file_location('natural_flow_frozen_prepare_configs',r/'prepare_configs.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
result={}
for d in ['llvip','drone']:
    cfg=yaml.safe_load((r.parent/'configs_draft_v1'/(d+'_C1.yaml')).read_text());can=m.configurations()[d+'_C1']
    keys=['teacher','reference','paths','imgsz','batch','workers','nbs','seed','evidence','localization','augmentation','teacher_cache_images']
    result[d]={'config_equal':all(cfg[k]==can[k] for k in keys),'nc':cfg['expected_nc'],'cuda_initialized':torch.cuda.is_initialized()}
print(json.dumps({'status':'PASS' if all(v['config_equal'] for v in result.values()) else 'FAIL','models':result,'generator':str(m.__file__),'torch':str(torch.__version__),'ultralytics':runtime.legacy.ultralytics.__version__}))
