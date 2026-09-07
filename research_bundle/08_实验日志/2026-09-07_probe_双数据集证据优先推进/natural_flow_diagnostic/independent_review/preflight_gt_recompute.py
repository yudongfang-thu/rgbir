"""CPU timing/scope check of model-free GT-only gate reconstruction."""
import importlib.util,json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
audit=load('real_natural_audit',HERE/'review_real_natural.py')
configs=load('independent_generator',audit.RELEASE/'prepare_configs.py').configurations()
audit.torch.set_num_threads(1)
results=[]
for key in ('llvip','drone'):
    line=(audit.OLD/('coverage_'+key+'_attempt1')/'natural_batches.jsonl').open(encoding='utf-8').readline()
    cfg=configs[key+'_C1'];t=time.time();counts=audit.gt_recompute(json.loads(line)['images'],cfg['localization'],cfg['expected_nc'])
    results.append(dict(dataset=key,seconds=time.time()-t,counts=counts))
result=dict(status='PASS',cuda_initialized=audit.torch.cuda.is_initialized(),no_model_forward=True,new_hash_computed=False,results=results)
with (HERE/'gt_recompute_preflight.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
