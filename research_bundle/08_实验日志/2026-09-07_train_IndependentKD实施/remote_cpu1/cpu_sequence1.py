import json,os,subprocess,time
from pathlib import Path
b=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_independent_kd_v2_20260907'); release=b/'release_cpu1'; py='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
steps=[('config',[py,str(release/'prepare_configs.py'),'--output',str(b/'configs_draft_v1')]),('unit',[py,'-m','unittest','discover','-s',str(release),'-p','test_*.py','-v']),('package',[py,'-m','pytest',str(release/'reference_package/test_reference_kernels.py'),'-q'])]
for ds in ('drone','llvip'):
 steps.append(('coverage_'+ds,[py,str(release/'coverage_probe.py'),'--config',str(b/'configs_draft_v1'/(ds+'_C1.yaml')),'--roster',str(b/'geometry_frozen_roster.json'),'--output',str(b/('coverage_'+ds+'_attempt1'))]))
results=[]
for name,cmd in steps:
 started=time.time()
 with (b/(name+'_cpu1.log')).open('x') as f:
  result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=release)
 results.append(dict(name=name,returncode=result.returncode,seconds=time.time()-started,command=cmd))
 (b/'cpu1_progress.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results),flush=True)
