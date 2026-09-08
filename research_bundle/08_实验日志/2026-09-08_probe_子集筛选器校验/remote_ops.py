"""Source-only deployment, pinned CPU preflight, reviewed launch, small-result collection."""
from pathlib import Path
import argparse,base64,io,json,subprocess,tarfile

ROOT=Path(__file__).parent
BASE='/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_subset_e8_check_20260908'
PY='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'

def remote(code):
    p=subprocess.run(['ssh','94',PY,'-'],input=code.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace')+'\n'+p.stdout.decode(errors='replace'))
    return json.loads(p.stdout)

def write_new(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('deploy','pinned','launch','collect'))
    p.add_argument('--version',type=int,default=1);p.add_argument('--attempt',type=int,default=1)
    p.add_argument('--tag');p.add_argument('--tests',default='test_queue_cpu')
    a=p.parse_args();assert a.version>0 and a.attempt>0
    release=BASE+'/release_v'+str(a.version);out=BASE+'/attempt'+str(a.attempt)
    if a.mode=='deploy':
        files={p.relative_to(ROOT/'release').as_posix():base64.b64encode(p.read_bytes()).decode()
            for p in sorted((ROOT/'release').rglob('*')) if p.is_file() and '__pycache__' not in p.parts
            and p.suffix in ('.py','.json','.yaml','.md')}
        code="""from pathlib import Path
import base64,json,ast
r=Path(%r);r.mkdir(parents=True,exist_ok=False);rows=[]
for name,data in %r.items():
 p=r/name
 if r not in p.resolve().parents:raise ValueError('Outside release')
 b=base64.b64decode(data);p.parent.mkdir(parents=True,exist_ok=True)
 if p.suffix=='.py':ast.parse(b.decode('utf-8-sig'))
 with p.open('xb') as f:f.write(b)
 if p.read_bytes()!=b:raise ValueError('Source differs')
 rows.append(dict(relative_path=name,bytes=len(b),byte_exact=True))
v=dict(status='DEPLOYED_SOURCE_ONLY',release=str(r),files=rows,new_hash_computed=False)
(r/'deployment.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
"""%(release,files)
        r=remote(code);write_new(ROOT/('DEPLOYMENT_v'+str(a.version)+'.json'),r)
        print(r['status'],len(r['files']))
    elif a.mode=='pinned':
        modules=a.tests.split(',');assert modules and all(x.startswith('test_') and x.replace('_','').isalnum() for x in modules)
        code="""from pathlib import Path
import os,sys,json,unittest,io,time,contextlib,subprocess
r=Path(%r);os.chdir(r);sys.path.insert(0,str(r));os.environ['CUDA_VISIBLE_DEVICES']=''
if (r/'pinned_cpu.json').exists():raise FileExistsError('Preserve receipt')
start=time.time();b=io.StringIO()
with contextlib.redirect_stdout(b),contextlib.redirect_stderr(b):
 suite=unittest.defaultTestLoader.loadTestsFromNames(%r)
 result=unittest.TextTestRunner(stream=b,verbosity=2).run(suite)
wrapper=subprocess.run([sys.executable,'test_subset_cpu.py','--reference-dir','/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_independent_kd_v2_20260907/release_gpu5','--output',str(r/'pinned_subset_cpu.json')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
b.write(wrapper.stdout);wr=json.loads((r/'pinned_subset_cpu.json').read_text())
cli={}
for name in ('train_short_screen.py','evaluate_short_screen.py','run_subset_queue.py'):
 p=subprocess.run([sys.executable,name,'--help'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 cli[name]=p.returncode;b.write(p.stdout)
ok=result.wasSuccessful() and result.testsRun==4 and wrapper.returncode==0 and wr['status']=='PASS' and wr['tests']==8 and not wr['cuda_initialized'] and all(v==0 for v in cli.values())
v=dict(status='PASS' if ok else 'FAIL',tests=result.testsRun+wr['tests'],failures=len(result.failures),errors=len(result.errors),modules=%r,wrapper=wr,
 cli=cli,GPU_used=False,new_hash_computed=False,seconds=time.time()-start)
with (r/'pinned_cpu.json').open('x') as f:json.dump(v,f,indent=2)
with (r/'pinned_cpu.log').open('x') as f:f.write(b.getvalue())
print(json.dumps(dict(receipt=v,log=b.getvalue())))
"""%(release,modules,modules)
        r=remote(code);write_new(ROOT/('PINNED_CPU_v'+str(a.version)+'.json'),r);print(json.dumps(r['receipt']))
        if r['receipt']['status']!='PASS':raise RuntimeError('Actual pinned CPU failed; preserve and inspect')
    elif a.mode=='launch':
        review=json.loads((ROOT/'independent_review/SOURCE_REVIEW.json').read_text(encoding='utf-8'))
        assert review['status']=='READY_FOR_GPU_PREFLIGHT'
        cpu=json.loads((ROOT/('PINNED_CPU_v'+str(a.version)+'.json')).read_text(encoding='utf-8'))['receipt']
        assert cpu['status']=='PASS' and cpu['tests']>0 and cpu['GPU_used'] is False
        bound={}
        for row in review['reviewed_source_files']:
            name=row['relative_path'];source=ROOT/'release'/name;snapshot=Path(row['snapshot_path'])
            assert ROOT.resolve() in source.resolve().parents and ROOT.resolve() in snapshot.resolve().parents
            b=source.read_bytes();assert b==snapshot.read_bytes();bound[name]=base64.b64encode(b).decode()
        required={'screen_common.py','train_short_screen.py','evaluate_short_screen.py','run_subset_queue.py','budget.py','validated_amp_prior.json',
            'configs/drone_N_s42_E8.yaml','configs/drone_C0_s42_E8.yaml'}
        assert required.issubset(bound)
        screen='subset_e8_check_s42_a'+str(a.attempt)
        command='exec '+PY+' '+release+'/run_subset_queue.py --release-dir '+release+' --output '+out+' > '+release+'/queue_launch_a'+str(a.attempt)+'.log 2>&1'
        code="""from pathlib import Path
import json,base64,subprocess
r=Path(%r);out=Path(%r)
if out.exists():raise FileExistsError('Preserve attempt')
for name,data in %r.items():
 if (r/name).read_bytes()!=base64.b64decode(data):raise ValueError('Remote reviewed source changed: '+name)
cpu=json.loads((r/'pinned_cpu.json').read_text())
if cpu['status']!='PASS' or cpu['tests']<=0 or cpu['GPU_used']:raise ValueError('Pinned CPU missing')
subprocess.run(['screen','-dmS',%r,'bash','-lc',%r],check=True)
print(json.dumps(dict(status='REVIEWED_SUBSET_QUEUE_STARTED',release=str(r),output=str(out),screen=%r,new_hash_computed=False)))
"""%(release,out,bound,screen,command,screen)
        r=remote(code);write_new(ROOT/('QUEUE_LAUNCH_attempt'+str(a.attempt)+'.json'),r);print(r['status'],r['output'])
    else:
        assert a.tag and all(c.isalnum() or c in '_-' for c in a.tag)
        dest=ROOT/('evidence_'+a.tag);assert not dest.exists()
        code="""from pathlib import Path
import json,base64
r=Path(%r);rows=[]
for p in sorted(r.rglob('*')):
 if not p.is_file() or any(x in p.parts for x in ('weights','source_copies','flow_reference')):continue
 if p.suffix not in ('.json','.jsonl','.yaml','.csv','.gz','.txt'):continue
 if p.stat().st_size>16000000:continue
 b=p.read_bytes();rows.append(dict(path=p.relative_to(r).as_posix(),bytes=len(b),data=base64.b64encode(b).decode()))
print(json.dumps(dict(status='COLLECTED_SMALL_EVIDENCE',source=str(r),files=rows,private_tensor_files_collected=False,new_hash_computed=False)))
"""%out
        r=remote(code);dest.mkdir()
        for row in r['files']:
            target=dest/row['path'];assert dest.resolve() in target.resolve().parents
            target.parent.mkdir(parents=True,exist_ok=True);b=base64.b64decode(row.pop('data'))
            with target.open('xb') as f:f.write(b)
            assert target.read_bytes()==b
        write_new(dest/'collection_receipt.json',r);print(r['status'],len(r['files']),sum(x['bytes'] for x in r['files']))

if __name__=='__main__':main()
