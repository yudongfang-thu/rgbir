"""Local static check then SSH-streamed CPU-only tests, no server files or GPU work."""
from pathlib import Path
import ast,hashlib,json,subprocess,datetime
here=Path(__file__).resolve().parent
before=(here/'source_baseline/train_object_evidence.py').read_text(encoding='utf-8')
after=(here/'code/train_object_evidence.py').read_text(encoding='utf-8')
expected=before.replace("config=self.evidence_cfg, arm='paired', seed=self.cfg['seed']+self.calls)","config=self.evidence_cfg, arm=('paired' if self.arm == 'weight0' else self.arm), seed=self.cfg['seed']+self.calls)").replace("parser.add_argument('--arm',choices=['paired','weight0'],required=True)","parser.add_argument('--arm',choices=['paired','weight0','paired_random'],required=True)")
assert after==expected
unchanged={p.name:p.read_bytes()==(here/'code'/p.name).read_bytes() for p in (here/'source_baseline').iterdir() if p.name!='train_object_evidence.py'}
assert all(unchanged.values())
for p in (here/'code').glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
sources={n:(here/'code'/n).read_text(encoding='utf-8') for n in ('object_evidence_loss.py','test_object_evidence_loss.py','test_random_control.py','train_object_evidence.py')}
sources['baseline_trainer']=before
script="""import os,sys,types,unittest,json
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
sources=SOURCES_JSON
for filename in ('object_evidence_loss.py','test_object_evidence_loss.py','test_random_control.py'):
    name=filename[:-3]
    module=types.ModuleType(name)
    module.__file__='<streamed-cpu-test>/'+filename
    module.EMBEDDED_SOURCES=sources
    sys.modules[name]=module
    exec(compile(sources[filename],module.__file__,'exec'),module.__dict__)
import torch
torch.set_num_threads(1)
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[name]) for name in ('test_object_evidence_loss','test_random_control'))
r=unittest.TextTestRunner(verbosity=2).run(suite)
print(json.dumps({'tests':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'passed':r.wasSuccessful(),'torch':str(torch.__version__),'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'cuda_initialized':torch.cuda.is_initialized()}))
raise SystemExit(0 if r.wasSuccessful() else 1)
""".replace('SOURCES_JSON',repr(sources))
response=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','94','/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python','-'],input=script.encode(),capture_output=True)
(here/'cpu_tests_stdout.log').write_bytes(response.stdout)
(here/'cpu_tests_stderr.log').write_bytes(response.stderr)
record={'captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'static_only_two_trainer_changes':True,'unchanged_files':unchanged,'returncode':response.returncode,'gpu_jobs_started':0,'server_files_written':0,'stdout':response.stdout.decode(errors='replace')}
(here/'CPU_VALIDATION.json').write_bytes((json.dumps(record,ensure_ascii=False,indent=2)+'\n').encode())
print(response.stderr.decode(errors='replace'))
print(response.stdout.decode(errors='replace'))
raise SystemExit(response.returncode)
