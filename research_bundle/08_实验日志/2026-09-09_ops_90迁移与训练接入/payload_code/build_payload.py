"""Local, bounded packaging only. No SSH, CUDA, hashing, or original-file edits."""
from pathlib import Path
import json
import shutil
import tarfile

OUT = Path(__file__).resolve().parent
WS = OUT.parents[2]
ENG = WS / '03_现行工程/SpaceNet6_OTD_official_reproduction'
FAST = WS / '08_实验日志/2026-09-08_ops_C1训练路径提速'
ROOT90 = '/mnt/dataX/ydf/projects/RGBT_campaign_90'
PY90 = '/mnt/dataX/ydf/projects/RGBT_campaign_90/environments/rgbir90/bin/python'
OLDREPO = '/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction'
OLDCAMP = '/mnt/dataset/yudongfang/projects/RGBT_campaign'
changes = []

def copy(source, relative):
    target = OUT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    shutil.copyfile(source, target)
    changes.append({'source': str(source), 'destination': relative, 'operation': 'copy'})
    return target

def write(relative, text):
    path = OUT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(text)

def edit(relative, old, new):
    path = OUT / relative
    source = path.read_text(encoding='utf-8')
    if old not in source:
        raise ValueError('Missing exact edit anchor in ' + relative + ': ' + old[:80])
    write(relative, source.replace(old, new))
    changes.append({'destination': relative, 'operation': 'text-edit', 'anchor': old[:120]})

archive = WS / '08_实验日志/2026-09-07_train_IndependentKD实施/release_gpu5.tar.gz'
copy(archive, 'provenance/release_gpu5.original.tar.gz')
release = OUT / 'release_gpu5'
release.mkdir(exist_ok=False)
with tarfile.open(archive, 'r:gz') as bundle:
    for member in bundle.getmembers():
        target = (release / member.name).resolve()
        if release.resolve() not in target.parents or not (member.isfile() or member.isdir()):
            raise ValueError('Unexpected archive member: ' + member.name)
    bundle.extractall(release)

for name in ('project_resource_guard.py', 'write_jstars_run_receipt.py', 'validate_jstars_data_contract.py'):
    copy(ENG / 'tools' / name, 'tools/' + name)
write('tools/__init__.py', '')
copy(ENG / 'configs/research/jstars_dataset_exposure_v1.json', 'configs/research/jstars_dataset_exposure_v1.json')
for src, dst in [
    ('fast_candidate/batched_selection_v1.py', 'fast_candidate/batched_selection_v1.py'),
    ('native_fast/native_fast.py', 'native_fast/native_fast.py'),
    ('benchmark_review/cadence106.py', 'benchmark_review/cadence106.py'),
    ('production_entry/train_c1_fast.py', 'production_entry/train_c1_fast.py'),
    ('production_entry/dispatch_single_formal.py', 'dispatch_single_formal.py'),
]:
    copy(FAST / src, dst)

# All modifications are to derived payload copies. The archive stays untouched.
for path in list(release.rglob('*.py')) + list(release.rglob('*.yaml')) + [OUT/'dispatch_single_formal.py']:
    source = path.read_text(encoding='utf-8')
    updated = source.replace(OLDREPO + '/environments/sn6-int8-kd/bin/python', PY90)
    updated = updated.replace(OLDREPO, ROOT90).replace(OLDCAMP, ROOT90)
    if source != updated:
        write(path.relative_to(OUT).as_posix(), updated)
        changes.append({'destination': path.relative_to(OUT).as_posix(), 'operation': '94-to-90-path-only'})

edit('dispatch_single_formal.py', "PY=str(REPO/'environments/sn6-int8-kd/bin/python')", "PY=os.environ.get('RGBIR90_PYTHON', '" + PY90 + "')")
edit('dispatch_single_formal.py', "REPO=Path('" + ROOT90 + "')", "REPO=Path(os.environ.get('RGBIR90_PROJECT_ROOT', '" + ROOT90 + "'))")
edit('tools/project_resource_guard.py', 'REPO_ROOT = Path(__file__).resolve().parents[1]', "REPO_ROOT = Path(os.environ.get('RGBIR90_PROJECT_ROOT', str(Path(__file__).resolve().parents[1])))")
edit('tools/project_resource_guard.py', 'DEFAULT_LEASE_FILE = REPO_ROOT / "runs" / ".project_resource_leases.json"', 'DEFAULT_LEASE_FILE = REPO_ROOT / "runs" / ".project_resource_leases.json"  # One pool for all 90 payload releases.')
edit('release_gpu5/train_independent.py', 'def validate_execution(cfg, formal=False):\n', "def validate_execution(cfg, formal=False):\n    if cfg.get('port90', {}).get('status') != 'READY':\n        raise ValueError('PORT90_BLOCKED: bind actual 90 inputs and complete the local readiness work first')\n")

edit('benchmark_review/cadence106.py', 'import argparse, functools, hashlib, importlib.util, json, math, os', 'import argparse, functools, importlib.util, json, math, os')
edit('benchmark_review/cadence106.py', "if os.name!='posix' or not str(output).startswith('/mnt/dataset/yudongfang/'):", "project_root=Path(os.environ.get('RGBIR90_PROJECT_ROOT', '" + ROOT90 + "')).resolve()\n    if os.name!='posix' or project_root not in output.parents:")
edit('benchmark_review/cadence106.py', "sha256=hashlib.sha256(raw).hexdigest()", "size_bytes=len(raw),content_binding='preserved_source_copy_no_new_hash'")
edit('benchmark_review/cadence106.py', "weights_hashed=False", "weights_hashed=False,new_hashes_computed=False,server='90'")

entry = OUT/'production_entry/train_c1_fast.py'
source = entry.read_text(encoding='utf-8')
start, end = source.index('def sha(path):'), source.index('\ndef run(args):')
verification = '''def source_record(path):
    path=Path(path); info=path.stat()
    return dict(path=str(path),size_bytes=info.st_size,mtime_ns=info.st_mtime_ns,hash_computed=False)

def verify_inputs(plan_path, seed, output):
    plan=json.loads(Path(plan_path).read_text(encoding='utf-8'))
    if plan.get('status') != 'PORT90_READY' or plan.get('server') != '90':
        raise ValueError('PORT90_BLOCKED: template has no 90 input/resource/readiness acceptance')
    if seed not in (0,42,123):raise ValueError('Unregistered seed')
    cell=plan['cells'][str(seed)]
    if Path(output).resolve()!=Path(cell['output']).resolve():raise ValueError('Unregistered output attempt')
    if Path(output).exists():raise FileExistsError(output)
    for row in plan['source_bindings']:
        if Path(row['path']).read_bytes()!=Path(row['accepted_copy']).read_bytes():
            raise ValueError('Changed source/config binding: '+row['path'])
    if not plan['source_bindings']:raise ValueError('No byte-bound 90 source/config copies')
    candidate=json.loads(Path(plan['cadence_receipt']).read_text())
    if candidate.get('server')!='90' or candidate.get('status')!='CADENCE106_COMPLETED':
        raise ValueError('Require actual 90 cadence evidence; 94 evidence is not admission')
    if candidate.get('mode')!='candidate' or candidate.get('actual_optimizer_calls',0)<24:
        raise ValueError('Incomplete candidate optimizer exposure')
    if not candidate.get('final_student_ema_finite') or candidate['candidate_execution']['fallback_batches']!=0:
        raise ValueError('Candidate execution not accepted')
    return plan,cell
'''
source = source[:start] + verification + source[end:]
source = source.replace('import argparse, hashlib,', 'import argparse,')
source = source.replace("if os.name!='posix' or not str(args.output.resolve()).startswith('/mnt/dataset/yudongfang/'):", "project_root=Path(os.environ.get('RGBIR90_PROJECT_ROOT', '"+ROOT90+"')).resolve()\n    if os.name!='posix' or project_root not in args.output.resolve().parents:")
source = source.replace('sha256=sha(target)', 'source_metadata=source_record(target)')
source = source.replace("candidate_sha256=sha(plan['candidate_source'])", "candidate_source_metadata=source_record(plan['candidate_source'])")
source = source.replace("old_readiness_scope='original science/calibration; not new implementation acceptance'", "readiness_scope='90 deployment only; no inherited 94 admission'")
source = source.replace("new_implementation_admission=plan['admission_inputs']", "new_implementation_admission=plan['source_bindings']")
source = source.replace("training.validate_execution(cfg,formal=True)", "ready=json.loads(Path(cfg['readiness_receipt']).read_text())\n    if ready.get('server')!='90':raise ValueError('Original 94 readiness cannot admit a 90 run')\n    training.validate_execution(cfg,formal=True)")
write('production_entry/train_c1_fast.py', source)
changes.append({'destination':'production_entry/train_c1_fast.py','operation':'90-only blocked template gate; byte-copy bindings replace new hashes; preserve trainer and original readiness validation'})

cfg=(FAST/'production_entry/configs/C1_s42.yaml').read_text(encoding='utf-8')
cfg=cfg.replace(OLDREPO,ROOT90).replace(OLDCAMP,ROOT90)
rows=[]
for line in cfg.splitlines():
    if line.startswith(('calibration_receipt:', 'canary_acceptance:', 'readiness_receipt:')):
        line=line.split(':')[0]+': null'
    elif line=='protocol_status: FROZEN':line='protocol_status: TEMPLATE_BLOCKED'
    elif line=='formal_training_authorized: true':line='formal_training_authorized: false'
    elif line.startswith('model:'):line='model: '+ROOT90+'/weights/pretrained/yolo11n.pt'
    elif line.startswith('teacher:'):line='teacher: '+ROOT90+'/inputs/TO_BIND_IR42/weights/last.pt'
    elif line.startswith('reference:'):line='reference: '+ROOT90+'/inputs/TO_BIND_RGB42/weights/last.pt'
    rows.append(line)
cfg='\n'.join(rows)+'''\nport90:
  status: BLOCKED
  server: '90'
  teacher_reference_available: false
  original_coefficient_scope: historical_Drone_C1_value_not_90_acceptance
  measured_vram_mib: null
  measured_rss_mib: null
  cadence_receipt: null
  blockers:
    - Bind generic initialization and actual teacher/reference weights with their args.yaml
    - Bind 90 train/dev YAML and complete pair mapping to the frozen data protocol
    - Measure 90 resource peaks and wall time with the actual configuration
    - Produce 90 source/input/readiness bindings; no 94 admission is inherited
'''
write('configs/C1_s42.template.yaml',cfg)
write('configs/N_s42.template.yaml',cfg.replace('arm: C1','arm: N').replace('classification_coefficient: 0.09227393550836771','classification_coefficient: 0.0').replace('RGBIR-INDEPENDENT-KD-v2-C1','RGBIR-INDEPENDENT-KD-v2-N'))
write('configs/C0_s42.template.yaml',cfg.replace('arm: C1','arm: C0').replace('classification_coefficient: 0.09227393550836771','classification_coefficient: 0.1').replace('RGBIR-INDEPENDENT-KD-v2-C1','RGBIR-INDEPENDENT-KD-v2-C0'))
plan={'status':'BLOCKED_MISSING_90_INPUTS_AND_MEASUREMENTS','server':'90','implementation_id':'RGBIR-C1-BATCHED-SELECTION-v1-PORT90-20260909','reference_dir':ROOT90+'/release_gpu5','candidate_source':ROOT90+'/fast_candidate/batched_selection_v1.py','cadence_receipt':None,'source_bindings':[],'cells':{'42':{'config':ROOT90+'/configs/C1_s42.template.yaml','output':ROOT90+'/runs/TO_CREATE_C1_s42_attempt1'}},'formal_training_started':False,'inherited_94_admission':False}
write('production_entry/production_plan90.template.json',json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
write('job.template.json',json.dumps({'status':'BLOCKED','jobs':[{'id':'TO_BIND_90_canary_attempt1','kind':'train','formal':False,'vram_mib':None,'rss_mib':None,'command':[],'note':'Set bounded canary ceilings from current free memory; these nulls are not measured peaks.'}]},indent=2)+'\n')
write('PACKAGING_RECEIPT.json',json.dumps({'status':'PAYLOAD_PREPARED_NOT_DEPLOYED','server_target':'90','target_root':ROOT90,'python_target':PY90,'new_hashes_computed':False,'remote_commands_executed':False,'gpu_work_started':False,'inherited_94_admission':False,'changes':changes},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'PAYLOAD_PREPARED','directory':str(OUT),'files':sum(p.is_file() for p in OUT.rglob('*')),'new_hashes_computed':False},ensure_ascii=False))
