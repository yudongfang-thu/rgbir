"""Freeze the authorized C-attribution fallback after completed, bound canaries."""
import json
from pathlib import Path
import subprocess

PY='/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'
SCRIPT=r'''
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,subprocess,yaml,torch
from pathlib import Path
b=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_task_conditional_v1_20260907')
r=b/'release_v8';py=PYTHON
out=b/'c_controls_full_v1';out.mkdir(exist_ok=False)
eq=json.loads((b/'criterion_equivalence_s42_attempt2/completion_receipt.json').read_text())
assert eq['status']=='ACCEPTED' and eq['legacy_C_N_equivalence']
stream=json.loads((b/'canary_initialization_stream_audit_v1.json').read_text())
assert stream['passed'] and len(stream['checks'])==7
for arm in ['c_shuffled','c_same_modal']:
 canary=b/f'canary_drone_{arm}_s42_attempt1'
 done=json.loads((canary/'completion_receipt.json').read_text())
 bound=json.loads((canary/'run_evidence/run_receipt.json').read_text())
 assert done['status']=='canary_completed' and done['optimizer_updates']>=24
 assert bound['terminal_status']=='COMPLETED' and bound['seed']==42
 assert any(g['kd_score_gradient_l2']>0 for g in done['gradient_checks'])
 assert all(not g['teacher_has_grad'] and not g['reference_has_grad'] for g in done['gradient_checks'])
 assert max(done['resources']['per_gpu_peak_vram_mib'].values())<8300
 assert done['resources']['peak_rss_mib']<32768
 cfg=yaml.safe_load((canary/'protocol_config.yaml').read_text())
 if arm=='c_same_modal':
  batch=torch.load(canary/'first_batch.pt',map_location='cpu',weights_only=True)
  assert torch.equal(batch['img'],batch['strong_img'])
  runtime=json.loads((canary/'runtime_ready.json').read_text())
  assert runtime['privileged_ir_labels_used_by_kd'] is False
 acceptance={'status':'ACCEPTED','arms':[arm],'minimum_successful_updates':done['optimizer_updates'],
   'nonzero_C_gradient':True,'dataset':cfg['dataset'],'teacher':cfg['teacher'],'reference':cfg['reference'],
   'legacy_C_N_equivalence':True,'canary_receipt':str(canary/'completion_receipt.json'),
   'bound_canary_receipt':str(canary/'run_evidence/run_receipt.json'),
   'criterion_equivalence_receipt':str(b/'criterion_equivalence_s42_attempt2/completion_receipt.json'),
   'loader_and_initialization_audit':str(b/'canary_initialization_stream_audit_v1.json'),
   'derangement_roster':cfg.get('derangement_roster'),'resources':done['resources'],
   'scope':'C attribution only; no formal L authorization'}
 ap=out/(arm+'_acceptance.json');ap.write_text(json.dumps(acceptance,indent=2))
 cfg.update(protocol_status='FROZEN',formal_training_authorized=True,canary_acceptance=str(ap),
            method_id='RGBIR-OEV1-C-CONTENT-CONTROLS-v1',description='Unchanged OEv1 C content attribution; localization disabled',
            seed=42,localization_coefficient=0.0)
 cp=out/(arm+'_s42.yaml');cp.write_text(yaml.safe_dump(cfg,sort_keys=False))
 run=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/runs/rgbir_task_conditional_c_attribution_20260907')/f'full_{arm}_s42_attempt1'
 jobs=[{'id':f'oev1_content_{arm}_s42_a1','kind':'train','formal':True,'vram_mib':8300,'rss_mib':32768,
        'requires_profile':str(canary/'completion_receipt.json'),
        'command':[py,str(r/'train_task_conditional.py'),'--config',str(cp),'--arm',arm,'--seed','42','--output',str(run)]},
       {'id':f'oev1_content_{arm}_eval_s42_a1','kind':'eval','vram_mib':8300,'rss_mib':32768,
        'command':[py,str(r/'evaluate_task_conditional.py'),'--config',str(cp),'--run',str(run)]}]
 mp=out/(arm+'_manifest.json');mp.write_text(json.dumps({'branch':'authorized_C_attribution_when_L_not_admissible','jobs':jobs},indent=2))
 screen='oev1content_'+arm.removeprefix('c_')+'_42'
 subprocess.run(['screen','-L','-Logfile',str(out/(arm+'_screen.log')),'-dmS',screen,py,str(r/'resource_dispatch.py'),
                 '--manifest',str(mp),'--output',str(out/(arm+'_dispatch_a1'))],check=True,env={k:v for k,v in os.environ.items() if k!='CUDA_VISIBLE_DEVICES'})
 print(json.dumps({'status':'DISPATCHED','arm':arm,'screen':screen,'formal_run':str(run),'canary_acceptance':str(ap)}))
'''.replace('PYTHON',repr(PY))

if __name__=='__main__':
    result=subprocess.run(['ssh','-o','BatchMode=yes','94',PY,'-'],input=SCRIPT.encode(),capture_output=True)
    here=Path(__file__).resolve().parent
    (here/'dispatch_stdout.jsonl').write_bytes(result.stdout)
    (here/'dispatch_stderr.txt').write_bytes(result.stderr)
    print(result.stdout.decode(errors='replace'))
    if result.returncode:raise RuntimeError(result.stderr.decode(errors='replace'))
