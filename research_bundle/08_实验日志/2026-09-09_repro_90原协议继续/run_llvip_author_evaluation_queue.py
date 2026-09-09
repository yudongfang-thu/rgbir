"""Two author checkpoints, each with measured canary then original full evaluation."""
import json
import math
import os
from pathlib import Path
import subprocess
import time

root=Path('/mnt/dataX/ydf/projects/RGBT_campaign_90')
out=root/'artifacts/original_repro_continue_20260909_attempt1'
python=str(root/'environments/cft90/bin/python')
source=root/'external_reproductions/llvip_author_baseline/author_source/yolov5'
wrapper=out/'llvip_author_baseline.py'
os.environ.update(RGBIR90_PROJECT_ROOT=str(root), YOLOV5_CONFIG_DIR=str(root/'cache/llvip_yolov5'),
    TMPDIR=str(root/'cache/tmp'), PYTHONDONTWRITEBYTECODE='1')

def write_new(path,data):
    with path.open('x') as f: json.dump(data,f,indent=2);f.write('\n')

protocols={}
for modality,reference in [('visible',[90.8,51.9,50.0]),('infrared',[94.6,72.2,61.9])]:
    identity_path=out/f'llvip_{modality}_identity_attempt1/identity.json'
    identity=json.loads(identity_path.read_text())
    assert identity['status']=='IDENTITY_READ_CPU' and identity['names']==['person']
    assert identity['parameters']==46631350 and not identity['cuda_initialized']
    assert Path(identity['weights']).name==f'yolov5_{modality}.pt'
    protocol={'status':'FROZEN_BEFORE_AP','identity':'PAPER-RECONSTRUCTED','modality':modality,
        'weights':identity['weights'],'identity_receipt':str(identity_path),
        'data':str(root/f'data_author_protocol/llvip_baseline_attempt1/{modality}/data.yaml'),
        'expected_roster':str(root/'data_author_protocol/llvip_baseline_attempt1/test_roster.json'),
        'device':'0','imgsz':1280,'batch_size':32,'conf_thres':.001,'iou_thres':.6,
        'half':False,'augment':False,'save_hybrid':False,'label_version':'previous',
        'expected_images':3463,'expected_gt':7931,'paper_reference_percent':dict(zip(['AP50','AP75','AP50_95'],reference)),
        'paper_reference_caveat':'v1 paper describes earlier 16836/70-30 split; public author release after removals uses15488 pairs. Not exact historical training/split reconstruction.',
        'source_revision':(out/'llvip_source_revision.txt').read_text().strip(),
        'canary_receipt':str(out/f'llvip_{modality}_canary_attempt1/receipt.json')}
    path=out/f'llvip_{modality}_protocol.json'
    write_new(path,protocol)
    protocols[modality]=path

def run(mode,modality,reserve):
    job=f'llvip_author_{modality}_{mode}_20260909'
    attempt=out/f'llvip_{modality}_{"canary" if mode=="canary" else "full"}_attempt1'
    command=[python,'-B','-u',str(root/'tools/project_resource_guard.py'),'--lease-file',str(root/'runs/.project_resource_leases.json'),
        'run','--job-id',job,'--kind','eval','--expected-vram-mib',str(reserve),'--expected-rss-mib','32768','--free-safety-mib','2048',
        '--',python,'-B','-u',str(wrapper),mode,'--source',str(source),'--protocol',str(protocols[modality]),'--output',str(attempt)]
    write_new(out/(job+'_command.json'),{'command':command,'reserved_vram_mib':reserve})
    with (out/(job+'.log')).open('x') as log:
        started=time.perf_counter()
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    write_new(out/(job+'_exit.json'),{'exit_code':result.returncode,'wall_seconds':time.perf_counter()-started})
    if result.returncode:
        raise RuntimeError(f'{job} failed; preserve attempt and diagnose before further GPU stages')
    return json.loads((attempt/'receipt.json').read_text())

receipts={}
for modality in ('visible','infrared'):
    measured=run('canary',modality,17152)
    assert measured['status']=='CANARY_PASS'
    nvml=max(measured['observed_lease_resources']['per_gpu_peak_vram_mib'].values())
    peak=max(nvml,measured['torch_max_reserved_mib']+512)
    reserve=math.ceil((peak+1024)/256)*256
    # Reservation must fit the existing strict 70% project cap, never override it.
    if reserve>=.70*24576:
        raise RuntimeError(f'{modality} canary does not leave 1GiB measured reservation margin under project cap')
    receipts[modality]=run('evaluate',modality,reserve)
write_new(out/'llvip_author_pair_completed.json',{'status':'COMPLETED','receipts':receipts})
print('LLVIP_AUTHOR_BOTH_FULL_EVALUATIONS_COMPLETED',flush=True)
