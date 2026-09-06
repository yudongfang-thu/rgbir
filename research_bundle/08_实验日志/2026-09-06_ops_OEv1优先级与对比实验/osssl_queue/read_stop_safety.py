from pathlib import Path
import json,hashlib,datetime,os,zipfile
base=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign')
leasepath=Path('/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/runs/.project_resource_leases.json')
procs={}
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
        c=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        s=(p/'stat').read_text().rsplit(')',1)[1].split()
        procs[int(p.name)]={'pid':int(p.name),'state':s[0],'ppid':int(s[1]),'pgrp':int(s[2]),'session':int(s[3]),'starttime_ticks':int(s[19]),'cmd':c}
    except (FileNotFoundError,PermissionError):pass
roots={p for p,d in procs.items() if '/artifacts/osssl_ir_20260906/worker' in d['cmd'] and not d['cmd'].startswith('SCREEN')}
roots|={p for p,d in procs.items() if 'project_resource_guard.py run --job-id osssl-ir-' in d['cmd']}
members=set(roots)
while True:
    more={p for p,d in procs.items() if d['ppid'] in members}-members
    if not more:break
    members|=more
out={'captured_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'processes':[procs[p] for p in sorted(members)],'leases':json.loads(leasepath.read_text()),'checkpoints':[]}
for p in (base/'runs/osssl_ir_20260906/sar_only_rgb_s0_e200/weights').glob('*.pt'):
    b=p.read_bytes()
    z=zipfile.ZipFile(p)
    out['checkpoints'].append({'path':str(p),'size':len(b),'sha256':hashlib.sha256(b).hexdigest(),'mtime':datetime.datetime.fromtimestamp(p.stat().st_mtime,datetime.timezone.utc).isoformat(),'zip_crc_first_bad_member':z.testzip()})
print(json.dumps(out))
