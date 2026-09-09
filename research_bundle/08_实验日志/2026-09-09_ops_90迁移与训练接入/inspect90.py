"""Bounded read-only inventory. Does not import ML frameworks or read secrets."""
import json, os, pathlib, shutil, subprocess
P=pathlib.Path
roots=[P('/mnt/dataX/ydf'), P('/mnt/dataY/ydf'), P('/home/ydf'), P('/zssd/sr')]
out={'roots': [], 'dataset_candidates': [], 'environments': [], 'weight_candidates': [], 'GPU_process_owners': []}
for root in roots:
    if not root.exists(): continue
    try:
        children=list(root.iterdir())
        out['roots'].append({'path': str(root),'resolved':str(root.resolve()),'writable':os.access(root,os.W_OK),
            'children':[{'name':p.name,'directory':p.is_dir(),'writable':os.access(p,os.W_OK)} for p in children if not p.name.startswith('.')]})
    except PermissionError: continue
    if root==P('/zssd/sr'): continue
    for current, ds, fs in os.walk(str(root), followlinks=False):
        depth=len(P(current).relative_to(root).parts)
        ds[:]=[d for d in ds if not d.startswith('.') and d not in ('site-packages','pkgs','node_modules','__pycache__')]
        if depth>=3:ds[:]=[]
        for name in ds+fs:
            q=P(current)/name
            lower=name.lower()
            if 'llvip' in lower or 'drone' in lower or lower=='rgbt_campaign':
                s=q.stat()
                out['dataset_candidates'].append({'path':str(q),'directory':q.is_dir(),'bytes':s.st_size,'mtime':s.st_mtime})
            if lower=='yolo11n.pt' or (lower.endswith('.pt') and any(v in lower for v in ('llvip','drone','infrared'))):
                out['weight_candidates'].append(str(q))
    for pattern in ('miniconda3/envs/*/bin/python','anaconda3/envs/*/bin/python','conda/envs/*/bin/python','envs/*/bin/python'):
        out['environments'] += [str(p) for p in root.glob(pattern)]
try:
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    ids=sorted(set(x.strip() for x in raw.splitlines() if x.strip().isdigit()))
    for pid in ids:
        try:
            val=subprocess.run(['ps','-p',pid,'-o','user=,pid=,ppid=,rss=,comm='],capture_output=True,text=True).stdout.strip()
            out['GPU_process_owners'].append({'pid':int(pid),'ps':val or 'not visible in host proc namespace'})
        except Exception as exc:out['GPU_process_owners'].append({'pid':pid,'error':type(exc).__name__})
except Exception as exc:out['gpu_query_error']=str(exc)
out['tools']={k:shutil.which(k) for k in ('python3','tmux','screen','rsync','tar','unzip','nvidia-smi')}
print(json.dumps(out,ensure_ascii=True,indent=2))
