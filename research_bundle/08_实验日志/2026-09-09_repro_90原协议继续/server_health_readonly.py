import datetime
import json
import pathlib
import socket
import subprocess

def command(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return {'exit_code': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    except Exception as e:
        return {'error': repr(e)}

host = socket.gethostname()
root = pathlib.Path('/mnt/dataset/yudongfang/projects/RGBT_campaign') if '94' in host else pathlib.Path('/mnt/dataX/ydf/projects/RGBT_campaign_90')
out = {'checked_at': datetime.datetime.now().astimezone().isoformat(), 'hostname': host, 'root': str(root)}
out['uptime'] = command(['uptime'])
out['gpu'] = command(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu', '--format=csv,noheader'])
out['gpu_processes'] = command(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv,noheader'])
out['disk'] = command(['df', '-h', '/', str(root)])
out['memory'] = command(['free', '-m'])
out['tmux'] = command(['tmux', 'list-sessions'])
out['screen'] = command(['screen', '-ls'])
out['own_processes'] = command(['ps', '-u', str(__import__('os').getuid()), '-o', 'pid,ppid,etime,rss,args', '--sort=-rss'])
out['root_readable'] = root.is_dir()
lease = root / 'runs/.project_resource_leases.json'
out['lease'] = json.loads(lease.read_text()) if lease.is_file() else {'found': False}
out['guard_candidates'] = [str(p) for p in root.glob('*/project_resource_guard.py')]
out['python_candidates'] = [str(p) for p in root.glob('environments/*/bin/python')]
print(json.dumps(out, ensure_ascii=False, indent=2))
