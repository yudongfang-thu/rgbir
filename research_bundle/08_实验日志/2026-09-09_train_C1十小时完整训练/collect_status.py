"""Read-only bounded status capture for the finite C1 campaign."""
import datetime
import json
from pathlib import Path
import subprocess

ROOT = Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/c1_budget10h_20260909')


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    result = dict(observed_at=datetime.datetime.now().astimezone().isoformat(),
                  campaign=read(ROOT/'attempt1/campaign_status.json'), runs={})
    for seed in (42,0,123):
        run = ROOT/f'attempt1/C1_seed{seed}'
        if not run.exists():
            result['runs'][seed] = dict(status='NOT_LAUNCHED')
            continue
        entry = {name:read(run/name) for name in ('run_status.json','run_final.json',
                 'training/fastpath_launch.json','training/progress.json',
                 'training/budget_progress.json','training/failure_receipt.json')}
        for name in ('training/epoch_timing.jsonl','resources.jsonl'):
            p = run/name
            if p.exists():
                with p.open('rb') as f:
                    f.seek(max(0,p.stat().st_size-20000))
                    lines = f.read().decode().splitlines()
                entry[name+'_last'] = [json.loads(line) for line in lines[-3:]]
        p = run/'training.log'
        if p.exists():
            with p.open('rb') as f:
                f.seek(max(0,p.stat().st_size-1600))
                entry['log_tail'] = f.read().decode(errors='replace')
        result['runs'][seed] = entry
    result['tmux'] = subprocess.run(['tmux','list-sessions'],capture_output=True,text=True).stdout
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
