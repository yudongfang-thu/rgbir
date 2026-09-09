"""Finite C1 seed42->0->123 queue, gated only by engineering evidence."""
import argparse
import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from budget_policy import write_json

HERE = Path(__file__).resolve().parent
PYTHON = '/mnt/dataset/yudongfang/projects/SpaceNet6_OTD_official_reproduction/environments/sn6-int8-kd/bin/python'


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def main(root):
    root.mkdir(parents=True, exist_ok=False)
    launched = []
    while True:
        states = {s:read(root/f'C1_seed{s}'/'run_final.json') for s in launched}
        failures = {s:v['status'] for s,v in states.items() if v and v['status'] != 'COMPLETED'}
        active = [s for s,v in states.items() if v is None]
        gate = read(root/'C1_seed42/training/budget_progress.json')
        allow_more = bool(gate and gate.get('eligible'))
        pending = [s for s in (42,0,123) if s not in launched]
        status = dict(launched_seeds=launched, pending_seeds=pending, active_seeds=active,
                      failures=failures, seed42_five_epoch_pass=allow_more, reads_ap=False,
                      max_concurrent_runs=2, updated_at=datetime.datetime.now().astimezone().isoformat())
        if failures:
            write_json(root/'campaign_status.json', dict(status, status='PAUSED_TECHNICAL',
                note='Unstarted seeds held; already running seeds retain their own deadlines'))
            return 1
        if not pending and not active:
            write_json(root/'campaign_status.json', dict(status, status='COMPLETED'))
            return 0
        if pending and len(active) < 2 and (not launched or allow_more):
            seed = pending[0]
            session = f'c1budget_paired_{seed}'
            command = [PYTHON,'-B',str(HERE/'run_budgeted.py'),'--seed',str(seed),'--output',str(root/f'C1_seed{seed}')]
            shell_command = shlex.join(command)+' > '+shlex.quote(str(root/f'manager_s{seed}.log'))+' 2>&1'
            subprocess.run(['tmux','new-session','-d','-s',session,shell_command], check=True)
            launched.append(seed)
            write_json(root/f'launch_seed{seed}.json', dict(session=session, command=command,
                       launched_at=datetime.datetime.now().astimezone().isoformat(), seed42_gate=gate))
        write_json(root/'campaign_status.json', dict(status, status='RUNNING'))
        time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    raise SystemExit(main(parser.parse_args().output))
