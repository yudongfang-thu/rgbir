"""Read-only deployed C1 inventory. No GPU, model loading, or content hashing."""
import datetime
import json
from pathlib import Path
import re

root = Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/rgbir_c1_fastpath_20260908')
release = root / 'production_release_v2'
plan = json.loads((release / 'production_plan_v2.json').read_text())
items = {'entry': release/'train_c1_fast.py', 'launcher': release/'launch_prepared_v2.py',
         'candidate': Path(plan['candidate_source']), 'evaluator': Path(plan['post_training']['evaluator'])}
output = {'checked_at': datetime.datetime.now().astimezone().isoformat(), 'gpu_used': False,
          'new_hash': False, 'model_loaded': False, 'attempts': {}, 'files': [], 'source_references': {}}
for seed, cell in plan['cells'].items():
    config = Path(cell['config'])
    items['config_seed'+seed] = config
    output['attempts'][seed] = {'output_exists': Path(cell['output']).exists(),
                               'completed': (Path(cell['output'])/'fastpath_completion_receipt.json').exists()}
    if seed == '42':
        content = config.read_text()
        for field in ('model','teacher','reference'):
            match = re.search(r'^'+field+r':\s*(.+)$', content, re.M)
            if match:
                items[field] = Path(match.group(1).strip())
for name, path in items.items():
    output['files'].append({'name': name, 'path': str(path), 'exists': path.is_file(),
                            'bytes': path.stat().st_size if path.is_file() else None})
for name in ('train_c1_fast.py','launch_prepared_v2.py','dispatch_single_formal.py'):
    output['source_references'][name] = [
        {'line': i, 'text': line} for i, line in enumerate((release/name).read_text().splitlines(), 1)
        if any(word in line.lower() for word in ('timeout','deadline','walltime','wall_time','36000','time_budget','max_hours','eval','time_limit'))
    ]
output['automatic_evaluation_in_launcher'] = plan['post_training']['automatic_evaluation_in_this_launcher']
output['full_epoch_measured_in_prior_plan'] = plan['timing']['full_epoch_measured']
output['status'] = 'READ_ONLY_INVENTORY_COMPLETED_NOT_LAUNCH_ADMISSION'
print(json.dumps(output, ensure_ascii=False, indent=2))
