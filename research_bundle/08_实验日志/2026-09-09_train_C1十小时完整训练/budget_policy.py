"""Outcome-blind run timing and terminal-state checks; no ML imports."""
import json
from pathlib import Path
import statistics

BUDGET_SECONDS = 36000
TAIL_RESERVE_SECONDS = 3600


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def forecast(rows, elapsed, total_epochs=200, budget=BUDGET_SECONDS, reserve=TAIL_RESERVE_SECONDS):
    completed = len(rows)
    recent = [r['epoch_seconds'] for r in rows[-3:]]
    projected = elapsed + (total_epochs-completed)*statistics.mean(recent) + reserve
    return dict(completed_epochs=completed, elapsed_seconds=elapsed,
                recent_mean_epoch_seconds=statistics.mean(recent),
                projected_total_seconds=projected, tail_reserve_seconds=reserve,
                budget_seconds=budget, eligible=completed >= 5 and projected <= budget,
                stop_for_budget=completed >= 5 and projected > budget,
                reads_ap=False)


def require_e200(run):
    run = Path(run)
    r = json.loads((run/'completion_receipt.json').read_text())
    if (r.get('status') != 'training_completed' or r.get('last_epoch') != 200 or
            r.get('epochs_configured') != 200 or r.get('official_test_accessed') is not False):
        raise RuntimeError('A completed, unexposed E200 endpoint is required before evaluation')
    return r


def require_full_evaluation(run):
    run = Path(run)
    value = json.loads((run/'evaluation_val.json').read_text())
    if (value.get('status') != 'completed' or value.get('endpoint') != 'fixed_budget_last_ema' or
            value.get('split') != 'val' or value.get('official_test_accessed') is not False or
            value.get('metric_units') != 'fraction_0_to_1'):
        raise RuntimeError('Independent dev evaluation has not completed')
    contract = json.loads(Path(value['evaluation_contract']).read_text())
    if contract.get('observed_images') != 1469 or contract.get('expected_val_images') != 1469:
        raise RuntimeError('Incomplete Drone development population')
    return value
