"""Freeze the reviewed E8 screen using existing executed evidence, before AP."""
import json
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parent
ART = Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts')
PERF = ART/'rgbir_throughput_20260908'
PROFILE = PERF/'short_profile_attempt1'
ENTRY = ROOT/'short_screen_E8_release'
CANDIDATE = ROOT/'performance_candidate/selected_only_v1.py'

def read(p):
    return json.loads(p.read_text())

def save(p, obj):
    with p.open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)

profiles = {a: read(PROFILE/('profile_'+a)/'short_profile_receipt.json') for a in ('N','C0','C1')}
for arm, p in profiles.items():
    assert p['arm']==arm and p['status']=='SHORT_SCREEN_THROUGHPUT_PROFILED'
    assert p['successful_updates']==24 and p['actual_optimizer_calls']==24
    assert p['batches']==30 and p['amp_skips']==6
    assert p['timing']['interval_includes_between_batch_loader_wait']
    assert p['dev_or_ap_computed'] is False
    g = read(PROFILE/'queue_attempt1'/('short_profile_'+arm+'_resource_profile.json'))
    assert g['status']=='COMPLETED' and not g['monitor_errors'] and g['minimum_free_mib']>=2048
assert profiles['C1']['candidate_execution']==dict(applicable=True,
    thin_learning_batches=30, fallback_batches=0, full_diagnostics_batches=3)
assert CANDIDATE.read_bytes()==(PROFILE/'performance_candidate/selected_only_v1.py').read_bytes()
comparison = PERF/'selected_update24_probe_attempt1/comparison_attempt1/receipt.json'
cmp = read(comparison)
assert cmp['trajectory_bitwise_exact'] and cmp['trajectory_numeric_agreement']

pure = 8*563*sum(p['timing']['seconds_per_batch'] for p in profiles.values())
estimated = pure*1.20 + 3600
assert estimated<=43200
admissions=ROOT/'admissions';admissions.mkdir(exist_ok=False)
approved=ROOT/'approved';approved.mkdir(exist_ok=False)
for entry in ('train_short_screen.py','evaluate_short_screen.py'):
    shutil.copyfile(ENTRY/entry, approved/entry)
shutil.copyfile(CANDIDATE, approved/CANDIDATE.name)
for arm, p in profiles.items():
    config=ENTRY/'configs'/('drone_'+arm+'_s42_E8.yaml')
    shutil.copyfile(config, approved/config.name)
    peak=max(p['resources']['per_gpu_peak_vram_mib'].values())
    host=p['resources']['peak_rss_mib']
    budget=dict(status='PASS_FOR_SHORT_SCREEN_RESOURCE_AND_BUDGET',globallease_required=True,
        measured_training_nvml_peak_mib=peak,measured_training_rss_peak_mib=host,
        profile=str(PROFILE/('profile_'+arm)/'short_profile_receipt.json'),
        reserved_training_nvml_mib=8192,reserved_training_rss_mib=32768,
        epoch_batches=563,epochs=8,pure_three_arm_training_seconds= pure,
        training_margin_factor=1.20,setup_evaluation_and_scheduling_allowance_seconds=3600,
        estimated_three_arm_seconds_with_margin=estimated,
        evaluation_measured_nvml_mib=1370,evaluation_measured_rss_mib=6260,
        evaluation_basis=str(ART/'rgbir_independent_kd_v2_20260907/evaluator_profile_attempt2/evidence_metrics.json'),
        limitations=['24 post-warmup batches are not a completed epoch',
            'Unknown shared-load changes and admission waits can exceed this planning estimate',
            'Three warmup epochs are retained: E8 measures early learning, not converged AP'],
        formal_e200_training_admitted=False,new_hash_computed=False)
    tech=dict(status='PASS_FOR_SHORT_SCREEN',arm=arm,new_successful_updates=24,
        candidate_source_reviewed=True,
        new_update_evidence=str(PROFILE/('profile_'+arm)/'short_profile_receipt.json'),
        update_probe_interpretation='Actual-cadence 24-update execution, not a new trajectory equivalence test',
        correctness_evidence=str(comparison) if arm=='C1' else str(ROOT/'short_screen_draft/NC0_REUSE_REVIEW.json'),
        correctness_interpretation='C1 selected-only exact24; N/C0 reuse accepted exact24 with unchanged source and first-epoch schedule',
        E8_changes=['common epochs=8 and LR horizon=8','separate SHORT_SCREEN endpoint and identity'],
        new_hash_computed=False,formal_gain_claim=False)
    save(admissions/(arm+'_technical_review.json'),tech)
    save(admissions/(arm+'_resource_budget.json'),budget)
    candidate=dict(kind='original') if arm!='C1' else dict(kind='criterion_factory',
        source=str(CANDIDATE),approved_source_copy=str(approved/CANDIDATE.name))
    save(admissions/(arm+'.json'),dict(status='SHORT_SCREEN_ADMITTED',
        scope='SINGLE_SEED_E8_SCREEN_ONLY',queue_authorized=True,formal_e200_authorized=False,
        arm=arm,seed=42,epochs=8,approved_config_copy=str(approved/config.name),
        approved_training_entry_copy=str(approved/'train_short_screen.py'),
        approved_evaluation_entry_copy=str(approved/'evaluate_short_screen.py'),
        technical_review=str(admissions/(arm+'_technical_review.json')),
        resource_budget_review=str(admissions/(arm+'_resource_budget.json')),candidate=candidate,
        decision_time=time.time(),decision_basis='User requests accelerated experimental feedback; fixed E8 before any new AP',
        existing_long_runs_modified=False,new_hash_computed=False))
save(ROOT/'freeze_receipt.json',dict(status='SHORT_SCREEN_E8_FROZEN',
    arms=['N','C0','C1'],seed=42,epochs=8,pure_training_hours=pure/3600,
    estimated_hours_with_margin=estimated/3600,new_hash_computed=False,
    E20_not_admitted_due_measured_cost=True,AP_observed_before_decision=False))
print(json.dumps(read(ROOT/'freeze_receipt.json')))
