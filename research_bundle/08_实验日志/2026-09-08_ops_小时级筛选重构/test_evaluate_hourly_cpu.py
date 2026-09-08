"""Tiny fake-receipt tests for the new eval adapter, without Torch or inference."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types

HERE = Path(__file__).parent
SOURCE = HERE / 'release/evaluate_hourly.py'
before = SOURCE.read_bytes()
helpers = types.ModuleType('hourly_common')
helpers.ENDPOINT = 'HOURLY_SCREEN_FT_E3_LAST_EMA'
helpers.read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
helpers.stat = lambda p: dict(path=str(Path(p).resolve()), bytes=Path(p).stat().st_size,
                              mtime_ns=Path(p).stat().st_mtime_ns)
for name in ('copy_sources', 'data_output', 'load_config', 'write_new'):
    setattr(helpers, name, lambda *a, **k: None)
sys.modules['hourly_common'] = helpers
spec = importlib.util.spec_from_file_location('hourly_eval_under_test', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
checks = {}


def reject(name, operation):
    try:
        operation()
    except (ValueError, KeyError, FileNotFoundError):
        checks[name] = True
    else:
        checks[name] = False


with tempfile.TemporaryDirectory(prefix='eval_fixture_', dir=HERE) as temporary:
    tmp = Path(temporary).resolve()
    assert tmp.parent == HERE.resolve()  # bound cleanup to this new fixture only
    (tmp / 'weights').mkdir()
    (tmp / 'weights/last.pt').write_bytes(b'FAKE-STAT-ONLY')
    (tmp / 'hourly_config.yaml').write_text('fake: fixed-new-training-config\n', encoding='utf-8')
    cfg = dict(arm='C1', seed=42, dataset='dronevehicle', model='warm-start.pt', teacher='ir.pt',
               reference='rgb.pt', classification_coefficient=.09227393550836771,
               localization_coefficient=0., expected_nc=5, expected_val_images=1469,
               expected_train_images=2048, imgsz=640, batch=32, workers=4,
               torch_version='2.10.0+cu128', ultralytics_version='8.4.115',
               native_contract_config=str(tmp / 'original_config.yaml'),
               paths=dict(student_data_yaml=str(tmp / 'subset.yaml')),
               auxiliary_data_identity=dict(student_data_yaml=str(tmp / 'full.yaml')))
    canary = dict(status='HOURLY_SCREEN_CANARY_COMPLETED', arm='C1', successful_updates=24,
                  config_copy=str(tmp / 'hourly_config.yaml'))
    def save_canary(value):
        (tmp / 'canary.json').write_text(json.dumps(value), encoding='utf-8')
    save_canary(canary)
    train = dict(status='HOURLY_SCREEN_TRAINING_COMPLETED', scope='HOURLY_SCREEN_FT',
                 single_seed=True, last_epoch=3, epochs_configured=3, endpoint=m.ENDPOINT,
                 formal_e200_complete=False, new_hash_computed=False, official_test_accessed=False,
                 checkpoint=helpers.stat(tmp / 'weights/last.pt'), canary_receipt=str(tmp / 'canary.json'),
                 **{k: cfg[k] for k in ('arm', 'seed', 'dataset', 'model', 'teacher', 'reference',
                                       'classification_coefficient', 'localization_coefficient')})
    def save_train(value):
        (tmp / 'hourly_training_receipt.json').write_text(json.dumps(value), encoding='utf-8')
    save_train(train)
    checks['valid_hourly_completion_default_canary'] = m.hourly_completion(tmp, cfg)[0] == train
    checks['valid_explicit_same_canary'] = m.hourly_completion(tmp, cfg, tmp / 'canary.json')[2] == (tmp / 'canary.json').resolve()
    for label, patch in [('reject_e8_status', dict(status='SHORT_SCREEN_TRAINING_COMPLETED')),
                         ('reject_wrong_epoch', dict(last_epoch=2)),
                         ('reject_initialization_mismatch', dict(model='other.pt')),
                         ('reject_checkpoint_stat_mismatch', dict(checkpoint={**train['checkpoint'], 'bytes': 99}))]:
        save_train({**train, **patch})
        reject(label, lambda: m.hourly_completion(tmp, cfg))
    save_train(train)
    reject('reject_other_explicit_canary', lambda: m.hourly_completion(tmp, cfg, tmp / 'other_canary.json'))
    for label, patch in [('reject_incomplete_canary', dict(status='FAILED')),
                         ('reject_canary_wrong_arm', dict(arm='N')),
                         ('reject_short_canary', dict(successful_updates=23))]:
        save_canary({**canary, **patch})
        reject(label, lambda: m.hourly_completion(tmp, cfg))
    (tmp / 'wrong_config.yaml').write_text('fake: wrong\n', encoding='utf-8')
    save_canary({**canary, 'config_copy': str(tmp / 'wrong_config.yaml')})
    reject('reject_canary_config_bytes', lambda: m.hourly_completion(tmp, cfg))
    save_canary(canary)

    native = {**cfg, 'model': 'yolo11n.pt', 'paths': dict(student_data_yaml=str(tmp / 'full.yaml'))}
    roster = [f'/original/dev/{i:05d}.jpg' for i in range(1469)]
    class FakeProfile:
        def __init__(self):
            self.seen = None
            self.subset = list(roster)
            self.kwargs = dict(m.EFFECTIVE)
        def validate_evaluation_profile_binding(self, binding, configuration, ref):
            self.seen = configuration
            return dict(actual_effective_kwargs=self.kwargs)
        def dev_roster(self, data):
            return self.subset if data == cfg['paths']['student_data_yaml'] else list(roster)
    profile = FakeProfile()
    def project():
        return m.evaluation_projection(cfg, native, tmp / 'original_config.yaml', profile, tmp / 'binding.json', tmp)
    rr, projection = project()
    checks['original_config_passed_to_native_binding'] = profile.seen is native and profile.seen['model'] == 'yolo11n.pt'
    checks['changed_training_identity_explicit'] = projection['training_model'] == 'warm-start.pt' and projection['training_identity_unchanged_claim'] is False
    checks['actual_eval_data_original_full'] = projection['actual_evaluation_data_yaml'] == native['paths']['student_data_yaml'] and rr == roster
    profile.subset.reverse()
    reject('reject_same_set_wrong_dev_order', project)
    profile.subset = list(roster)
    profile.kwargs['half'] = True
    reject('reject_changed_native_kwargs', project)
    profile.kwargs = dict(m.EFFECTIVE)
    cfg['batch'] = 16
    reject('reject_training_eval_invariant_change', project)

checks['source_bytes_unchanged'] = SOURCE.read_bytes() == before
checks['no_torch_imported'] = 'torch' not in sys.modules
result = dict(status='PASS_CPU_EVALUATION_ADAPTER_FIXTURES' if all(checks.values()) else 'FAIL',
              checks=checks, count=len(checks), common_helpers='I/O-only fixture stub; root common integration separate',
              real_receipts_used=False, real_checkpoint_loaded=False, gpu=False, new_hashes=False)
with (HERE / 'evaluation_cpu_checks.json').open('x', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
    f.write('\n')
print(json.dumps(result, ensure_ascii=True, indent=2))
