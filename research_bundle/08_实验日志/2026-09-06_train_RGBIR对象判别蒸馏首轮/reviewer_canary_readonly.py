"""Independent final canary audit: read JSON and checkpoint on CPU only."""
import json
import os
from pathlib import Path
import torch

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
assert not torch.cuda.is_initialized()
project = Path('/mnt/dataset/yudongfang/projects/RGBT_campaign')
artifacts = project / 'artifacts/rgbir_object_evidence_v1_20260906'
runs = project / 'runs/rgbir_object_evidence_v1_20260906'
result = {'reviewer': '/root/rgbir_code_review', 'cuda_used': False,
          'allocation': json.loads((artifacts/'allocation.json').read_text()),
          'comparison': json.loads((artifacts/'canary_comparison.json').read_text()),
          'runs': {}}
for arm in ('paired', 'weight0'):
    run = runs / f'canary_{arm}_s42_attempt1'
    record = {
        'path': str(run),
        'completion': json.loads((run/'completion_receipt.json').read_text()),
        'run_evidence': json.loads((run/'run_evidence/run_receipt.json').read_text()),
        'ready': json.loads((run/'runtime_ready.json').read_text()),
    }
    checkpoint = torch.load(run/'weights/last.pt', map_location='cpu', weights_only=False)
    model = checkpoint.get('ema') or checkpoint.get('model')
    state = model.state_dict()
    initial = torch.load(run/'initial_student.pt', map_location='cpu', weights_only=True)
    suspect_keys = [k for k in state if any(word in k.lower() for word in ('teacher', 'reference', 'criterion'))]
    direct_aux = [name for name in ('teacher', 'reference', 'trainer', 'criterion')
                  if getattr(model, name, None) is not None]
    snapshots = record['run_evidence']['source_snapshots']
    missing_copies = [p for paths in snapshots.values() for p in paths
                     if not (run/'run_evidence'/p).is_file()]
    record['checkpoint_audit'] = {
        'path': str(run/'weights/last.pt'),
        'model_class': type(model).__name__,
        'checkpoint_fields': sorted(checkpoint),
        'model_field_is_none': checkpoint.get('model') is None,
        'ema_present': checkpoint.get('ema') is not None,
        'criterion_is_none': getattr(model, 'criterion', None) is None,
        'direct_auxiliary_attributes': direct_aux,
        'suspect_state_keys': suspect_keys,
        'state_keys_and_shapes_match_initial': set(state) == set(initial) and
            all(state[k].shape == initial[k].shape for k in state),
        'state_tensor_count': len(state),
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'all_checkpoint_tensors_cpu': all(v.device.type == 'cpu' for v in state.values()),
        'all_floating_tensors_finite': all(bool(torch.isfinite(v).all()) for v in state.values() if v.is_floating_point()),
        'missing_evidence_copies': missing_copies,
    }
    assert not suspect_keys and not direct_aux and not missing_copies
    assert record['checkpoint_audit']['state_keys_and_shapes_match_initial']
    result['runs'][arm] = record
assert not torch.cuda.is_initialized()
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
