"""Additional bounded CPU truths for L2; no AP, GPU or new hash."""
from pathlib import Path
import importlib.util
import json
import sys
import torch

HERE = Path(__file__).resolve().parent
PINNED = Path('E:/SHARE/光sar/03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_independent_kd_v2')
SOURCE = HERE.parent / 'newentry/release'
sys.dont_write_bytecode = True
sys.argv = ['fixture_import', '--pinned', str(PINNED), '--receipt', str(HERE / 'unused.json')]
spec = importlib.util.spec_from_file_location('fixture_only', SOURCE / 'test_localization_box_v2_cpu.py')
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)
l2 = f.l2
torch.set_num_threads(1)
checks = []

# The actual LLVIP class dimension is one; author fixtures intentionally use two.
s, t, r, batch = f.fixture()
for raw in [s, t, r]:
    raw['scores'] = raw['scores'][:, :1].detach().requires_grad_(True)
loss, stats = l2.compute(s, t, r, batch)
loss.backward()
assert stats['selected_count'] == 1 and s['boxes'].grad.isfinite().all() and s['boxes'].grad.abs().sum() > 0
assert all(raw['scores'].grad is None for raw in [s, t, r])
checks.append('LLVIP_nc1_selected_DFL_gradient_and_detached_class_scores')

# Highest-confidence reference wins even when a lower-confidence anchor has perfect IoU.
s, t, r, batch = f.fixture()
r['boxes'] = r['boxes'].detach()
r['scores'] = r['scores'].detach()
f.set_box(r, 18, (16., 16., 40., 40.))
r['scores'][0, 0, 18] = 3.
_, stats = l2.compute(s, t, r, batch)
assert stats['base_records'][0]['reference_anchor'] == 27 and stats['selected_count'] == 1
r['scores'][0, 0, 18] = 5.
_, stats = l2.compute(s, t, r, batch)
assert stats['base_records'][0]['reference_anchor'] == 18 and stats['base_count'] == 1 and stats['selected_count'] == 0
checks.append('reference_confidence_first_and_perfect_R_gap_exclusion')

# Positive independent x/y affine scale preserves box-vs-own-GT IoU.
ir = torch.tensor([[10., 20., 30., 60.]])
rgb = torch.tensor([[20., 40., 80., 100.]])
teacher = torch.tensor([[8., 24., 34., 56.]])
mapped = l2.map_teacher_box(teacher, ir, rgb)
original, _ = l2._helpers()
assert torch.allclose(original._iou(teacher, ir), original._iou(mapped, rgb), atol=1e-6, rtol=1e-6)
checks.append('mapped_IoU_is_annotation_affine_invariant_not_independent_RGB_geometry_evidence')

# Shared labels make the transformation the identity, not a measured registration.
assert torch.equal(l2.map_teacher_box(teacher, ir, ir), teacher)
checks.append('shared_GT_mapping_identity')

receipt = {'status': 'PASS', 'additional_checks': checks, 'cuda_initialized': torch.cuda.is_initialized(),
           'scope': 'BOUNDED_CPU_OPERATOR_REVIEW_ONLY', 'L1_geometry_admitted': False,
           'new_AP_read': False, 'new_hash_computed': False,
           'source': {'path': str(SOURCE / 'localization_box_v2.py'),
                      'bytes': (SOURCE / 'localization_box_v2.py').stat().st_size}}
assert not receipt['cuda_initialized']
with (HERE / 'independent_checks_receipt.json').open('x', encoding='utf-8') as out:
    json.dump(receipt, out, ensure_ascii=False, indent=2)
print(json.dumps(receipt))
