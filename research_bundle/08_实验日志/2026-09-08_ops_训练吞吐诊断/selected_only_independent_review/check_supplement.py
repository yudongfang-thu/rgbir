"""Independent CPU checks only; no model, CUDA, hash, or source mutation."""
import json
from pathlib import Path
import sys
import torch

OPS = Path(__file__).resolve().parents[1]
CANDIDATE = OPS / 'performance_candidate'
REFERENCE = Path(r'E:/SHARE/光sar/03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_independent_kd_v2')
sys.path.insert(0, str(CANDIDATE))
import benchmark_selected_only as bench
import benchmark_candidate as shared

torch.set_num_threads(4)
adapter, classification = shared.load_reference(REFERENCE)
api = bench.make_api(adapter, classification)
original_binding = adapter.build_classification_selection
source_before = {name: (CANDIDATE/name).read_bytes() for name in ('selected_only_v1.py','benchmark_selected_only.py')}

def evaluate(data, thin, eta, full):
    raw = {name: dict(data[name]) for name in ('student','teacher','reference')}
    for values in raw.values():
        for key in ('scores','boxes'):
            values[key] = values[key].detach().clone().requires_grad_(True)
    args = [raw[name] for name in ('student','teacher','reference')] + [data['batch']]
    kwargs = dict(strides=data['strides'],config=data['evidence'],selection_seed=20260907)
    if thin:
        payload = api.build(*args,full_diagnostics=full,**kwargs)
        loss, stats = api.loss(payload,off_target_weight=eta)
    else:
        payload = adapter.build_classification_selection(*args,**kwargs)
        loss, stats = classification.classification_loss_from_selection(payload,off_target_weight=eta)
    payload.assert_source(*args)
    components = classification.classification_loss_components(payload,off_target_weight=eta)
    result = dict(loss=loss.detach())
    for key in ('target_loss','off_target_loss_unit','off_target_loss','c0_loss'):
        result[key] = components[key].detach()
    for key, value in [('loss',loss),('target',components['target_loss']),('off_unit',components['off_target_loss_unit'])]:
        gradient, = torch.autograd.grad(value,raw['student']['scores'],retain_graph=True,allow_unused=True)
        result[key+'_gradient'] = gradient
    if thin:
        result['used'] = payload.thin_path_used
        if not full:
            assert all(stats[name] is None for name in stats['missing_diagnostics'])
    return result, stats

rows = []
for name,data in bench.fixture_data(REFERENCE,True):
    if name not in ('empty','one_class','many_objects','unmatched_rgb_gt_in_background_exclusion'):
        continue
    for eta in (0.,.25):
        old, oldstats = evaluate(data,False,eta,False)
        for full in (False,True):
            new, newstats = evaluate(data,True,eta,full)
            differences = {key:shared.tensor_difference(value,new[key]) for key,value in old.items()}
            assert new['used'] is True
            full_stats_exact = all(oldstats[key] == newstats[key] for key in oldstats) if full else None
            passed = all(value['pass_numeric'] for value in differences.values()) and full_stats_exact is not False
            rows.append(dict(case=name,eta=eta,full=full,pass_numeric=passed,
                             full_stats_exact=full_stats_exact,differences=differences))

name, data = next((name,data) for name,data in bench.fixture_data(REFERENCE,True) if name == 'many_objects')
data['teacher']['scores'] = data['reference']['scores'].detach().clone()
old, stats = evaluate(data,False,.25,False)
assert stats['base_count'] > 0 and stats['selected_count'] == 0
for full in (False,True):
    new, newstats = evaluate(data,True,.25,full)
    differences = {key:shared.tensor_difference(value,new[key]) for key,value in old.items()}
    assert new['loss_gradient'] is not None and torch.count_nonzero(new['loss_gradient']) == 0
    rows.append(dict(case='nonempty_base_zero_selected',full=full,base=stats['base_count'],selected=0,
                     pass_numeric=all(value['pass_numeric'] for value in differences.values()),differences=differences))

for dtype in (torch.float16,torch.float32):
    for bad in (float('nan'),float('inf')):
        name,data = next((name,data) for name,data in bench.fixture_data(REFERENCE,True) if name == 'one_class')
        for raw in ('student','teacher','reference'):
            data[raw]['scores'] = data[raw]['scores'].detach().to(dtype)
        data['student']['scores'].reshape(-1)[0] = bad
        errors=[]
        for thin in (False,True):
            try:
                evaluate(data,thin,.25,False)
                errors.append(None)
            except Exception as error:
                errors.append(dict(type=type(error).__name__,text=str(error)))
        rows.append(dict(case='nonfinite_'+str(dtype)+'_'+str(bad),errors=errors,
                         pass_numeric=errors[0] is not None and errors[0]==errors[1]))

name,data = next((name,data) for name,data in bench.fixture_data(REFERENCE,False) if name == 'one_class')
sel = adapter.build_classification_selection(data['student'],data['teacher'],data['reference'],data['batch'],
                                            strides=data['strides'],config=data['evidence'])
region = sel.regions[0]
data['student']['scores'] = data['student']['scores'].detach().clone()
values = data['student']['scores'][region.image_index,:,region.anchor_start:region.anchor_end]
values[:,region.rgb_foreground[0]] = 3e38
values[:,region.rgb_background[0]] = -3e38
errors=[]
for thin in (False,True):
    try:
        evaluate(data,thin,.25,False)
        errors.append(None)
    except Exception as error:
        errors.append(dict(type=type(error).__name__,text=str(error)))
rows.append(dict(case='finite_fp32_pool_overflow_fallback',errors=errors,
                 pass_numeric=errors[0] is not None and errors[0]==errors[1]))

receipt = dict(status='PASS' if all(row['pass_numeric'] for row in rows) else 'FAIL',
               torch=str(torch.__version__),cuda_initialized=torch.cuda.is_initialized(),new_hash_computed=False,
               source_bytes_unchanged=all(value==(CANDIDATE/name).read_bytes() for name,value in source_before.items()),
               original_selector_binding_unchanged=adapter.build_classification_selection is original_binding,checks=rows)
output = Path(__file__).resolve().parent/'supplement_receipt.json'
with output.open('x',encoding='utf-8') as handle:
    json.dump(receipt,handle,indent=2,ensure_ascii=False,allow_nan=False)
print(json.dumps({key:value for key,value in receipt.items() if key!='checks'},ensure_ascii=False))
print('checks',len(rows))
if receipt['status'] != 'PASS':raise SystemExit(2)
