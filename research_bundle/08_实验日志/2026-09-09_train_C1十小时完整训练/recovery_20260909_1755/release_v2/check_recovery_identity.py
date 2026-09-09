from pathlib import Path
import json,torch
root=Path('/mnt/dataset/yudongfang/projects/RGBT_campaign/artifacts/c1_budget10h_20260909')
left=root/'attempt1/C1_seed0/training';right=root/'recovery_attempt2/C1_seed0/training'
def equal(a,b):
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
 if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return type(a)==type(b) and a==b
checks={}
for name in ('initial_student.pt','first_batch.pt'):
 a=torch.load(left/name,map_location='cpu',weights_only=False);b=torch.load(right/name,map_location='cpu',weights_only=False)
 checks[name]=dict(equal=bool(equal(a,b)),top_level_count=len(a))
 del a,b
for name in ('train_c1_budget.py','evaluate_c1_budget.py','budget_policy.py'):
 checks[name]=dict(byte_identical=(root/'release_v1'/name).read_bytes()==(root/'release_v2'/name).read_bytes())
result=dict(status='PASS' if all(v.get('equal',v.get('byte_identical')) for v in checks.values()) else 'FAIL',checks=checks,cuda_initialized=torch.cuda.is_initialized(),scope='Initial student and first actual batch only; no all-flow equality claim',resume=False,new_hash_computed=False)
with (root/'recovery_identity_receipt.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
