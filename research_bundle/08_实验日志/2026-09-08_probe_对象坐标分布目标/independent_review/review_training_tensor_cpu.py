"""Independent analytical loss/gradient checks on CPU plus full synthetic suite."""
from pathlib import Path
import sys,math,json,io,unittest,contextlib
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
attempt=sys.argv[1] if len(sys.argv)>1 else 'attempt1'
suffix='' if attempt=='attempt1' else '_'+attempt
RELEASE=HERE.parent/'trainer_release'
WORKSPACE=HERE.parents[2]
REFERENCE=WORKSPACE/'03_现行工程'/'SpaceNet6_OTD_official_reproduction'/'experiments'/'rgbir_independent_kd_v2'
sys.path.insert(0,str(RELEASE));sys.path.insert(0,str(HERE))
sys.argv=[__file__,'--reference-dir',str(REFERENCE),'--output',str(HERE/'unused_author_main_output.json')]
import test_l3_cpu as fixtures
import l3_distribution_loss as l3
import torch
from independent_oracle import reference
torch.set_num_threads(1)
checks={};metrics={}

z=torch.linspace(-3,3,128).reshape(1,64,2).clone().requires_grad_(True)
q=torch.full((1,4,16),1/32);q[:,:,7]+=0.5
ids=torch.tensor([[0,1,1,0,0]])
loss=l3.distribution_loss({'boxes':z},ids,q,3)
loss.backward()
selected=z.detach().permute(0,2,1)[0,1].reshape(4,16)
weights=(selected/2).softmax(-1)
analytic=(2/(4*3))*(weights-q[0])
actual=z.grad.permute(0,2,1)[0,1].reshape(4,16)
metrics['max_analytical_gradient_error']=float((actual-analytic).abs().max())
checks['T_squared_four_mean_base_analytic_gradient']=metrics['max_analytical_gradient_error']<2e-8
manual=4/3*sum(float(q[0,s,j])*(math.log(float(q[0,s,j]))-math.log(float(weights[s,j]))) for s in range(4) for j in range(16))/4
metrics['manual_KL_absolute_error']=abs(float(loss)-manual)
checks['full_KL_matches_manual']=metrics['manual_KL_absolute_error']<1e-6
checks['unselected_anchor_zero_gradient']=bool(z.grad[:,:,0].eq(0).all())

s,t,r,b=fixtures.fixture();b['teacher_batch']=fixtures.labels(((15.,15.,41.,41.),));t=fixtures.raw(box=(15.,15.,41.,41.))
orig,native=l3._helpers();chosen,gt,target,stats,c,st=l3._select(t,r,b,(8,16,32),orig,native)
raw=t['boxes'][0,:,27].detach().reshape(4,16).tolist()
p=[]
for row in raw:
    e=[math.exp((v-max(row))/2) for v in row];p.append([x/math.fsum(e) for x in e])
expected=reference(p,[15.,15.,41.,41.],[16.,16.,40.,40.],[28.,28.],[28.,28.],8,8)
metrics['T2_transport_reference_max_error']=max(abs(float(target[0,a,j])-expected['targets'][a][j]) for a in range(4) for j in range(16))
checks['T2_before_nonidentity_independent_hat_basis']=expected['accepted'] and metrics['T2_transport_reference_max_error']<3e-8

dist=torch.tensor([[0.,1.25,2.5,14.75]])
g=l3.gt_temperature2_target(dist)
expected_gt=[[1.]+[0.]*15,[0.,math.sqrt(.75)/(math.sqrt(.75)+.5),.5/(math.sqrt(.75)+.5)]+[0.]*13,[0.,0.,.5,.5]+[0.]*12,[0.]*14+[.5/(math.sqrt(.75)+.5),math.sqrt(.75)/(math.sqrt(.75)+.5)]]
metrics['GT_adjacent_expected_float32_rounding_abs_error']=max(abs(float(g[0,a,j])-expected_gt[a][j]) for a in range(4) for j in range(16))
# Two FP32 operations (sqrt, division) compared against exact double reference.
checks['GT_adjacent_then_sqrt_reference']=metrics['GT_adjacent_expected_float32_rounding_abs_error']<=torch.finfo(torch.float32).eps

log=io.StringIO()
with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
    authored=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(fixtures.Truths))
(HERE/('trainer_tensor_cpu_test_output'+suffix+'.txt')).write_text(log.getvalue(),encoding='utf-8')
result={'status':'PASS' if all(checks.values()) and authored.wasSuccessful() else 'FAIL','independent_checks':checks,'metrics':metrics,
  'author_tensor_tests_executed_by_reviewer':authored.testsRun,'author_tensor_tests_passed':authored.wasSuccessful(),
  'python_executable':sys.executable,'torch_version':torch.__version__,'device':'cpu','reference_helpers':str(REFERENCE),
  'new_GPU_or_SSH':False,'new_hash_computed':False,'scope':'Synthetic numerical and autograd checks only; production CUDA canary remains mandatory.'}
with (HERE/('TRAINER_TENSOR_CPU_TESTS'+suffix+'.json')).open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
raise SystemExit(result['status']!='PASS')
