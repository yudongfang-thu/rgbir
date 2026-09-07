"""Independent CPU fixtures for current natural selection diagnostics. No model/GPU/SSH/hash."""
import copy,importlib.util,io,json,math,sys,tempfile,types,unittest
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
TASK=HERE.parent
WORKSPACE=HERE.parents[3]
RELEASE=WORKSPACE/'03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_independent_kd_v2'
sys.path[:0]=[str(RELEASE),str(RELEASE/'task_conditional_reference')]
from selection_adapter import build_classification_selection,EvidenceConfig,original
from localization_loss import build_localization_selection,LocalizationConfig
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
diag=load('reviewed_natural_diagnostic',TASK/'diagnose_natural_flow.py')
stub=types.ModuleType('resource_dispatch')
def forbid_dispatch(*args,**kwargs):raise AssertionError('No real dispatch in independent CPU fixtures')
stub.run_job=forbid_dispatch
sys.modules['resource_dispatch']=stub
campaign=load('reviewed_natural_campaign',TASK/'run_campaign.py')

def labels(boxes,indices,classes=None):
    boxes=torch.tensor(boxes,dtype=torch.float32).reshape(-1,4)
    return dict(batch_idx=torch.tensor(indices,dtype=torch.long),cls=torch.tensor(classes or [0]*len(indices),dtype=torch.long).reshape(-1,1),
        bboxes=torch.cat(((boxes[:,:2]+boxes[:,2:])/2,boxes[:,2:]-boxes[:,:2]),-1)/64)
def raw(batch_size,distance=1.,classification_foreground=None,own_labels=None):
    n=84;out=dict(scores=torch.full((batch_size,2,n),-20.),boxes=torch.full((batch_size,32,n),-40.),
        feats=[torch.zeros(batch_size,2,n,n) for n in (8,4,2)])
    dfl=out['boxes'].reshape(batch_size,4,8,n);dfl[:,:,1,:]=0.
    for bi in range(batch_size):
        out['scores'][bi,0,27]=4.;dfl[bi,:,:,27]=-40.
        lo=int(distance);f=distance-lo;dfl[bi,:,lo,27]=math.log(1-f)
        if f:dfl[bi,:,lo+1,27]=math.log(f)
    if classification_foreground is not None:
        out['scores'][:,0,:]=0.;out['scores'][:,1,:]=-5.
        centers,_,_,_=original._layout(out,EvidenceConfig(input_size=64),(8,16,32))
        for bi,c,box in zip(own_labels['batch_idx'],own_labels['cls'].flatten(),own_labels['bboxes']):
            corners=torch.cat((box[:2]-box[2:]/2,box[:2]+box[2:]/2))*64
            inside=original._inside(corners[None],centers)[0]
            out['scores'][int(bi),int(c),inside]=classification_foreground
    return out
def slice_raw(r,i):return dict(scores=r['scores'][i:i+1],boxes=r['boxes'][i:i+1],feats=[f[i:i+1] for f in r['feats']])
def valid_summary(dataset='llvip',batches=2):
    return dict(status='COMPLETED',dataset=dataset,batches=batches,seed=20260907,canary_only=batches==2,
        trace_exact_all_recorded_fields=True,diagnostic_status='UNVERIFIED_GEOMETRY_DIAGNOSTIC',
        classification=dict(C0_C1_selection_identical=True),localization=dict(geometry_verified=False,formal_L1_admitted=False),
        backward_executed=False,training_executed=False,calibration_executed=False,validation_or_test_accessed=False,new_hash_computed=False,
        resources=dict(per_gpu_peak_vram_mib={'2':1024},peak_rss_mib=4000),gpu_reserved_peak_mib=800,gpu_allocated_peak_mib=700)

class IndependentNaturalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_C_global_quota_cannot_be_per_image(self):
        b=labels([[16,16,40,40]]*3,[0,1,2]);b['teacher_batch']=copy.deepcopy(b);b['im_file']=['a','b','c']
        t=raw(3,1.5,4.,b['teacher_batch']);r=raw(3,1.5,1.,b)
        sel=build_classification_selection(r,t,r,b,config=EvidenceConfig(input_size=64))
        self.assertEqual(sel.c0_stats['eligible_count'],3);self.assertEqual(sel.c0_stats['selected_count'],2)
        self.assertEqual(sel.c0_stats['normalizer'],3)
        each=[]
        for i in range(3):
            one=labels([[16,16,40,40]],[0]);one['teacher_batch']=copy.deepcopy(one);one['im_file']=[b['im_file'][i]]
            q=build_classification_selection(slice_raw(r,i),slice_raw(t,i),slice_raw(r,i),one,config=EvidenceConfig(input_size=64))
            each.append(q.c0_stats['selected_count'])
        self.assertEqual(sum(each),3)
        image_rows=[dict(governed_group='g') for _ in range(3)]
        rec=diag.classification_record(sel,b,image_rows)
        self.assertEqual(sum(v['selected'] for v in rec['base_records']),2)
        self.assertEqual(rec['counts']['normalizer'],3)
    def ragged_L(self):
        b=labels([[16,16,40,40],[16,16,40,40],[44,44,60,60],[16,16,40,40]],[2,0,2,3])
        b['teacher_batch']=labels([[16,16,40,40],[16,16,40,40],[16,16,40,40],[44,44,60,60]],[3,2,0,2])
        b['im_file']=['img0','empty1','img2','img3']
        t,r=raw(4,1.5),raw(4,1.)
        t['scores'][3,1,27]=5. # Fail T top class on image 3.
        cfg=LocalizationConfig(input_size=64)
        primary=build_localization_selection(t,r,b,config=cfg,mode='teacher',geometry_eligible=None,geometry_verified=False,return_records=True)
        return t,r,b,cfg,primary
    def test_L_ragged_empty_image_global_indices(self):
        t,r,b,cfg,primary=self.ragged_L()
        self.assertEqual(primary.stats['base_count'],3);self.assertEqual(primary.stats['selected_count'],2)
        self.assertEqual(primary.stats['normalizer'],3)
        self.assertEqual(primary.rgb_gt_indices.tolist(),[1,0,3]);self.assertEqual(primary.ir_gt_indices.tolist(),[2,1,0])
        out=diag.replay_by_image(t,r,b,cfg,(8,16,32),build_localization_selection,primary)
        self.assertEqual(len(out),4);self.assertEqual(out[1]['gate_counts']['base_count'],0)
        self.assertEqual(sum(x['gate_counts']['selected_count'] for x in out),2)
    def test_L_tampered_count_rejected(self):
        t,r,b,cfg,p=self.ragged_L();p.stats['common_count']+=1
        with self.assertRaises(AssertionError):diag.replay_by_image(t,r,b,cfg,(8,16,32),build_localization_selection,p)
    def test_L_tampered_global_identity_rejected(self):
        t,r,b,cfg,p=self.ragged_L();p.rgb_gt_indices[0]=3
        with self.assertRaises(AssertionError):diag.replay_by_image(t,r,b,cfg,(8,16,32),build_localization_selection,p)
    def test_L_tampered_distance_or_gate_rejected(self):
        t,r,b,cfg,p=self.ragged_L();p.rgb_distances[0,0]+=.001
        with self.assertRaises(AssertionError):diag.replay_by_image(t,r,b,cfg,(8,16,32),build_localization_selection,p)
        t,r,b,cfg,p=self.ragged_L();p.quality_gate[0]=False
        with self.assertRaises(AssertionError):diag.replay_by_image(t,r,b,cfg,(8,16,32),build_localization_selection,p)
    def test_L_zero_geometry_is_not_all_true_diagnostic(self):
        t,r,b,cfg,p=self.ragged_L()
        zero=build_localization_selection(t,r,b,config=cfg,geometry_eligible=torch.zeros(4,dtype=torch.bool),geometry_verified=False)
        self.assertEqual(zero.stats['base_count'],0);self.assertEqual(zero.stats['selected_count'],0)
        self.assertFalse(p.stats['geometry_verified']);self.assertFalse(p.stats['geometry_mask_supplied'])
    def test_trace_each_semantic_change_rejected(self):
        prior=[dict(image='rgb/a',rgb_source='raw/a',ir_source='ir/a',rgb_classes=[0],ir_classes=[0],
            rgb_boxes_normalized_xywh=[[.5,.5,.2,.2]],ir_boxes_normalized_xywh=[[.5,.5,.2,.2]],
            pair_info=dict(coverage=dict(torch_worker_seed=12),rgb_matrix=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]),
            student_tensor_shape=[3,640,640],student_dtype='torch.uint8')]
        diag.compare_trace(copy.deepcopy(prior),prior,0)
        cases=[]
        for key in ('image','rgb_source','ir_source','student_dtype'):
            a=copy.deepcopy(prior);a[0][key]='changed';cases.append(a)
        for key in ('rgb_classes','ir_classes'):
            a=copy.deepcopy(prior);a[0][key]=[1];cases.append(a)
        for key in ('rgb_boxes_normalized_xywh','ir_boxes_normalized_xywh'):
            a=copy.deepcopy(prior);a[0][key][0][0]+=1e-6;cases.append(a)
        a=copy.deepcopy(prior);a[0]['pair_info']['coverage']['torch_worker_seed']+=1;cases.append(a)
        a=copy.deepcopy(prior);a[0]['pair_info']['rgb_matrix'][0][0]+=1e-6;cases.append(a)
        for altered in cases:
            with self.assertRaises(ValueError):diag.compare_trace(altered,prior,0)
    def test_loader_metadata_all_contract_fields_enforced(self):
        prior={k:k for k in diag.METADATA_KEYS};diag.metadata_matches(dict(prior),prior)
        for key in diag.METADATA_KEYS:
            a=dict(prior);a[key]='changed'
            with self.assertRaises(ValueError):diag.metadata_matches(a,prior)
    def test_only_two_or_sixtyfour_batches(self):
        for n in (0,1,3,63,65):
            with self.assertRaises(ValueError):diag.run(types.SimpleNamespace(batches=n))
    def test_campaign_valid_two_datasets_two_sizes(self):
        for dataset,key in [('llvip','llvip'),('dronevehicle','drone')]:
            for n in (2,64):campaign.validate_summary(valid_summary(dataset,n),key,n)
    def test_campaign_semantic_corruption_rejected(self):
        s=valid_summary();cases=[]
        for key,val in [('status','FAILED'),('dataset','dronevehicle'),('batches',1),('seed',0),('canary_only',False),
                ('trace_exact_all_recorded_fields',False),('diagnostic_status','UNVERIFIED_GEOMETRY_UPPER_BOUND'),
                ('backward_executed',True),('training_executed',True),('calibration_executed',True),
                ('validation_or_test_accessed',True),('new_hash_computed',True)]:
            a=copy.deepcopy(s);a[key]=val;cases.append(a)
        for key in ['geometry_verified','formal_L1_admitted']:
            a=copy.deepcopy(s);a['localization'][key]=True;cases.append(a)
        a=copy.deepcopy(s);a['classification']['C0_C1_selection_identical']=False;cases.append(a)
        for a in cases:
            with self.assertRaises(ValueError):campaign.validate_summary(a,'llvip',2)
    def test_campaign_invalid_resource_measurements_rejected(self):
        for bad in [0,-1,float('nan'),float('inf')]:
            for key in ['gpu_reserved_peak_mib','gpu_allocated_peak_mib']:
                a=valid_summary();a[key]=bad
                with self.assertRaises(ValueError):campaign.validate_summary(a,'llvip',2)
            for key in ['peak_rss_mib','per_gpu_peak_vram_mib']:
                a=valid_summary();a['resources'][key]={'2':bad} if key=='per_gpu_peak_vram_mib' else bad
                with self.assertRaises(ValueError):campaign.validate_summary(a,'llvip',2)
        a=valid_summary();a['resources']['per_gpu_peak_vram_mib']={}
        with self.assertRaises(ValueError):campaign.validate_summary(a,'llvip',2)
    def test_campaign_invalid_full_cannot_write_completion(self):
        original_root,original_runner=campaign.ROOT,campaign.run_job
        calls=[]
        def simulated(job,queue):
            calls.append(job['id']);cmd=job['command'];p=Path(cmd[cmd.index('--output')+1]);p.mkdir()
            n=int(cmd[cmd.index('--batches')+1]);s=valid_summary('llvip',n)
            if n==64:s['batches']=63
            (p/'summary.json').write_text(json.dumps(s),encoding='utf-8')
        try:
            with tempfile.TemporaryDirectory(dir=str(HERE)) as td:
                campaign.ROOT=Path(td);campaign.run_job=simulated
                with self.assertRaises(ValueError):campaign.main()
                self.assertEqual(calls,['natural_llvip_canary','natural_llvip_full'])
                self.assertFalse((Path(td)/'queue_attempt1/completion.json').exists())
        finally:campaign.ROOT,campaign.run_job=original_root,original_runner
    def test_no_cuda_initialized_no_autograd_created(self):
        self.assertFalse(torch.cuda.is_initialized())
        t,r,b,cfg,p=self.ragged_L()
        for raw_ in (t,r):
            for v in [raw_['scores'],raw_['boxes']]+raw_['feats']:
                self.assertFalse(v.requires_grad);self.assertIsNone(v.grad)

def main():
    stream=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromTestCase(IndependentNaturalTests)
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    print(stream.getvalue())
    report=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),python=sys.version,torch=torch.__version__,
        cpu_only=True,cuda_initialized=torch.cuda.is_initialized(),new_hashes_computed=False,output=stream.getvalue())
    target=HERE/'cpu_test_result_v2.json'
    if target.exists():raise FileExistsError('Preserve earlier result; choose new result path')
    target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    if not result.wasSuccessful():raise SystemExit(1)
if __name__=='__main__':main()
