"""Pinned installed Detect/DFL synthetic-raw CPU integration, no backbone/weights/GPU."""
import argparse,json,sys
from pathlib import Path

def run(a):
    sys.path.insert(0,str(a.reference_dir))
    import torch,ultralytics,selection_adapter
    from ultralytics.nn.modules.head import Detect
    from dfl_export import export_dfl
    torch.set_num_threads(2);torch.manual_seed(73)
    assert str(torch.__version__)=='2.10.0+cu128' and ultralytics.__version__=='8.4.115'
    models={};raw={}
    for name in ('S','R','T'):
        head=Detect(nc=1,reg_max=16,end2end=False,ch=(64,128,256));head.stride=torch.tensor([8.,16.,32.])
        model=torch.nn.Module();model.model=torch.nn.ModuleList([head]);models[name]=model
        raw[name]=dict(boxes=torch.randn(2,64,84),scores=torch.randn(2,1,84),
            feats=[torch.empty(2,c,h,h) for c,h in [(64,8),(128,4),(256,2)]])
    item=dict(image_index=0,frame_id='fixture0',stable_rgb_gt_id='rgb0',stable_ir_gt_id='ir0',bucket='both05_onlyT075',
        rgb_gt_xyxy=[4.,4.,32.,40.],ir_gt_xyxy=[6.,4.,34.,40.],C_selected=True,
        historical_L2_record=dict(reference_anchor=10,teacher_anchor=30),historical_L2_gates=dict(selected=True),
        native_iou50_matches={'S':dict(anchor_index=11),'R':dict(anchor_index=12),'T':dict(anchor_index=20)})
    original={n:{k:v.clone() for k,v in m.state_dict().items()} for n,m in models.items()}
    out=export_dfl(dict(img=torch.empty(2,3,64,64)),raw,models,[item],selection_adapter.original,selection_adapter.EvidenceConfig(input_size=64),(8,16,32))
    assert len(out['distributions'])==9
    roles=out['objects'][0]['roles'];assert roles['T']['same_S_native_iou50_index']['anchor_index']==11
    assert roles['T']['same_R_native_iou50_index']['anchor_index']==12
    assert roles['T']['same_historical_R_index']['own_gt_xyxy']==item['ir_gt_xyxy']
    assert roles['S']['historical_R_candidate']['own_gt_xyxy']==item['rgb_gt_xyxy']
    for n,m in models.items():
        assert all(torch.equal(v,m.state_dict()[k]) for k,v in original[n].items())
        assert all(p.grad is None for p in m.parameters())
    assert not torch.cuda.is_initialized()
    receipt=dict(status='PASS',scope='INSTALLED_NATIVE_CPU_SYNTHETIC_RAW_ONLY',checks=9,unique_distributions=9,
        native_probability_hooks=True,native_and_FP32_decode_exact=True,GT_role_semantics_exact=True,
        state_unchanged=True,backbone_forward=False,checkpoint_loaded=False,GPU_used=False,new_hash_computed=False)
    with a.output.open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
