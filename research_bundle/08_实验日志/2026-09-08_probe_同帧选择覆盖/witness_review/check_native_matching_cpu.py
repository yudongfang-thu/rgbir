"""Independent CPU witnesses for the captured pinned native matcher."""
import __future__
import argparse
import ast
import json
from pathlib import Path
import sys
import types
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'08_实验日志/2026-09-07_probe_双数据集证据优先推进/llvip_full_eval/remote_completed_attempt2/N42_full_attempt1/sources/4_validator.py'


def run():
    tree=ast.parse(SOURCE.read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BaseValidator')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='match_predictions')
    env=dict(np=np,torch=torch)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(SOURCE),'exec',flags=__future__.annotations.compiler_flag),env)
    native=env['match_predictions'];witness=[]
    cases=[('native_unique_order',[[.8,.9]],[0,0],[0],[[0,0]]),
           ('one_detection_two_gt',[[.8],[.9]],[0],[0,0],[[1,0]]),
           ('wrong_class',[[.9]],[1],[0],[]),
           ('tied_iou',[[.9,.9]],[0,0],[0],[[0,0]])]
    for name,iou,pred,gt,expected in cases:
        captured=[];old=sys.getprofile()
        def profile(frame,event,arg):
            if event=='return' and frame.f_code is native.__code__:
                captured.append(np.asarray(frame.f_locals['matches']).copy())
        try:
            sys.setprofile(profile)
            tp=native(types.SimpleNamespace(iouv=torch.tensor([.5])),torch.tensor(pred),torch.tensor(gt),torch.tensor(iou),use_scipy=False)
        finally:sys.setprofile(old)
        if len(captured)!=1 or captured[0].reshape(-1,2).tolist()!=expected:raise AssertionError(name)
        reconstructed=torch.zeros_like(tp)
        for _,di in expected:reconstructed[di,0]=True
        if not torch.equal(tp,reconstructed):raise AssertionError('Full TP bits differ')
        witness.append(dict(case=name,iou=iou,actual_gt_detection_pairs=expected,native_tp=tp.tolist()))
    s=SOURCE.stat()
    return dict(status='PASS',tests=len(cases),source=dict(path=str(SOURCE),bytes=s.st_size,mtime_ns=s.st_mtime_ns),
        cases=witness,method='native function return-frame local matches; no reconstructed greedy matcher',
        GPU_used=False,new_hash_computed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
    result=run()
    with a.receipt.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(result['status'],result['tests'])
