"""Independent operator tests; never reads the real cache or invokes a GPU."""
from pathlib import Path
import importlib.util
import json
import math
import sys
from fractions import Fraction as F
from independent_oracle import reference, small_truth

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('reviewed_transport',HERE.parent/'transport'/'probability_transport.py')
operator=importlib.util.module_from_spec(spec);spec.loader.exec_module(operator)


def check_operator():
    checks=small_truth()
    max_p_error=max_d_error=0.
    # Deliberately varied logit shapes and nontrivial annotation/stride affine.
    logits=[[math.sin(j+side)*2-j/7 for j in range(16)] for side in range(4)]
    probabilities=[]
    for row in logits:
        weights=[math.exp(x-max(row)) for x in row]
        probabilities.append([x/math.fsum(weights) for x in weights])
    cases=[
       ([0,0,16,24],[0,0,16,24],[8,12],[8,12],1,1),
       ([0,0,16,24],[7,13,23,37],[8,12],[15,25],1,1),
       ([0,0,16,24],[7,13,15,25],[8,12],[11,19],2,2),
       ([0,0,16,24],[0,0,16,24],[8,12],[8,12],1,2),
       ([0,0,16,24],[0,0,16,24],[8,12],[9,12],1,1),
       ([0,0,16,24],[0,0,32,48],[8,12],[16,24],1,1),
    ]
    for i,args in enumerate(cases):
        expected=reference(probabilities,*args)
        actual=operator.transport_logits(logits,*args)
        supported=actual['status']=='SUPPORTED'
        checks[f'operator_case_{i}_support']=supported==expected['accepted']
        ds=max(abs(a-b) for row_a,row_b in zip(expected['distances'],[s['mapped_distance_bins'] for s in actual['sides']]) for a,b in zip(row_a,row_b))
        max_d_error=max(max_d_error,ds)
        checks[f'operator_case_{i}_all_mapped_distances']=ds<1e-12
        if supported:
            ps=max(abs(a-b) for qa,qb in zip(expected['targets'],actual['target_probabilities']) for a,b in zip(qa,qb))
            max_p_error=max(max_p_error,ps)
            checks[f'operator_case_{i}_full_target_mass']=ps<1e-13
        else:
            checks[f'operator_case_{i}_whole_object_null']=actual['target_probabilities'] is None
    for label,p,a,b,expected in [
       ('last_bin',[0]*15+[1],F(1),F(0),'SUPPORTED'),
       ('tiny_positive_outside',[1e-300]+[0]*14+[1],F(1),F(-1),'OUTSIDE_STUDENT_SUPPORT'),
       ('zero_mass_outside',[0,1]+[0]*14,F(1),F(-1),'SUPPORTED'),
       ('integer',[0,1]+[0]*14,F(2),F(0),'SUPPORTED'),
       ('fractional',[0,1]+[0]*14,F(1,2),F(0),'SUPPORTED')]:
        actual=operator.scatter_complete(p,a,b)
        checks['operator_'+label]=actual['status']==expected
        if label=='fractional':checks['fractional_expected_two_bins']=actual['target_probabilities']==[.5,.5]+[0]*14
    invalids=[(None,[0,0,16,24]),(logits,[0,0,0,24]),([[float('nan')]*16]*4,[0,0,16,24]),([[-10000]+[0]*15]*4,[0,0,16,24])]
    for i,(ls,tg) in enumerate(invalids):
        actual=operator.transport_logits(ls,tg,[0,0,16,24],[8,12],[8,12],1,1)
        checks[f'operator_invalid_{i}_rejected']=actual['status']=='INVALID_INPUT' and actual['target_probabilities'] is None
    return {'checks':checks,'pass':all(checks.values()),'max_probability_error':max_p_error,'max_distance_error':max_d_error,
            'cache_executed':False,'training_executed':False,'new_hash_computed':False}


if __name__=='__main__':
    result=check_operator()
    output=HERE/'source_small_truth_result.json'
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['pass'] else 1)
