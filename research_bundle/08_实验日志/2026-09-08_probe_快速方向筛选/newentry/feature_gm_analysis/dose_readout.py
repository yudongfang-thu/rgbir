"""CPU arithmetic on previously collected fixed-eight calibration gradients."""
import argparse
import json
import math
from pathlib import Path
import statistics

LAMBDA_F = 14.438521129817886
LAMBDA_C1 = 0.09227393550836771


def distribution(values):
    return dict(n=len(values), minimum=min(values), median=statistics.median(values),
                mean=statistics.mean(values), maximum=max(values), values=values)


def summarize(rows, receipt):
    assert receipt['status'] == 'DIRECTION_CALIBRATION_COMPLETED' and receipt['dataset'] == 'drone'
    assert receipt['coefficients']['F-rel'] is None and 'F-rel' in receipt['blocked']
    assert receipt['coefficients']['C1'] == LAMBDA_C1
    assert len(rows) == 8 and [r['batch'] for r in rows] == list(range(1, 9))
    values=[]
    for row in rows:
        n=row['native_norm']; c=row['unit_B_kd_norms']['C1']; f=row['unit_B_kd_norms']['F-rel']
        assert all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in (n,c,f))
        values.append(dict(batch=row['batch'], native_norm=n, C1_unit_B_norm=c, F_unit_B_norm=f,
            coefficient_to_match_C1=LAMBDA_C1*c/f, projected_F_over_weighted_C1=LAMBDA_F*f/(LAMBDA_C1*c),
            projected_F_over_native=LAMBDA_F*f/n, weighted_C1_over_native=LAMBDA_C1*c/n,
            F_native_cosine=row['native_cosines']['F-rel']))
    ratios=[r['coefficient_to_match_C1'] for r in values]
    assert statistics.median(ratios) == LAMBDA_F
    assert all(math.isclose(a,b,rel_tol=1e-14,abs_tol=0) for a,b in zip(ratios,receipt['details']['F-rel']['ratios']))
    return dict(status='FIXED8_PROJECTED_DOSE_READOUT', fixed_lambda_F=LAMBDA_F, fixed_lambda_C1=LAMBDA_C1,
        scope='CPU projection of previously measured shared-parameter gradients; no new backward or F AP',
        norm_definition='L2 norm of B*unit-KD gradients on the original model16/model19 shared parameter list; native.sum() uses the same list',
        original_F_rel_still_blocked=True, new_protocol='FEATURE_RELATION_GM_FT3', protocol_revised_after_calibration=True,
        distributions={k:distribution([v[k] for v in values]) for k in
          ('coefficient_to_match_C1','projected_F_over_weighted_C1','projected_F_over_native','weighted_C1_over_native','F_native_cosine')},
        batches=values, new_hash_computed=False,
        limitations=['Eight initial fixed batches and a parameter subset do not bound whole-model or later-step gradients.',
          'Gradient norm matching does not imply direction matching, equal updates, stability or AP benefit.',
          'The even-sample median of reciprocal ratios need not equal one.',
          'This is a post-calibration protocol revision fixed before any F AP; the original numeric-cap block is preserved.'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--calibration',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=json.loads((a.calibration/'calibration_receipt.json').read_text(encoding='utf-8-sig'))
    rows=[json.loads(s) for s in (a.calibration/'calibration_batches.jsonl').read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    result=summarize(rows,r)
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False)
    print(json.dumps(result['distributions'],indent=2))


if __name__=='__main__':main()
