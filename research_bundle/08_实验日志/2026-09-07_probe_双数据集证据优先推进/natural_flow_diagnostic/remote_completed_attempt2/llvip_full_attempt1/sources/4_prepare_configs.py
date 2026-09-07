"""Materialize draft configurations; never manufactures calibration/readiness."""
import argparse
import copy
from pathlib import Path
import yaml

HERE=Path(__file__).resolve().parent


def configurations():
    result={}
    for dataset in ('drone','llvip'):
        base=yaml.safe_load((HERE/'task_conditional_reference/configs'/f'{dataset}_draft.yaml').read_text())
        for arm in ('N','C0','C1','C1_y','L1','L_GT'):
            cfg=copy.deepcopy(base)
            cfg.update(method_id='RGBIR-INDEPENDENT-KD-v2-'+arm,method_identity='PROTOCOL-ADAPTED',
                description='Independent task-specific KD; empirical selection, no joint branch',
                arm=arm,source='paired',protocol_status='DRAFT',formal_training_authorized=False,
                classification_coefficient=0.0 if arm in ('N','L1','L_GT') else .1 if arm=='C0' else None,
                localization_coefficient=None if arm in ('L1','L_GT') else 0.0,
                readiness_receipt=None,calibration_receipt=None,canary_acceptance=None,
                geometry_contract=None,d2_receipt=None)
            cfg['classification']=dict(temperature=2.0,teacher_raw_delta_clip=16.0,
                off_target_weight=0.0 if arm=='C1_y' else .25,levels=[0,1],
                loss='bernoulli_relative_kl',normalization='pre_teacher_base_objects')
            cfg['calibration']=dict(seed=20260907,batches=64,min_nonzero_batches=16,
                train_mode=True,reset_parameters_and_buffers_each_batch=True)
            cfg['evaluation_contract']=dict(endpoint='fixed_budget_last_ema',split='val',
                expected_val_images=1469 if dataset=='drone' else 2406)
            result[f'{dataset}_{arm}']=cfg
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    for name,cfg in configurations().items():
        (a.output/(name+'.yaml')).write_text(yaml.safe_dump(cfg,sort_keys=False))
    print('Draft configurations written; no formal authorization or coefficient invented.')


if __name__=='__main__':main()
