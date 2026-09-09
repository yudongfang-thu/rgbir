"""Create immutable diagnostic drafts; formal values require observed evidence."""
import argparse
import copy
from pathlib import Path
import yaml

HERE = Path(__file__).resolve().parent
ROOT = '/mnt/dataX/ydf/projects/RGBT_campaign_90'
ARTIFACTS = ROOT + '/artifacts/rgbir_task_conditional_v1_20260907'


def configurations():
    old = yaml.safe_load((HERE/'legacy_oev1/config_drone.yaml').read_text(encoding='utf-8'))
    base = copy.deepcopy(old)
    base.update(method_id='RGBIR-OEV1-PLUS-LOC-v1', description='Original OEv1 plus frozen-reference conditional localization; exploratory',
        protocol_status='DRAFT', geometry_contract=None, calibration_receipt=None, d2_receipt=None,
        canary_acceptance=None, localization_coefficient=None,
        localization={'input_size':640,'levels':[0,1],'temperature':2.0,'match_iou':.5,'pair_iou':.8,
            'reference_conf':.05,'reference_iou':.1,'reliable_conf':.25,'reference_iou_max':.7,
            'teacher_rgb_iou_min':.6,'teacher_ir_iou_min':.5,'localization_margin':.05,
            'support_epsilon':.01,'native_candidate_epsilon':1e-9},
        expected_train_images=17990, expected_val_images=1469,
        formal_training_authorized=False, source_geometry_scope='not_yet_verified')
    llvip = copy.deepcopy(base)
    llvip.update(dataset='llvip', expected_nc=1, student_modality='visible',
        expected_train_images=9619, expected_val_images=2406)
    prep = ROOT+'/artifacts/rgbt_p3_causal_v1/prepared/llvip'
    runs = ROOT+'/runs/rgbt_p3_causal_v1/formal_native/llvip'
    llvip['paths'] = {'student_data_yaml':prep+'/visible.data.yaml',
        'privileged_data_yaml':prep+'/infrared.data.yaml',
        'paired_train_mapping':prep+'/mappings/visible_to_infrared_train.json'}
    llvip.update(teacher=runs+'/infrared_seed42_native_b32a2/weights/last.pt',
        reference=runs+'/visible_seed42_native_b32a2/weights/last.pt')
    return {'drone_draft.yaml':base, 'llvip_draft.yaml':llvip}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    for name,cfg in configurations().items():
        path=a.output/name
        text=yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True)
        if path.exists():
            if path.read_text(encoding='utf-8')!=text:
                raise FileExistsError('Existing config differs; use a new version: '+str(path))
        else:
            path.write_text(text,encoding='utf-8')
        print(path)


if __name__=='__main__':main()
