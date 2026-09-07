"""Bind only root-validated second-campaign LLVIP exports; no hashes or model calls."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/'llvip_full_eval/remote_completed_attempt2'
def main():
    manifest=[]
    for model,modality in [('N42','visible'),('T42','infrared')]:
        p=ROOT/(model+'_full_attempt1')
        s=json.loads((p/'summary.json').read_text(encoding='utf-8'))
        assert s['status']=='completed' and s['images']==2406 and s['gt_count']==7879
        assert s['canary'] is False and s['new_protocol_N'] is False
        manifest.append(dict(name='LLVIP_'+model,path=str(p/'capture/objects.jsonl.gz'),
            contract_path=str(p/'capture_contract.json'),metric_path=str(p/'capture_metrics.json'),
            receipt_path=str(p/'summary.json'),population_path=str(p/'population.json'),
            identity_path=str(p/'model_identity.json'),class_names=['person'],expected_gt=7879,
            dataset='llvip',modality=modality,seed=42,scope='historical_baseline_not_new_protocol_N'))
    (HERE/'llvip_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Two complete LLVIP endpoint bindings prepared')
if __name__=='__main__':main()
