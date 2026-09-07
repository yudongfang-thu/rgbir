"""Independent read-only verification of collected LLVIP evaluation attempt2."""
from pathlib import Path, PurePosixPath
import gzip,json

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'remote_completed_attempt2'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def keyed(rows,modality):
    prefix=PurePosixPath('/mnt/dataset/yudongfang/projects/RGBT_campaign/data/processed/llvip/yolo/grouped_v1')/modality/'images/dev'
    result={}
    for row in rows:
        key=str(PurePosixPath(row['image']).relative_to(prefix))
        assert key not in result
        result[key]=row
    return result

def main():
    models={}; pairs={};profiles=[]
    for model,modality in [('N42','visible'),('T42','infrared')]:
        for stage,n,ngt in [('canary',64,279),('full',2406,7879)]:
            p=DATA/(model+'_'+stage+'_attempt1');s=read(p/'summary.json');c=read(p/'capture_contract.json');pop=read(p/'population.json')
            assert s['status']=='completed' and s['images']==n and s['gt_count']==ngt
            assert s['official_test_accessed'] is False and s['new_protocol_N'] is False
            assert c['loader_per_image_labels_exact'] is True
            assert c['expected_gt']==c['loader_gt']==c['captured_gt']==ngt
            assert c['observed_images']==n and len(c['alias_to_canonical'])==n
            assert set(c['actual_loader_roster'])==set(c['roster'])
            assert set(c['alias_to_canonical'].values())==set(c['roster'])
            with gzip.open(p/'capture/objects.jsonl.gz','rt',encoding='utf-8') as f: rows=[json.loads(x) for x in f]
            assert len(rows)==n and sum(len(r['gt_boxes']) for r in rows)==ngt
            assert set(r['image'] for r in rows)==set(c['alias_to_canonical'])
            assert set(pop['evaluated_aliases'])==set(c['alias_to_canonical'])
            assert sum(pop['original_label_counts'])==pop['full_gt']==7879
            assert pop['expected_evaluated_gt']==ngt
            assert s['native_capture_exact'] is (stage=='canary')
            prof=read(DATA/'queue_attempt1'/('llvip_full_'+model+'_'+stage+'_resource_profile.json'))
            assert prof['status']=='COMPLETED' and prof['exit_code']==0 and not prof['monitor_errors']
            assert prof['minimum_free_mib']>=2048
            assert max(v['project_rss_mib'] for v in prof['samples'])<=300*1024
            profiles.append(dict(model=model,stage=stage,minimum_free_mib=prof['minimum_free_mib'],max_project_rss_mib=max(v['project_rss_mib'] for v in prof['samples']),nvml_peak=s['resources']['per_gpu_peak_vram_mib']))
            if stage=='full':
                pairs[model]=keyed(rows,modality);models[model]=s['metrics']
    assert set(pairs['N42'])==set(pairs['T42'])
    manifest=[]
    for key in sorted(pairs['N42']):
        n=pairs['N42'][key];t=pairs['T42'][key]
        for field in ['gt_boxes','gt_classes','canvas_shape','original_shape']:assert n[field]==t[field],(key,field)
        manifest.append(dict(pair_key=key,rgb_image=n['image'],ir_image=t['image'],gt_count=len(n['gt_boxes']),canvas_shape=n['canvas_shape'],original_shape=n['original_shape']))
    with (ROOT/'verified_pair_manifest.jsonl').open('w',encoding='utf-8') as f:
        for row in manifest:f.write(json.dumps(row)+'\n')
    receipt=dict(status='PASS_LIMITED_EVALUATION_SCOPE',valid_attempt=2,invalid_attempt1_retained=True,models=models,
      delta_teacher_minus_reference_pp={k:100*(models['T42'][k]-models['N42'][k]) for k in ['AP50','AP75','mAP50_95','precision','recall']},
      images=2406,gt=7879,pair_key_contract='Exact relative path under registered grouped_v1/{visible,infrared}/images/dev; unique sets and per-image GT/canvas/original shape exact',
      native_capture_equivalence='64 first canonical dev images for each model; full evaluates capture only',
      resources=profiles,physical_registration_proved=False,new_protocol_baseline=False,kd_gain=False,
      scope='Full old baseline dev evaluation and label/pair consistency only; neither cross-modal physical alignment nor train admission')
    (ROOT/'completed_verification_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt))

if __name__=='__main__':main()
