"""Read existing small local records; do not run models or compute hashes."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
REPO = ROOT / '03_现行工程/SpaceNet6_OTD_official_reproduction'
RELATIVE = [
    'tools/train_rgbt_cmdistill.py', 'yolo_osssl/rgbt_cmdistill_kd.py',
    'yolo_osssl/rgbt_hnewa_pairing.py', 'tools/eval_rgbt_detector.py',
    'configs/research/rgbt_cmdistill_protocol_drone.yaml',
    'configs/research/rgbt_cmdistill_protocol_llvip.yaml',
    'tests/test_rgbt_cmdistill_kd.py', 'tests/test_train_rgbt_cmdistill.py',
    'tools/train_rgbt_cclkd.py', 'yolo_osssl/rgbt_cclkd_kd.py',
    'configs/research/rgbt_cclkd_protocol_drone.yaml',
    'configs/research/rgbt_cclkd_protocol_llvip.yaml',
]
files = [REPO / name for name in RELATIVE]
for name in [
    '06_历史工程_只读/LADD_public/ladd/code/src/teacher_student_decomposition_kd_hbb/loss.py',
    '06_历史工程_只读/LADD_public/comparison/fgd/README.md',
    '06_历史工程_只读/LADD_public/comparison/ld/README.md',
    '06_历史工程_只读/LADD_public/cclkd_reproduction/code/train_cclkd_online_hbb.py',
    '06_历史工程_只读/LADD_public/cclkd_reproduction/yolov5_sanity/code/train_yolov5_cclkd_full.py',
    '08_实验日志/2026-09-07_train_IndependentKD实施/EXTERNAL_BASELINE_PREPARATION.md',
    '08_实验日志/2026-09-07_train_IndependentKD实施/external_baseline_proposals.json',
    '08_实验日志/2026-09-07_train_IndependentKD实施/external_baseline_availability.json',
]:
    files.append(ROOT / name)
assets = [{'path': str(p), 'exists': p.is_file(),
           'bytes': p.stat().st_size if p.is_file() else None,
           'mtime_ns': p.stat().st_mtime_ns if p.is_file() else None} for p in files]
snap_path = ROOT / '08_实验日志/2026-09-08_audit_项目与94最新全景/historical_inventory/remote_snapshot_20260908.json'
snap = json.loads(snap_path.read_text(encoding='utf-8'))
cmd = []
for path, value in snap['files'].items():
    if 'rgbt_cmdistill_adapted_v2' in path and path.endswith('metrics_record.json'):
        metric = json.loads(value['text'])
        csv = snap['csv_endpoints'][path.rsplit('/', 1)[0] + '/results.csv']
        cmd.append({'source_container': str(snap_path), 'remote_original': path,
                    'source_record_bytes': value['bytes'], 'metric_record': metric,
                    'csv_training_rows': csv['rows'], 'csv_last_epoch': csv['last']['epoch'],
                    'csv_time_seconds': float(csv['last']['time']), 'csv_AP_not_used': True})
cmd.sort(key=lambda row: (row['metric_record']['dataset'], row['metric_record']['seed']))
comp_path = ROOT / '08_实验日志/2026-09-07_audit_RGBIR实施起点/comparator_analysis.json'
comp = json.loads(comp_path.read_text(encoding='utf-8'))
ccroot = ROOT / '08_实验日志/2026-09-06_ops_OEv1优先级与对比实验/cclkd/eval_raw/runs/oev1_comparators_20260906'
cc = []
for seed in [0, 42, 123]:
    path = ccroot / f'cclkd_partial_s{seed}_full_attempt1/evaluation_val.json'
    value = json.loads(path.read_text(encoding='utf-8'))
    cc.append({'source': str(path), 'bytes': path.stat().st_size,
               **{k: value[k] for k in ['status', 'method_id', 'seed', 'endpoint',
                   'evaluated_images', 'metric_units', 'mAP50_95', 'AP50', 'AP75', 'checkpoint']}})
obj = {
    'date': '2026-09-09', 'status': 'PREPARED_NOT_ADMITTED',
    'scope': 'Bounded local asset inspection plus three author repository homepages; no model execution',
    'auditor': '/root/baseline_feature_analysis',
    'independent_review': 'No independent-pass claim for this bounded inventory',
    'recommended_minimum_method': 'CMDistill-corrected PROTOCOL-ADAPTED',
    'reason': 'Existing YOLO11 trainer, declared PCCFD/SLRD/IBCLD, two dataset configs, 22 test functions and six historical E200 endpoints; not author-exact.',
    'assets': assets, 'historical_cmdistill_raw_records': cmd,
    'historical_cclkd_independent_evaluations': cc,
    'prior_accepted_comparator_source': str(comp_path),
    'prior_accepted_drone_summaries': {
        key: {k: v for k, v in comp['methods'][key].items() if k in ['stats_percent', 'minus_historical_native']}
        for key in ['cmdistill', 'historical_native']},
    'method_scope': {
        'CMDistill': 'Paper-reconstructed corrected/protocol-adapted; historical/current source identity is not assumed exact',
        'CCLKD': 'RGBIR history LLD+CCL partial; historical complete source missing; old LADD online code is a separate lineage',
        'FGD': 'Historical FGD-style excludes trainable Global context; current RGBIR full FGD only prepared',
        'LD': 'Historical LD-style uses teacher-quality VLR; current RGBIR author Main+VLR only prepared',
        'BCKD': 'Official repo public; current RGBIR only prepared; BCDL alone is partial'},
    'web_sources': [
        {'url': 'https://github.com/TinyTigerPan/BCKD', 'observed': 'Official repository homepage readable; MMDetection2.28.2 GFL configs'},
        {'url': 'https://github.com/yzd-v/FGD', 'observed': 'Author repository homepage readable'},
        {'url': 'https://github.com/HikariTJU/LD', 'observed': 'Author repository homepage readable'}],
    'negative_search_limit': 'Exact CMDistill DOI+GitHub query returned no results; no verified CMDistill/CCLKD author repo bound in inspected assets. This does not prove no official code exists.',
    'new_hash_computed': False, 'weights_loaded': False, 'gpu_used': False,
    'ssh_used': False, 'AP_recomputed': False, 'historical_source_modified': False,
}
assert len(cmd) == 6 and len(cc) == 3
assert all(row['exists'] for row in assets)
(OUT / 'assets.json').write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'assets': len(assets), 'cmd_endpoints': len(cmd), 'cclkd_endpoints': len(cc)}))
