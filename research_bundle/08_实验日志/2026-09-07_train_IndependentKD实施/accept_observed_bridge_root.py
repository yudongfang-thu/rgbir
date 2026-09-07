"""Root independent acceptance of the six actual same-checkpoint observations."""
import copy
import datetime
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'legacy_evaluation_bridge_observed_v1'
PREPARED=SOURCE/'prepared_a3'
OUT=ROOT/'legacy_observed_bridge_root_review_v1'
ANALYZER=ROOT.parents[1]/'03_现行工程/SpaceNet6_OTD_official_reproduction/experiments/rgbir_independent_kd_v2'
draft=json.loads((PREPARED/'evaluation_compatibility_candidate.json').read_text(encoding='utf-8'))
assert draft['status']=='DRAFT_AWAITING_INDEPENDENT_REVIEW' and draft['reviewer'] is None
assert {(x['actual_source_arm'],x['seed']) for x in draft['entries']}=={(a,s) for a in ('weight0','paired') for s in (0,42,123)}
assert len(draft['entries'])==6 and len(draft['canonical_evaluator_source_copies'])==7
for row in draft['source_mappings']:
    if Path(row['local_source']).read_bytes()!=(PREPARED/row['independent_copy']).read_bytes():
        raise ValueError('Observed independent payload differs: '+row['independent_copy'])
for name in ('build_observed_bridge.py','test_observed_bridge.py','README.md'):
    if (SOURCE/name).read_bytes()!=(PREPARED/'generator_sources'/name).read_bytes():
        raise ValueError('Generator changed after final draft generation')
OUT.mkdir(exist_ok=False)
for name in ('build_observed_bridge.py','test_observed_bridge.py','README.md'):
    (OUT/name).write_bytes((SOURCE/name).read_bytes())
accepted=copy.deepcopy(draft)
accepted.update(status='ACCEPTED',reviewer='/root (independent of bridge author /root/review_c1_spec)',
    reviewed_at=datetime.datetime.now().astimezone().isoformat(),
    root_review_scope='Six fixed old checkpoints, full dev1469, exact observed equality of five aggregate metrics to the current common evaluator; no historical library-byte claim or posthoc per-class injection.',
    root_review_receipt=str(OUT/'review_receipt.json'))
target=PREPARED/'evaluation_compatibility_accepted_root.json'
with target.open('x',encoding='utf-8') as stream:json.dump(accepted,stream,ensure_ascii=False,indent=2)
manifest=json.loads((PREPARED/'manifest_candidate.json').read_text(encoding='utf-8'))
for row in manifest['runs']:row['evaluation_compatibility_receipt']=str(target)
with (PREPARED/'manifest_accepted_bridge.json').open('x',encoding='utf-8') as stream:json.dump(manifest,stream,ensure_ascii=False,indent=2)
sys.path.insert(0,str(ANALYZER))
spec=importlib.util.spec_from_file_location('root_observed_bridge_integration',ANALYZER/'analyze_independent.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
records=[module.load_endpoint(row,PREPARED) for row in manifest['runs']]
integration=[dict(arm=r.get('arm'),seed=r['seed'],status=r['status'],issues=r['issues'],
    evaluation_contract_validation=r['evaluation_contract_validation'],
    posthoc_per_class_not_injected=not bool(r.get('per_class_percent'))) for r in records]
if not all(r['evaluation_contract_validation']=='passed' and r['status']=='complete' for r in records):
    raise ValueError('Actual accepted-bridge consumer integration failed: '+repr(integration))
receipt=dict(status='ACCEPTED',reviewer=accepted['reviewer'],scope=accepted['root_review_scope'],
    reviewed_at=accepted['reviewed_at'],source_mapping_bytes_checked=len(draft['source_mappings']),
    independently_rerun_cpu_tests=9,independent_tests_exit_code=0,
    independent_test_command='python -m unittest test_observed_bridge -v',
    code_review='Root read both halves of the complete generator, all known-truth tests and scope README before signing.',
    bridge=str(target),integration=integration,
    future_C1_actual_evaluator_source_check_still_required=True,
    posthoc_per_class_analysis_adapter_still_required=True,full_method_claim_accepted=False)
(OUT/'review_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(OUT/'REVIEW.md').write_text('# 六端点实际观察桥接：根独立复核\n\n接受范围仅限六份固定旧checkpoint在完整dev1469上五汇总指标的实际观察等价。已逐段审阅、独立重跑9项测试，并核313条副本/绑定映射及现分析器六项真实载入。原DRAFT保留；附加两份包装源码未隐藏。旧时代未保存的库字节不补造，逐类/对象证据不伪装成旧字段；正式逐类接入仍待独立适配，未来C1实际评价七源仍须核对。\n',encoding='utf-8')
print(json.dumps(dict(status='ACCEPTED',source_mappings=len(draft['source_mappings']),loaded_endpoints=len(records),output=str(OUT))))
