from pathlib import Path
from datetime import datetime,timezone
import sys,json,copy
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;analysis=HERE.parent/'analysis';sys.path.insert(0,str(analysis))
import analyze_object_dfl as analyzer
import test_analyzer_cpu as fixture
checks={}
r=fixture.receipt('N');checks['explicit_method_identity_accepted']=analyzer.validate(r,'llvip','N') is r
for name,value in [('wrong','DIRECTION_FT3_BNFROZEN'),('missing',None)]:
    changed=copy.deepcopy(r)
    if value is None:changed.pop('method_identity')
    else:changed['method_identity']=value
    try:analyzer.validate(changed,'llvip','N');checks[name+'_method_identity_rejected']=False
    except ValueError:checks[name+'_method_identity_rejected']=True
assert all(checks.values())
author=json.loads((analysis/'CPU_attempt2.json').read_text(encoding='utf-8'))
assert (author['status'],author['tests'],author['failures'],author['errors'])==('PASS',6,0,0)
snapshot=HERE/'analyzer_reviewed_source';snapshot.mkdir(exist_ok=False);sources=[]
for name in ('analyze_object_dfl.py','test_analyzer_cpu.py'):
    source=analysis/name;copy_to=snapshot/name;raw=source.read_bytes();copy_to.write_bytes(raw)
    assert raw==source.read_bytes()==copy_to.read_bytes()
    sources.append({'source_path':str(source.resolve()),'snapshot_path':str(copy_to.resolve()),'bytes':len(raw),'direct_byte_identity':True})
data={'date':datetime.now(timezone.utc).isoformat(),'auditor':'/root/object_transport_review','status':'READY_FOR_RECEIPT_READOUT','source_integrity_status':'pass','actual_L3_AP_analyzed':False,
 'scope':'Receipt identity and arithmetic only; actual executed endpoint audit remains required after collection.',
 'checks':{'previous_independent_arithmetic_checks':8,'previous_authored_synthetic_tests_run_by_reviewer':6,'targeted_method_identity_checks':checks,'author_current_source_CPU_attempt2_tests':6},
 'reviewed_source_files':sources,'new_hash_computed':False,'audited_input_hashes':[],
 'supported_behavior':['Explicit new method, dataset/scope/seed/endpoint identities and complete dev 2406/7879 population checks.',
 'Fraction metrics retained; display percent and signed percentage-point differences are separate; per-class macro consistency checked.',
 'All controls must share initialization path/class map/full-dev YAML; DFL/GT coefficients equal.',
 'Missing/failed/conflicting arms retain explicit states and null differences; no implicit zeros or result selection.',
 'Single-seed output has null SD, no significance claim, no best-arm choice, no automatic expansion/E200.'],
 'limitations':['This analyzer reports inter-arm contrasts only. Each arm versus mature initialization requires the separately matched original baseline AP evidence and root report.',
 'Receipt-only checks do not reconstruct actual AP from predictions or verify checkpoint tensors/GT files.',
 'No actual L3 training/evaluation result was available or read in this source review.'],
 'source_correction':'Require method_identity OBJECT_DFL_FT3_BNFROZEN; reject wrong/missing field. Old source and earlier tests preserved by root.',
 'evidence':[str(HERE/'ANALYZER_CPU_TESTS.json'),str(analysis/'CPU_attempt2.json')]}
with (HERE/'ANALYZER_SOURCE_REVIEW.json').open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2)
md='''# L3 receipt analyzer source review

**READY_FOR_RECEIPT_READOUT**, limited to receipt identity and arithmetic. No actual new L3 AP was read or accepted.

The reviewer executed 8 independent arithmetic/invalid-input checks and all 6 authored synthetic cases. The subsequent single-field method-identity correction was directly inspected and independently checked for valid, wrong and missing values; the author's current-source 6 tests also pass. Final two source files are frozen byte for byte under analyzer_reviewed_source/.

The analyzer preserves raw fractions, displays percent, reports signed percentage-point contrasts, verifies complete dev counts and common initialization/data/class identities, and leaves missing/failed/conflicting differences null. It does not report SD, select a best arm or trigger expansion.

It covers inter-arm contrasts only. The frozen mature-initialization comparisons still require separate matched baseline AP provenance in the root report. This source review does not establish that any new endpoint ran, or that reported AP corresponds to checkpoint/prediction contents; those require the subsequent executed-artifact audit.
'''
(HERE/'ANALYZER_SOURCE_REVIEW.md').write_text(md,encoding='utf-8')
print(json.dumps({'status':data['status'],'targeted_checks':checks}))
