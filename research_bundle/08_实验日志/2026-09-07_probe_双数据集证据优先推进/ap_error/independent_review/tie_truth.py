"""Additional reviewer truth: same-class equal scores preserve source order."""
import importlib.util,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('tide_adapter_tie_review',ROOT/'run_tide_audit.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
answers={}
for order,boxes,expected in [('tp_then_fp',[[0,0,10,10],[40,40,50,50]],100.),('fp_then_tp',[[40,40,50,50],[0,0,10,10]],50.)]:
    row={'image':'tie','gt_boxes':[[0,0,10,10]],'gt_classes':[0],'pred_boxes':boxes,'pred_classes':[0,0],'pred_confidence':[.8,.8]}
    run=module.evaluate([row],['one_class'],.5,False)
    reference=module.independent_ap([row],.5,['one_class'])
    assert run.ap==expected and reference['AP']==expected
    answers[order]={'expected_AP_pp':expected,'official_AP_pp':run.ap,'independent_AP_pp':reference['AP']}
(HERE/'tie_truth_receipt.json').write_text(json.dumps({'status':'PASS','tests':answers},indent=2),encoding='utf-8')
print(json.dumps(answers))
