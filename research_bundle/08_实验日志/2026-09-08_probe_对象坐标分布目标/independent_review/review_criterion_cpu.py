from pathlib import Path
import sys,json,io,contextlib,unittest
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent/'trainer_release'))
import test_object_dfl_wrappers_cpu as test
log=io.StringIO()
with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(test.CriterionTruth))
(HERE/'trainer_criterion_cpu_test_output.txt').write_text(log.getvalue(),encoding='utf-8')
data={'status':'PASS' if result.wasSuccessful() else 'FAIL','tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'reviewer_executed':True,'scope':'Synthetic criterion wiring; no actual training','new_GPU_or_SSH':False,'new_hash_computed':False}
with (HERE/'TRAINER_CRITERION_CPU_TESTS.json').open('x',encoding='utf-8') as f:json.dump(data,f,indent=2)
print(json.dumps(data))
