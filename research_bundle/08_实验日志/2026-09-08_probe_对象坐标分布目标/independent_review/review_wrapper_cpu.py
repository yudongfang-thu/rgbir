from pathlib import Path
import sys,json,io,contextlib,tempfile,unittest
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'trainer_release'))
scratch=HERE/'wrapper_scratch';scratch.mkdir(exist_ok=False);tempfile.tempdir=str(scratch)
import test_object_dfl_wrappers_cpu as test
stream=io.StringIO()
with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(test.Contracts))
(HERE/'trainer_wrapper_cpu_test_output.txt').write_text(stream.getvalue(),encoding='utf-8')
summary={'status':'PASS' if result.wasSuccessful() else 'FAIL','tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'reviewer_executed':True,'new_GPU_or_SSH':False,'new_hash_computed':False}
with (HERE/'TRAINER_WRAPPER_CPU_TESTS.json').open('x',encoding='utf-8') as f:json.dump(summary,f,indent=2)
print(json.dumps(summary))
raise SystemExit(not result.wasSuccessful())
