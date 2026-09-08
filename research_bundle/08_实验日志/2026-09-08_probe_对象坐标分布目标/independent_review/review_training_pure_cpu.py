"""Independent pure-CPU control checks plus executor queue mock tests; no torch/GPU."""
from pathlib import Path
import importlib, io, json, sys, tempfile, unittest, contextlib
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
attempt=sys.argv[1] if len(sys.argv)>1 else 'attempt1'
suffix='' if attempt=='attempt1' else '_'+attempt
release=HERE.parent/'trainer_release';sys.path.insert(0,str(release))
scratch=HERE/('trainer_test_scratch'+suffix);scratch.mkdir(exist_ok=False);tempfile.tempdir=str(scratch)
import object_dfl_common as common
import calibrate_object_dfl as calibration
import run_object_dfl_queue as queue
import test_object_dfl_queue_cpu as author_tests

checks={}
for arm in ('N','L3-DFL','L3-GT'):
    cfg=common.load_config(release/'configs'/f'llvip_{arm}_s42_FT3.yaml')
    checks['config_'+arm]=cfg['method_identity']=='OBJECT_DFL_FT3_BNFROZEN' and cfg['classification_coefficient']==0
coef,blocked,details=calibration.coefficient_plan({'L3-DFL':[.1,.2,.3,.4], 'L3-GT':[.9]*8})
checks['one_DFL_median_shared_GT']=coef=={'N':0.,'L3-DFL':.25,'L3-GT':.25} and blocked=={}
coef,blocked,_=calibration.coefficient_plan({'L3-DFL':[.1,.2,.3], 'L3-GT':[.9]*8})
checks['less4_blocks_both']=coef['L3-DFL'] is None and coef['L3-GT'] is None and set(blocked)=={'L3-DFL','L3-GT'}
coef,blocked,_=calibration.coefficient_plan({'L3-DFL':[2,3,4,5], 'L3-GT':[]})
checks['upper_cap1']=coef['L3-DFL']==coef['L3-GT']==1
for label,resources,allocated,reserved,expected in [
  ('within',{'per_gpu_peak_vram_mib':{'0':4096},'peak_rss_mib':8192},3500,4000,'PASS'),
  ('vram_exceeds',{'per_gpu_peak_vram_mib':{'0':8192},'peak_rss_mib':8192},8000,8192,'RESOURCE_LIMIT_EXCEEDED'),
  ('rss_exceeds',{'per_gpu_peak_vram_mib':{'0':4096},'peak_rss_mib':32768},3500,4000,'RESOURCE_LIMIT_EXCEEDED')]:
    checks['firstbatch_'+label]=common.calibration_resource_check(resources,allocated,reserved)['status']==expected
for label,resources in [('empty',{'per_gpu_peak_vram_mib':{},'peak_rss_mib':123}),('nan',{'per_gpu_peak_vram_mib':{'0':float('nan')},'peak_rss_mib':123})]:
    try:common.calibration_resource_check(resources,123,123);checks['invalid_resource_'+label]=False
    except ValueError:checks['invalid_resource_'+label]=True
class Module:pass
module=Module();module.check_amp=lambda model:'old'
class Trainer:pass
trainer=Trainer();trainer.model=object();trainer.amp=None
old=module.check_amp
def setup():trainer.amp=module.check_amp(trainer.model)
binding=common.setup_with_validated_amp(trainer,setup,True,True,module)
checks['amp_binding_restores_exact_original']=module.check_amp is old and binding['original_binding_restored'] and trainer.amp is True
try:common.setup_with_validated_amp(trainer,setup,False,True,module);checks['amp_false_rejected']=False
except ValueError:checks['amp_false_rejected']=True
log=io.StringIO()
with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
    authored=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(author_tests.Tests))
(HERE/('trainer_queue_cpu_test_output'+suffix+'.txt')).write_text(log.getvalue(),encoding='utf-8')
result={'status':'PASS' if all(checks.values()) and authored.wasSuccessful() else 'FAIL','independent_checks':checks,
        'independent_check_count':len(checks),'author_queue_tests_executed_by_reviewer':authored.testsRun,'author_queue_tests_passed':authored.wasSuccessful(),
        'new_GPU_or_SSH':False,'new_hash_computed':False,'scope':'Pure CPU source/control checks; tensor-loss numerical verification requires separate torch CPU evidence.'}
with (HERE/('TRAINER_PURE_CPU_TESTS'+suffix+'.json')).open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
raise SystemExit(result['status']!='PASS')
