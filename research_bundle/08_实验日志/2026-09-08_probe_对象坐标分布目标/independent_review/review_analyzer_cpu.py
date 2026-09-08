from pathlib import Path
import sys,json,io,contextlib,tempfile,unittest,copy
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ANALYSIS=HERE.parent/'analysis';sys.path.insert(0,str(ANALYSIS))
scratch=HERE/'analyzer_test_scratch';scratch.mkdir(exist_ok=False);tempfile.tempdir=str(scratch)
import analyze_object_dfl as analyzer
import test_analyzer_cpu as author
checks={}
n=author.receipt('N',.25);t=author.receipt('L3-DFL',.249);g=author.receipt('L3-GT',.251)
r=analyzer.summarize({('llvip','N'):n,('llvip','L3-DFL'):t,('llvip','L3-GT'):g})
comp=r['datasets']['llvip']['comparisons']
checks['negative_pp_preserved']=abs(comp[0]['delta_pp']['mAP50_95']-(-.1))<1e-12 and abs(comp[2]['delta_pp']['mAP50_95']-(-.2))<1e-12
checks['all_raw_values_preserved']=r['datasets']['llvip']['arms'][0]['raw_fraction']['mAP50_95']==.25
checks['percent_display_only']=r['datasets']['llvip']['arms'][0]['display_percent']['mAP50_95']==25
checks['single_seed_no_SD_or_auto_expansion']=r['n_seeds']==1 and r['standard_deviation'] is None and r['automatically_extend_matrix'] is False
for field,bad in [('mAP50_95',True),('recall',float('inf')),('seconds',-1),('kd_coefficient',True)]:
    changed=copy.deepcopy(n);changed[field]=bad
    try:analyzer.validate(changed,'llvip','N');checks['invalid_'+field]=False
    except ValueError:checks['invalid_'+field]=True
buf=io.StringIO()
with contextlib.redirect_stdout(buf),contextlib.redirect_stderr(buf):
    author_result=unittest.TextTestRunner(stream=buf,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(author.Truths))
(HERE/'analyzer_cpu_test_output.txt').write_text(buf.getvalue(),encoding='utf-8')
data={'status':'PASS' if all(checks.values()) and author_result.wasSuccessful() else 'FAIL','independent_checks':checks,
      'authored_tests_executed_by_reviewer':author_result.testsRun,'authored_tests_passed':author_result.wasSuccessful(),'actual_AP_read':False,'new_hash_computed':False}
with (HERE/'ANALYZER_CPU_TESTS.json').open('x',encoding='utf-8') as f:json.dump(data,f,indent=2)
print(json.dumps(data,indent=2))
