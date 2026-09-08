"""Adapt the already-used sole-lease queue into a separate L3 identity."""
from pathlib import Path
root=Path(__file__).parent
source=root.parent/'2026-09-08_probe_快速方向筛选/newentry/release/run_direction_queue.py'
s=source.read_text(encoding='utf-8')
s=s.replace("DATASETS = ('llvip', 'drone')","DATASETS = ('llvip',)")
s=s.replace("ARMS = {'llvip': ('N', 'L2-box', 'L2-GT'), 'drone': ('N', 'C1', 'C2', 'F-rel')}","ARMS = {'llvip': ('N', 'L3-DFL', 'L3-GT')}")
s=s.replace('L2-box','L3-DFL').replace('L2-GT','L3-GT').replace("startswith('L2')","startswith('L3')")
s=s.replace('DIRECTION_','OBJECT_DFL_').replace('EXPLORATORY_FIXED8_BNFROZEN','OBJECT_DFL_FIXED8_BNFROZEN')
s=s.replace('calibrate_direction.py','calibrate_object_dfl.py').replace('train_direction.py','train_object_dfl.py').replace('evaluate_direction.py','evaluate_object_dfl.py')
s=s.replace("id='direction_'","id='object_dfl_'")
needle="        report['blocked'] = blocked\n"
replacement=needle+"        if blocked:\n            report.update(status='BLOCKED_CALIBRATION',seconds=time.time()-started,new_hash_computed=False)\n            write_new(queue / (dataset + '_completion.json'),report)\n            return report\n"
assert s.count(needle)==1;s=s.replace(needle,replacement)
s=s.replace('LLVIP then Drone direction screen on the existing global lease only.','LLVIP L3 three-arm screen on the original global lease only.')
dest=root/'trainer_release';dest.mkdir(exist_ok=True)
with (dest/'run_object_dfl_queue.py').open('x',encoding='utf-8') as f:f.write(s)
test=(source.parent/'test_direction_queue_cpu.py').read_text(encoding='utf-8')
test=test.replace('import run_direction_queue as q','import run_object_dfl_queue as q')
test=test.replace('DIRECTION_','OBJECT_DFL_').replace('EXPLORATORY_FIXED8_BNFROZEN','OBJECT_DFL_FIXED8_BNFROZEN')
test=test.replace('L2-box','L3-DFL').replace('L2-GT','L3-GT')
test=test.replace("self.assertEqual(result['status'],'COMPLETED_WITH_BLOCKED_ARMS',result)","self.assertEqual(result['status'],'BLOCKED_CALIBRATION',result)")
test=test.replace("self.assertEqual(calls,[('calibration','N'),('canary','N'),('train','N'),('eval','N')])","self.assertEqual(calls,[('calibration','N')])")
test=test.replace("        out,cfgs,execute,calls=self.fixture('drone')\n        self.assertEqual(q.process_dataset(HERE,out,'drone',cfgs,execute)['status'],'COMPLETED')\n",'')
test=test.replace('test_invalid_candidate_is_skipped_and_N_continues','test_invalid_calibration_blocks_all_three_arms')
with (dest/'test_object_dfl_queue_cpu.py').open('x',encoding='utf-8') as f:f.write(test)
print('New L3 queue source prepared; no GPU launch')
