"""Only new execution-clock / watchdog behavior; no lease, process or GPU is launched."""
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import run_subset_queue as q

class FakeDispatch:
    def __init__(self,queued=.0,hang=False,fail=False):
        self.queued=queued;self.hang=hang;self.fail=fail;self.killed=[];self.stopped=threading.Event()
    def dump(self,path,value):Path(path).write_text(json.dumps(value))
    def stop_owned_tree(self,pid):self.killed.append(pid);self.stopped.set()
    def run_job(self,job,output):
        time.sleep(self.queued)
        if self.fail:raise RuntimeError('Synthetic admission failure')
        path=output/(job['id']+'_status.json')
        self.dump(path,dict(status='RUNNING',time=time.time(),launch={'pid':123}))
        if self.hang:self.stopped.wait(1.)
        r=dict(status='COMPLETED',exit_code=0,monitor_errors=[])
        self.dump(path,r);return r

class ClockTruths(unittest.TestCase):
    def test_driver_and_excluded_admission(self):
        with patch.object(q.time,'monotonic',return_value=100.):c=q.ExecutionClock()
        with patch.object(q.time,'monotonic',return_value=120.):
            self.assertEqual(c.spent(),20.)
            c.excluded_admission_seconds=15.
            self.assertEqual(c.spent(),5.)
            self.assertEqual(c.remaining(),2695.)
    def exercise(self,d,limit):
        with tempfile.TemporaryDirectory() as directory,patch.object(q,'LIMIT_SECONDS',limit):
            root=Path(directory);c=q.ExecutionClock();original=d.dump
            try:return q.execute(d,dict(id='fake',stage='canary',arm='N'),root,c),c
            finally:
                self.assertEqual(d.dump,original)
                self.assertTrue((root/'fake_timing.json').is_file())
    def test_queued_time_does_not_spend_execution_limit(self):
        d=FakeDispatch(queued=.08)
        r,c=self.exercise(d,.04)
        self.assertEqual(r['status'],'COMPLETED');self.assertFalse(d.killed)
        self.assertGreaterEqual(c.excluded_admission_seconds,.07)
        self.assertLess(c.spent(),.04)
    def test_watchdog_only_targets_declared_own_pid(self):
        d=FakeDispatch(hang=True)
        with self.assertRaisesRegex(ValueError,'remaining execution budget'):self.exercise(d,.01)
        self.assertEqual(d.killed,[123])
    def test_no_launch_failure_is_not_execution(self):
        d=FakeDispatch(queued=.02,fail=True)
        with self.assertRaisesRegex(RuntimeError,'Synthetic admission'):self.exercise(d,2700.)
        self.assertFalse(d.killed)

if __name__=='__main__':unittest.main()
