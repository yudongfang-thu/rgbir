"""Pure CPU checks; imports no GPU library and never reads/writes remote state."""
import copy,unittest
import revise_profiled_reservations as m

def fixture():
    state={'schema':'jstars-project-resource-leases-v1','policy':{'untouched':True},'leases':{}}
    processes={};gp=[];gpus={};progress={}
    for t in m.TARGETS:
        state['leases'][t['lease_id']]={'job_id':t['job_id'],'job_pid':t['guard_pid'],'owner_pid':t['guard_pid'],
            'gpus':[t['gpu']],'kind':'train','formal_train':True,'expected_vram_mib':10000,'expected_rss_mib':49152,
            'observed_cuda_pids':[t['cuda_pid']],'peak_cuda_pid_counts':{str(t['gpu']):1},
            'per_gpu_peak_vram_mib':{str(t['gpu']):7632},'peak_rss_mib':28898,'admission':{'keep':'original'}}
        processes[t['guard_pid']]={'ppid':1,'rss_mib':20,'cmd':f'python project_resource_guard.py run --job-id {t["job_id"]} -- {t["run"]} '}
        processes[t['cuda_pid']]={'ppid':t['guard_pid'],'rss_mib':28000,'cmd':f'python train_object_evidence.py --output {t["run"]} --arm {t["arm"]} --seed {t["seed"]} '}
        gp.append({'gpu':t['gpu'],'pid':t['cuda_pid'],'used_mib':7632});gpus[t['gpu']]={'memory_free_mib':16432}
        progress[t['run']]={'status':'running','arm':t['arm'],'epoch':10,'optimizer_updates':4000}
    state['leases']['unrelated']={'sentinel':'preserve this exactly'}
    return dict(state=state,processes=processes,gpu_processes=gp,gpus=gpus,
                profile={'returncode':0,'samples':[{'cuda_pids':[975949,99999999]}]},
                summary={'status':'passed','returncode':0,'run':str(m.PROFILE_RUN),'existing_run':m.TARGETS[0]['run'],
                         'minimum_observed_free_mib':10111,'cuda_peak_process_count':2,'samples_with_two_cuda_pids':30,'canary_check':{'status':'passed'}},
                receipt={'optimizer_updates':24,'arm':'paired_random','resources':{'gpu_ids':[2]},'official_test_accessed':False},progress=progress)

class Tests(unittest.TestCase):
    def assert_reject(self,data,needle):
        with self.assertRaisesRegex(ValueError,needle):m.validate_revision(**data)
    def test_valid_four_scalars_only_input_immutable(self):
        d=fixture();saved=copy.deepcopy(d);after,records=m.validate_revision(**d)
        self.assertEqual(d,saved);self.assertEqual(len(records),2)
        for t in m.TARGETS:
            self.assertEqual(after['leases'][t['lease_id']]['expected_vram_mib'],8300)
            self.assertEqual(after['leases'][t['lease_id']]['expected_rss_mib'],32768)
        self.assertEqual(after['leases']['unrelated'],d['state']['leases']['unrelated']);self.assertEqual(after['policy'],d['state']['policy'])
    def test_profile_running_pid_rejected(self):
        d=fixture();d['processes'][99999999]={'ppid':1,'rss_mib':1,'cmd':'training'};self.assert_reject(d,'profile CUDA PID still alive')
    def test_profile_failure_rejected(self):
        d=fixture();d['profile']['returncode']=1;self.assert_reject(d,'not completed/passed')
    def test_inadequate_vram_margin_rejected(self):
        d=fixture();d['state']['leases'][m.TARGETS[0]['lease_id']]['per_gpu_peak_vram_mib']['2']=7789;self.assert_reject(d,'VRAM headroom')
    def test_boundary_vram_margin_accepted(self):
        d=fixture();d['state']['leases'][m.TARGETS[0]['lease_id']]['per_gpu_peak_vram_mib']['2']=7788;m.validate_revision(**d)
    def test_inadequate_rss_margin_rejected(self):
        d=fixture();d['state']['leases'][m.TARGETS[0]['lease_id']]['peak_rss_mib']=31745;self.assert_reject(d,'RSS headroom')
    def test_wrong_pid_identity_rejected(self):
        d=fixture();d['state']['leases'][m.TARGETS[0]['lease_id']]['job_pid']=2;self.assert_reject(d,'PID/job identity')
    def test_second_cuda_process_rejected(self):
        d=fixture();d['gpu_processes'].append({'gpu':2,'pid':1234,'used_mib':100});self.assert_reject(d,'no longer single')
    def test_full_training_disappeared_rejected(self):
        d=fixture();del d['processes'][975949];self.assert_reject(d,'PID disappeared')
    def test_already_revised_rejected(self):
        d=fixture();d['state']['leases'][m.TARGETS[0]['lease_id']]['expected_vram_mib']=8300;self.assert_reject(d,'already revised')
    def test_dataloader_rss_counted(self):
        d=fixture();d['processes'][123456]={'ppid':975949,'rss_mib':4000,'cmd':'worker'};self.assert_reject(d,'RSS headroom')
    def test_seed_command_mismatch(self):
        d=fixture();d['processes'][975949]['cmd']=d['processes'][975949]['cmd'].replace('--seed 42','--seed 123');self.assert_reject(d,'arm/seed mismatch')
    def test_profile_lease_not_released(self):
        d=fixture();d['state']['leases']['profile']={'job_id':'oev1_concurrent_profile_s0'};self.assert_reject(d,'has not been released')

if __name__=='__main__':unittest.main(verbosity=2)
