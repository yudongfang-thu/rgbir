from pathlib import Path
root=Path(__file__).parent
source=(root/'start_verified_queue.py').read_text(encoding='utf-8')
source=source.replace("B+'/release_v'+version","B+'/witness_release_v'+version")
source=source.replace("B+'/attempt'+attempt","B+'/witness_attempt'+attempt")
source=source.replace("('mapping_cpu.json','selection_cpu.json')","('witness_cpu.json',)")
source=source.replace('run_queue.py','run_witness_queue.py')
source=source.replace('selection_coverage_probe_s42','detector_witness_probe_s42')
source=source.replace("('queue_launch_attempt'+attempt+'.json')","('witness_queue_launch_attempt'+attempt+'.json')")
exec(compile(source,str(root/'start_verified_queue.py'),'exec'))
