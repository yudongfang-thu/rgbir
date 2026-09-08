"""Create an explicit separate measurement driver; keep the executed first release untouched."""
from pathlib import Path
import shutil
root=Path(__file__).parent;old=root/'release';new=root/'witness_release';new.mkdir(exist_ok=True)
for name in ('mapping_trace.py','raw_object_state.py','export_selection.py','llvip_N_s42_FT3.yaml'):
    target=new/name
    if target.exists():raise FileExistsError('Do not overwrite a frozen witness source: '+name)
    shutil.copyfile(old/name,target)
text=(old/'run_probe.py').read_text(encoding='utf-8')
text=text.replace("SCOPE='SAME_FORWARD_SELECTION_COVERAGE'","SCOPE='SAME_FORWARD_DETECTOR_WITNESS'")
text=text.replace('from export_selection import export_selection','from export_selection import export_selection\n    from detector_witness import analyze_witness')
needle="selection_seed=cfg['seed']+1,identity_contract=identity)"
if text.count(needle)!=1:raise ValueError('Expected unique exporter call')
text=text.replace(needle,needle+"\n            previous_objects=[json.loads(line) for line in args.previous_objects.read_text(encoding='utf-8').splitlines() if line.strip()]\n            if exported['records']!=previous_objects:raise AssertionError('Old same-batch object states or original selection changed')\n            witnesses=analyze_witness(batch,{'S':student,'T':teacher,'R':reference},\n                {'S':trainer.model,'T':criterion.teacher,'R':criterion.reference},\n                evidence_config=criterion.evidence_cfg,strides=tuple(int(s) for s in trainer.model.stride),existing_export=exported)")
needle="write_new(output/'export_contract.json'"
insert="""for name,rows in [('witness_objects',witnesses['records']),('witness_ir_objects',witnesses['ir_records']),('witness_frames',witnesses['frames'])]:
            with (output/(name+'.jsonl')).open('x',encoding='utf-8') as f:
                for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\\n')
        write_new(output/'witness_contract.json',{k:v for k,v in witnesses.items() if k not in ('records','ir_records','frames')})
        """
if text.count(needle)!=1:raise ValueError('Expected unique export writer')
text=text.replace(needle,insert+needle)
text=text.replace("raw_forward_counts=dict(student=1,teacher=1,reference=1),", "previous_objects_exact=True,previous_objects=stat(args.previous_objects),head_decode_nonstate_caches_may_change=True,\n            raw_forward_counts=dict(student=1,teacher=1,reference=1),")
text=text.replace("'expected-stream','output'", "'expected-stream','previous-objects','output'")
with (new/'run_witness.py').open('x',encoding='utf-8') as f:f.write(text)
queue=(old/'run_queue.py').read_text(encoding='utf-8').replace('run_probe.py','run_witness.py')
queue=queue.replace("id='selection_coverage_'","id='detector_witness_'")
queue=queue.replace("'--config',str(cp),'--expected-stream',str(expected),'--output'", "'--config',str(cp),'--expected-stream',str(expected),'--previous-objects',str(B/'rgbir_selection_coverage_20260908/attempt1/probe/objects.jsonl'),'--output'")
queue=queue.replace('SAME_FORWARD_SELECTION_COVERAGE_COMPLETED','SAME_FORWARD_DETECTOR_WITNESS_COMPLETED').replace('SAME_FORWARD_SELECTION_QUEUE_','SAME_FORWARD_DETECTOR_WITNESS_QUEUE_')
queue=queue.replace("optimizer_updates=0,backward=0,training=0,first_batch_stream_exact=True", "optimizer_updates=0,backward=0,training=0,first_batch_stream_exact=True,previous_objects_exact=True")
with (new/'run_witness_queue.py').open('x',encoding='utf-8') as f:f.write(queue)
print('WITNESS_DRIVER_CREATED_NO_GPU')
