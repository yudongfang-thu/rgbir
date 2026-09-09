"""Copy this stage's small evidence; no model/data copies and no content hashes."""
import datetime
import json
import pathlib
import re
import shutil
import tarfile

stage = pathlib.Path(__file__).resolve().parent
workspace = stage.parents[1]
repo = pathlib.Path('C:/Users/MSI-PC/rgbir_review_worktrees/evidence-20260906')
destination = repo / 'research_bundle' / stage.parent.name / stage.name
allowed = {'.md', '.py', '.json', '.jsonl', '.yaml', '.yml', '.log', '.txt', '.npz', '.sh', '.exit', '.csv', '.tsv'}

status = {'checked_at': {}, 'servers': {}, 'raw_host_process_lists_published': False}
for server in ('90', '94'):
    data = json.loads((stage / f'health{server}_final.json').read_text(encoding='utf-8-sig'))
    status['checked_at'][server] = data['checked_at']
    rows = []
    for row in data['gpu']['stdout'].splitlines():
        parts = [p.strip() for p in row.split(',')]
        rows.append({'index': int(parts[0]), 'name': parts[2], 'total': parts[3], 'used': parts[4], 'free': parts[5], 'utilization': parts[6]})
    status['servers'][server] = dict(ssh=True, project_root_readable=data['root_readable'], gpus=rows,
                                   project_tmux_session_present=data['tmux'].get('exit_code') == 0,
                                   lease=data['lease'], old_rgbir_training_resumed=False)
    if server == '90':
        status['servers'][server]['no_current_stage_training_process'] = not any(
            token in data['own_processes']['stdout'] for token in ('train_author_canary', 'train_canary_dp3', 'llvip_author_baseline.py'))
status['servers']['94']['functional_check'] = '16 forward/backward iterations and existing LLVIP image decode passed on one dynamically leased GPU; not an eight-card stress test'
common = json.loads((stage/'health94_common_lease_final.json').read_text(encoding='utf-8-sig'))
status['servers']['94']['lease'] = common['state']
status['servers']['94']['lease_file'] = common['lease_file']
status['servers']['94']['lease_checked_at'] = common['checked_at']
(stage/'SERVER_STATUS_FINAL.json').write_text(json.dumps(status, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

pair = json.loads((stage/'llvip_execution/llvip_author_pair_completed.json').read_text(encoding='utf-8'))
failure = json.loads((stage/'llvip_training_execution/llvip_train_canary_attempt2/failure.json').read_text(encoding='utf-8'))
summary = dict(status='STAGE_COMPLETED_WITH_TRAINING_RESOURCE_BLOCKS', identity='PAPER-RECONSTRUCTED',
               new_completed_evaluations={k:dict(metrics=v['metrics'], seconds=v['elapsed_seconds']) for k,v in pair['receipts'].items()},
               independent_numeric_coverage_checks_passed=44,
               training=dict(cft_successful_updates=0, llvip_successful_updates=failure['successful_optimizer_updates'],
                             llvip_amp_skips=failure['amp_skips'], llvip_status=failure['status'], required_updates=24, e200_started=False),
               new_drone_original_protocol_ap=False, new_kd_ap=False,
               scope='Published author checkpoints reproduce revised paper metrics closely; no from-initialization training reproduction or transferable KD gain established.')
(stage/'STAGE_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

files, excluded = [], []
for path in sorted(stage.rglob('*')):
    if not path.is_file(): continue
    relative = path.relative_to(stage)
    if '__pycache__' in path.parts or path.name.startswith('PUBLICATION_') or path.name.startswith('PUBLISH_'):
        continue
    if relative.parts[:2] == ('alternative_original', 'AMFD_source') or relative.as_posix() == 'alternative_original/AMFD_README.md':
        excluded.append(dict(path=relative.as_posix(), reason='Unexecuted third-party source retained locally; publish our review and upstream links'))
        continue
    if (path.name.startswith('health90_') or path.name.startswith('health94_')) and path.suffix == '.json':
        excluded.append(dict(path=relative.as_posix(), reason='Raw host snapshot includes unrelated user processes; publish SERVER_STATUS_FINAL.json instead'))
        continue
    if path.suffix.lower() not in allowed and path.name not in ('LICENSE', 'COPYING'):
        excluded.append(dict(path=relative.as_posix(), reason='Non-evidence media/archive or unsupported extension'))
        continue
    if path.stat().st_size > 2_000_000:
        excluded.append(dict(path=relative.as_posix(), reason='Exceeds small-artifact threshold'))
        continue
    target = destination/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(path,target)
    files.append(dict(path=relative.as_posix(),bytes=path.stat().st_size))

entry = (stage/'ORIGINAL_REPRO_CONTINUATION_20260909.md').read_text(encoding='utf-8')
entry = entry.replace('详细记录、原值、实际脚本、训练失败与复核入口见同目录README及其链接。',
                      f'详细记录、原值、实际脚本、训练失败与复核入口见[完整阶段记录](research_bundle/{stage.parent.name}/{stage.name}/README.md)。')
(repo/'ORIGINAL_REPRO_CONTINUATION_20260909.md').write_text(entry,encoding='utf-8')
banner = '> **2026-09-09 13:06 最新：**[LLVIP作者RGB/IR完整复评与94恢复检查](ORIGINAL_REPRO_CONTINUATION_20260909.md)。新增mAP52.664239/67.014908%，接近修订版52.7/67.0，44项数值/覆盖检查通过。CFT三卡0更新OOM，LLVIP原训练10更新后OOM，均未开E200。94通过实际GPU/数据检查，旧训练未恢复；当前其他用户已占用全部8卡。以下为历史阶段记录。\n\n'
for name in ('README.md','LATEST_RESULTS.md'):
    path=repo/name
    current=path.read_text(encoding='utf-8-sig')
    if not current.startswith(banner): path.write_text(banner+current,encoding='utf-8')
index_target=repo/'research_bundle/08_实验日志/README.md'
shutil.copy2(workspace/'08_实验日志/README.md',index_target)
receipt_target=repo/'research_bundle/99_整理回执/20260909_90原协议继续与94检查.md'
receipt_target.parent.mkdir(parents=True,exist_ok=True)
shutil.copy2(workspace/'99_整理回执/20260909_90原协议继续与94检查.md',receipt_target)

manifest=dict(prepared_at=datetime.datetime.now().astimezone().isoformat(),files=files,excluded=excluded,
              count=len(files),bytes=sum(p['bytes'] for p in files),new_content_hashes=False,
              policies=['No checkpoints, datasets, original image collections, raw third-party Drive HTML or unrelated host process lists',
                        'Original evaluation arrays, predictions, executed source, licenses, receipts and failure attempts preserved'])
(stage/'PUBLICATION_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
shutil.copy2(stage/'PUBLICATION_MANIFEST.json',destination/'PUBLICATION_MANIFEST.json')

# Archive the reviewed public subset for the server's stage_evidence/ directory.
with tarfile.open(stage/'PUBLICATION_stage_evidence.tar.gz','w:gz') as archive:
    for row in files:
        archive.add(stage/row['path'],arcname=row['path'],recursive=False)
    archive.add(stage/'PUBLICATION_MANIFEST.json',arcname='PUBLICATION_MANIFEST.json',recursive=False)
print(json.dumps({'copied_count':len(files),'bytes':manifest['bytes'],'excluded_count':len(excluded),
                  'leases':{s:status['servers'][s]['lease'] for s in ('90','94')},
                  'public_source_licenses':[p['path'] for p in files if p['path'].endswith('LICENSE')]}))
