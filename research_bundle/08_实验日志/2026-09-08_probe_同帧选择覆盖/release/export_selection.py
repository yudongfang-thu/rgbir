"""Same-forward full-GT states and exact original C0/C1 selection gates.

No model forward, training, loss backward, threshold change, hash or file I/O.
The caller supplies raw S/T/R, the original selected-only API and verified IDs.
"""
from dataclasses import asdict
import math
import torch
from torch.nn import functional as F
from raw_object_state import spatial_assign, state

SCOPE='SAME_FORWARD_SELECTION_COVERAGE'


def require(value,message):
    if not value:raise ValueError(message)


def identity_rows(contract,key,indices):
    rows=contract[key]
    require(len(rows)==len(indices),'Stable identity count differs: '+key)
    for i,(row,bi) in enumerate(zip(rows,indices.tolist())):
        require(row['global_gt_row']==i and row['image_index']==bi,'Stable identity order differs: '+key)
        require(isinstance(row.get('stable_gt_id'),str) and row['stable_gt_id'],'Missing stable GT identity')
        require(type(row.get('source_gt_row')) is int and row['source_gt_row']>=0,'Invalid source GT row')
        require(isinstance(row.get('image_path'),str) and row['image_path'],'Missing source image identity')
    return rows


def check_selection_equal(a,b):
    for key in ('matched_object_ids','base_object_ids'):
        require(getattr(a,key)==getattr(b,key),'Thin/full object identities differ: '+key)
    for key in ('valid_levels','selected','eligible','quality','labels','base_to_matched','matched_to_base','selected_matched_indices'):
        require(torch.equal(getattr(a,key),getattr(b,key)),'Thin/full exact selection differs: '+key)
    require(a.c0_stats==b.c0_stats,'Thin/full original C0 statistics differ')


def _all_gates(adapter,student,teacher,reference,batch,strides,cfg,selection_seed):
    """Re-execute original scalar helper operations to expose dropped nonbase rows.

    Every matched/base identity, mask, q and count is checked against the actual
    full selector by export_selection; this never supplies selection decisions.
    """
    o=adapter.original
    centers,sv,ranges,size=o._layout(student,cfg,strides)
    B,nc,_=student['scores'].shape
    rgb=o._labels(batch,B,nc,student['scores'].device,size)
    ir=o._labels(adapter._teacher_batch(batch),B,nc,student['scores'].device,size)
    decoded=[o._decode_boxes(raw,centers,sv) for raw in (student,teacher,reference)]
    scores=[raw['scores'].detach().float() for raw in (student,teacher,reference)]
    probs=[v.sigmoid() for v in scores]
    ids=[];common_rows=[];ref_rows=[];teacher_rows=[];es_rows=[];et_rows=[];er_rows=[];regions=[]
    for bi in range(B):
        rg=torch.nonzero(rgb[0]==bi,as_tuple=False).flatten();ig=torch.nonzero(ir[0]==bi,as_tuple=False).flatten()
        rb,rc,tb,tc=rgb[2][rg],rgb[1][rg],ir[2][ig],ir[1][ig]
        ri,ti=o._match_objects(rb,rc,tb,tc,cfg.match_iou)
        if not len(ri):continue
        es,vs=o._evidence(scores[0][bi],rc[ri],rb[ri],rb,centers,ranges,cfg)
        et,vt=o._evidence(scores[1][bi],tc[ti],tb[ti],tb,centers,ranges,cfg)
        er,vr=o._evidence(scores[2][bi],rc[ri],rb[ri],rb,centers,ranges,cfg)
        common=vs&vt&vr;denom=common.sum(1).clamp_min(1)
        ref=o._has_candidate(decoded[2][bi],probs[2][bi],rb[ri],cfg.reference_conf,cfg.reference_iou)
        teacher_ok=o._has_candidate(decoded[1][bi],probs[1][bi],tb[ti],cfg.teacher_conf,cfg.teacher_iou,tc[ti])
        first=len(ids)
        ids.extend((int(bi),int(r),int(t)) for r,t in zip(rg[ri].tolist(),ig[ti].tolist()))
        common_rows.append(common);ref_rows.append(ref);teacher_rows.append(teacher_ok)
        es_rows.append((es*common).sum(1)/denom);et_rows.append((et*common).sum(1)/denom);er_rows.append((er*common).sum(1)/denom)
        per_match=[[] for _ in ri]
        for li,level in enumerate(cfg.levels):
            start,end=ranges[level]
            fg,bg,_=adapter.region_masks(rb[ri],rb,centers[start:end],cfg)
            tfg,tbg,_=adapter.region_masks(tb[ti],tb,centers[start:end],cfg)
            for j in range(len(ri)):
                per_match[j].append(dict(level=int(level),rgb_fg=int(fg[j].sum()),rgb_bg=int(bg[j].sum()),
                    ir_fg=int(tfg[j].sum()),ir_bg=int(tbg[j].sum()),student_valid=bool(vs[j,li]),
                    teacher_valid=bool(vt[j,li]),reference_valid=bool(vr[j,li]),common_valid=bool(common[j,li])))
        regions.extend(per_match)
    device=student['scores'].device
    if ids:
        common=torch.cat(common_rows);ref=torch.cat(ref_rows);correct=torch.cat(teacher_rows)
        es=torch.cat(es_rows);et=torch.cat(et_rows);er=torch.cat(er_rows)
        q=(F.softplus(-er)-F.softplus(-et)).clamp_min(0).detach()
    else:
        common=torch.empty((0,len(cfg.levels)),device=device,dtype=torch.bool)
        ref=correct=torch.empty(0,device=device,dtype=torch.bool)
        es=et=er=q=torch.empty(0,device=device)
    base=common.any(1)&ref;eligible=base&correct&(q>0)
    chosen=o._choose(q,eligible,base,'paired',cfg.rho,selection_seed)
    selected=torch.zeros(len(ids),dtype=torch.bool,device=device);selected[chosen]=True
    counts=dict(rgb_gt_count=len(rgb[0]),teacher_gt_count=len(ir[0]),common_count=len(ids),
        valid_region_count=int(common.any(1).sum()),reference_candidate_count=int(ref.sum()),
        base_count=int(base.sum()),teacher_correct_base_count=int((base&correct).sum()),
        eligible_count=int(eligible.sum()),selected_count=len(chosen),normalizer=max(1,int(base.sum())))
    return dict(ids=tuple(ids),common=common,ref=ref,teacher=correct,base=base,eligible=eligible,selected=selected,
        chosen=chosen,q=q,es=es,et=et,er=er,regions=regions,counts=counts,rgb=rgb,ir=ir,decoded=decoded,probs=probs,size=size)


@torch.no_grad()
def export_selection(batch,student_raw,teacher_raw,reference_raw,*,api,evidence_config,strides,
                     selection_seed,identity_contract,batch_index=1):
    require(identity_contract.get('status')=='STABLE_GT_IDENTITY_VERIFIED','Stable GT mapping must be verified')
    require(isinstance(identity_contract.get('frame_id'),str) and identity_contract['frame_id'],'Missing forward/frame identity')
    adapter=api.adapter;cfg=adapter.evidence_config(evidence_config)
    payload=api.build(student_raw,teacher_raw,reference_raw,batch,strides=strides,config=cfg,
                      selection_seed=selection_seed,full_diagnostics=True)
    full=payload.diagnostics
    require(full is not None,'Real full diagnostics required; placeholders cannot describe coverage')
    check_selection_equal(payload.learning,full)
    full.assert_source(student_raw,teacher_raw,reference_raw,batch)
    g=_all_gates(adapter,student_raw,teacher_raw,reference_raw,batch,strides,cfg,selection_seed)
    require(g['ids']==full.matched_object_ids,'Reconstructed matched order differs from actual selector')
    for name,value in g['counts'].items():require(full.c0_stats[name]==value,'Actual gate count differs: '+name)
    keep=torch.nonzero(g['base'],as_tuple=False).flatten()
    require(torch.equal(keep,full.base_to_matched),'Actual base mapping differs')
    for value,actual,label in ((g['common'][keep],full.valid_levels,'common'),(g['q'][keep],full.quality,'quality'),
        (g['eligible'][keep],full.eligible,'eligible'),(g['selected'][keep],full.selected,'selected'),
        (g['chosen'],full.selected_matched_indices,'selected rank')):
        require(torch.equal(value,actual),'Exact actual selector mismatch: '+label)
    rgb_idx,rgb_cls,rgb_boxes=g['rgb'];ir_idx,ir_cls,ir_boxes=g['ir']
    rgb_ids=identity_rows(identity_contract,'rgb_rows',rgb_idx);ir_ids=identity_rows(identity_contract,'ir_rows',ir_idx)
    B=student_raw['scores'].shape[0]
    require(len(batch['im_file'])==B,'All image frames must be retained')
    by_rgb={ids[1]:mi for mi,ids in enumerate(g['ids'])};by_ir={ids[2]:mi for mi,ids in enumerate(g['ids'])}
    selected_rank={mi:rank+1 for rank,mi in enumerate(g['chosen'].tolist())}
    eligible_order=sorted(torch.nonzero(g['eligible'],as_tuple=False).flatten().tolist(),key=lambda i:-float(g['q'][i]))
    eligible_rank={mi:rank+1 for rank,mi in enumerate(eligible_order)}
    assignments={name:{} for name in ('S','T','R')};frames=[]
    for bi in range(B):
        rg=torch.nonzero(rgb_idx==bi,as_tuple=False).flatten();ig=torch.nonzero(ir_idx==bi,as_tuple=False).flatten()
        frame_id=identity_contract['frame_id']+'::image:'+str(bi)
        frame=dict(frame_id=frame_id,forward_id=identity_contract['frame_id'],image_index=bi,batch_index=batch_index,
            image=str(batch['im_file'][bi]),rgb_gt_count=len(rg),ir_gt_count=len(ig),rgb_global_rows=rg.tolist(),ir_global_rows=ig.tolist())
        if 'pair_info' in batch:
            meta=batch['pair_info'][bi]
            frame['rgb_image']=str(meta['weak_source'])
            frame['ir_image']=str(meta['strong_source'])
            frame['source']=str(meta['weak_source'])
        for name,pos,boxes,indices in (('S',0,rgb_boxes[rg],rg),('T',1,ir_boxes[ig],ig),('R',2,rgb_boxes[rg],rg)):
            conf,classes=g['probs'][pos][bi].max(0)
            assign,ncoarse=spatial_assign(boxes,g['decoded'][pos][bi],conf,classes,original=adapter.original)
            assignments[name].update({int(indices[local]):prediction for local,prediction in assign.items()})
            frame[name+'_coarse_predictions']=ncoarse
        frames.append(frame)
    records=[];ir_records=[]
    gate_fields=('valid_levels','region_valid','reference_candidate','teacher_correct_own','quality_q','quality_positive',
        'base','eligible','selected','matched_row','base_row','selected_rank','eligible_rank','region_counts',
        'student_evidence','teacher_evidence','reference_evidence')
    for row in range(len(rgb_idx)):
        bi=int(rgb_idx[row]);cl=int(rgb_cls[row]);mi=by_rgb.get(row)
        gates=dict(matched=mi is not None,**{key:None for key in gate_fields})
        record=dict(frame_id=frames[bi]['frame_id'],forward_id=identity_contract['frame_id'],batch_index=batch_index,
            image_index=bi,image=str(batch['im_file'][bi]),rgb_global_row=row,rgb_gt_xyxy=rgb_boxes[row].tolist(),
            gt_class=cl,stable_rgb_gt_id=rgb_ids[row]['stable_gt_id'],rgb_identity=rgb_ids[row],
            ir_global_row=None,ir_gt_xyxy=None,stable_ir_gt_id=None,ir_identity=None,pair_iou=None,
            states={name:state(assignments[name].get(row),cl) for name in ('S','R')},gates=gates)
        record['states'].update(T=None,T_to_RGB=None)
        if mi is not None:
            ti=g['ids'][mi][2];teacher_assigned=assignments['T'].get(ti)
            cross_iou=None if teacher_assigned is None else float(adapter.original._iou(rgb_boxes[row:row+1],rgb_boxes.new_tensor([teacher_assigned['box']]))[0,0])
            record.update(ir_global_row=ti,ir_gt_xyxy=ir_boxes[ti].tolist(),stable_ir_gt_id=ir_ids[ti]['stable_gt_id'],
                ir_identity=ir_ids[ti],pair_iou=float(adapter.original._iou(rgb_boxes[row:row+1],ir_boxes[ti:ti+1])[0,0]))
            record['states']['T']=state(teacher_assigned,int(ir_cls[ti]),coordinate_scope='ir_own_gt')
            record['states']['T']['iou_to_rgb_gt']=cross_iou
            record['states']['T_to_RGB']=state(teacher_assigned,cl,evaluation_iou=cross_iou,coordinate_scope='rgb_gt_without_geometric_remap')
            base_row=int(full.matched_to_base[mi])
            gates.update(valid_levels=g['common'][mi].tolist(),region_valid=bool(g['common'][mi].any()),
                reference_candidate=bool(g['ref'][mi]),teacher_correct_own=bool(g['teacher'][mi]),
                quality_q=float(g['q'][mi]),quality_positive=bool(g['q'][mi]>0),base=bool(g['base'][mi]),
                eligible=bool(g['eligible'][mi]),selected=bool(g['selected'][mi]),matched_row=mi,
                base_row=base_row if base_row>=0 else None,selected_rank=selected_rank.get(mi),eligible_rank=eligible_rank.get(mi),
                region_counts=g['regions'][mi],student_evidence=float(g['es'][mi]),teacher_evidence=float(g['et'][mi]),reference_evidence=float(g['er'][mi]))
        records.append(record)
    for row in range(len(ir_idx)):
        bi=int(ir_idx[row]);mi=by_ir.get(row)
        ir_records.append(dict(frame_id=frames[bi]['frame_id'],ir_global_row=row,stable_ir_gt_id=ir_ids[row]['stable_gt_id'],
            ir_identity=ir_ids[row],ir_gt_xyxy=ir_boxes[row].tolist(),gt_class=int(ir_cls[row]),
            rgb_global_row=None if mi is None else g['ids'][mi][1],
            teacher_state=state(assignments['T'].get(row),int(ir_cls[row]),coordinate_scope='ir_own_gt')))
    full.assert_source(student_raw,teacher_raw,reference_raw,batch)
    return dict(scope=SCOPE,forward_id=identity_contract['frame_id'],batch_index=batch_index,records=records,ir_records=ir_records,
        frames=frames,objects_n=len(records),ir_objects_n=len(ir_records),selector_counts=g['counts'],
        original_selector_stats=full.c0_stats,evidence_config=asdict(cfg),selection_seed=selection_seed,
        selector_version=adapter.SELECTION_VERSION,thin_path_used=payload.thin_path_used,
        thin_path_fallback_reason=payload.fallback_reason,full_diagnostics=True,exact_selector_crosscheck=True,
        raw_shapes={name:{'scores':list(raw['scores'].shape),'boxes':list(raw['boxes'].shape),'feats':[list(f.shape) for f in raw['feats']]} for name,raw in (('S',student_raw),('T',teacher_raw),('R',reference_raw))},
        state_definition=dict(coarse_confidence=.05,coarse_iou=.1,correct_confidence=.25,correct_iou=.5,
            assignment='baseline_GT_assisted_one_to_one_spatial_assignment',T_primary='IR own GT',T_secondary='same teacher box against RGB GT without remap',
            state_priority=['no_candidate','low_confidence','class_and_localization','class_only','localization_only','correct']),
        selection_population='whole true batch; rank among original eligible, normalization by original base',
        new_model_forward=False,training=False,backward=False,optimizer_updates=0,new_hash_computed=False,
        AP_oracle_coverage_claim=False,stable_identity_contract_status=identity_contract['status'])
