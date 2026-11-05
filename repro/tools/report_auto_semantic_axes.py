"""Post-recognition audit; manual labels are read only here for comparison."""
import argparse
import json
import pathlib
import shutil
import hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from auto_part_semantics import sha256,write_json
from collision_refinement_common import load_asset,transforms


def label_from_name(name):
    name=name.lower()
    for key in ['handle','drawer','door','wheel','knob','shelf','tray']:
        if key in name:return key
    if 'shutter' in name:return 'door'
    if 'body' in name or 'base' in name:return 'base'
    return 'unknown'


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--reference-root',type=pathlib.Path,required=True);p.add_argument('--pilot-root',type=pathlib.Path);p.add_argument('--previous-auto-root',type=pathlib.Path);a=p.parse_args();root=a.root.resolve();base=a.reference_root.resolve()
    read=lambda path:json.loads(path.read_text());protocol=read(root/'protocol.json');assert read(root/'controller_status.json')['state']=='axes_complete'
    seal=read(root/'RECOGNITION_SEAL.json')
    for rel,h in seal['files'].items():assert sha256(root/rel)==h
    for name,h in read(root/'source_hashes.json').items():assert sha256(root/'source_snapshot'/name)==h
    for path,h in read(root/'previous_preservation.json').items():assert sha256(path)==h
    original_protocol=read(base/'protocol.json');records={};all_rows=[];pilot_rows=[]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    for case in protocol['cases']:
        out=root/case;inputs=read(out/'inputs.json');recognized=read(out/'semantics.json')['parts'];requests=read(out/'axis_requests.json')['parts'];exports=read(out/'axis_only_export_manifest.json');native=load_asset(inputs['source_run']);native_nodes={n['id']:n for n in native['nodes']};newobj=read(pathlib.Path(exports['object']))
        assert sha256(native['object_path'])==exports['original_object_sha256']
        assert sha256(exports['object'])==exports['export_sha256']
        for g in exports['geometry_files']:assert sha256(g['path'])==g['sha256']
        oldtr=transforms(native['nodes'],0.);newtr=transforms(newobj['diffuse_tree'],0.)
        assert set(oldtr)==set(newtr)
        for key in oldtr:assert np.allclose(oldtr[key],newtr[key],atol=1e-12),'Initial reference pose changed'
        # Evaluator-only access to earlier human semantic annotation.
        names=read(base/case/'semantic_labels.json')['owner_names'];mapping=original_protocol['label_to_owner'][case]
        byrequest={r['instance_id']:r for r in requests};rows=[]
        for part in recognized:
            owner=part['instance_id'];expected=label_from_name(names[str(mapping[str(owner)])]);source=inputs['input_label_to_native_node'][str(owner)];native_label=native_nodes[source]['name'];request=byrequest[owner]
            row=dict(case=case,instance_id=owner,automatic_semantic=part['semantic'],manual_reference_semantic=expected,semantic_matches_reference=part['semantic']==expected,native_index_semantic=native_label,native_index_matches_reference=native_label==expected,motion_role=part['motion_role'],semantic_confidence=part['semantic_confidence'],motion_confidence=part['motion_confidence'],axis_family=request['family'],axis_applied=request['applied'],joint_changed=request['applied'] and request['selected_joint']!=request['original_joint'],reason=request['reason'],correspondence_verified=bool(request.get('correspondence',{}).get('verified')))
            rows.append(row);all_rows.append(row)
            if a.pilot_root:
                pilot=read(a.pilot_root/case/'semantics.json')['parts'];old=next(x for x in pilot if x['instance_id']==owner);pilot_rows.append(dict(case=case,instance_id=owner,prediction=old['semantic'],matches_reference=old['semantic']==expected))
        photo=Image.open(out/'reference.png').convert('RGB').resize((600,600));mask=Image.open(out/'numbered_masks.png').convert('RGB').resize((600,600));canvas=Image.new('RGB',(1200,650+32*len(rows)),'white');canvas.paste(photo,(0,35));canvas.paste(mask,(600,35));draw=ImageDraw.Draw(canvas);draw.text((8,8),case+' | original RGB',font=font,fill='black');draw.text((608,8),'Original instance mask IDs',font=font,fill='black')
        for i,row in enumerate(rows):
            text=f"ID {row['instance_id']}: {row['automatic_semantic']} ({row['semantic_confidence']}) | {row['motion_role']} | axis: {row['axis_family']} / "+('APPLIED' if row['axis_applied'] else 'RETAIN NATIVE')
            draw.text((8,646+32*i),text,font=font,fill='black')
        canvas.save(out/'automatic_semantics.png');records[case]=dict(parts=rows,geometry_unchanged=True,reference_pose_unchanged=True)
    total=len(all_rows);summary=dict(instance_count=total,case_count=len(protocol['cases']),semantic_matches=sum(r['semantic_matches_reference'] for r in all_rows),semantic_reference_agreement=sum(r['semantic_matches_reference'] for r in all_rows)/total,unknown_count=sum(r['automatic_semantic']=='unknown' for r in all_rows),applied_requests=sum(r['axis_applied'] for r in all_rows),modified_joints=sum(r['joint_changed'] for r in all_rows),verified_correspondences=sum(r['correspondence_verified'] for r in all_rows),native_index_reference_agreement=sum(r['native_index_matches_reference'] for r in all_rows)/total)
    if pilot_rows:summary['pilot_semantic_reference_agreement']=sum(r['matches_reference'] for r in pilot_rows)/len(pilot_rows)
    if a.previous_auto_root:summary['previous_auto_semantic_reference_agreement']=read(a.previous_auto_root/'results.json')['summary']['semantic_reference_agreement']
    write_json(root/'results.json',dict(summary=summary,cases=records,pilot=pilot_rows,evaluation_policy='Compared against existing human reference semantic annotations after recognition seal; not a blind benchmark. Native index labels can also reflect incorrect generated ownership, not just classification. No GT joint axis used to tune rules.'))
    lines=['# 自动部件语义，只用于运动轴','',f"已完成 {len(protocol['cases'])} 个 case、{total} 个参考实例的本地识别与轴规则分发。原生 PAct 几何、权重和原参考姿态保留。",'',f"模型：`{protocol['model']}`；固定 revision：`{protocol['revision']}`。无需图像API，本地模型一次加载后处理整批。原 PAct 环境保留，新 Transformers 安装在独立的 `repro/envs/auto_semantics`。",'',
    '输入是原始RGB与现有实例mask。提示词不包含case编号、既有人工语义、GT关节或数值轴。VLM输出类别、运动解释、父实例、定性置信度及可见证据。原mask是已有oracle实例分割，本轮不是自动分割实验。','',
    '**语义识别与运动类型分开。** 识别出门不会直接强制旋转；开合方式或连接边不明时保留原预测并记录候选。抽屉使用规范坐标+Z抽拉先验；门/抽屉上的固定拉手跟随父部件；活动提手还需验证几何支撑。轮子、旋钮、架子、托盘目前只提供候选族，未补齐专门轴求解器。','',
    '**参考实例到生成部件还需要验证。** 输入mask的排序对应只是一条索引关系，不保证生成几何正确。程序用原始RGB/轮廓的既有配准，检查生成表面投影是否支持该实例；不一致时拒绝改轴。本轮没有通过人工配方拆分、重组或修补几何来提高通过率，也没有自动解决所有错误部件归属。','',
    '配准复用只读取原RGB/轮廓阶段的尺度、屏幕平移、相机和坐标变换，没有使用旧的语义/几何拟合参数。类别与运动规则由新识别结果选择，未读取旧case配方。对应关系通过类型/轴无关的投影匹配选择；静态主体接触几何包含自动识别的主体与原PAct的base假设。','',
    f"与既有人工语义注释的一致率：**{summary['semantic_matches']}/{total} = {100*summary['semantic_reference_agreement']:.1f}%**；未知 {summary['unknown_count']} 个。轴请求接受 {summary['applied_requests']} 个，其中实际关节参数改变 {summary['modified_joints']} 个；投影对应通过 {summary['verified_correspondences']}/{total}。",'',
    '这些人工注释只由本报告评价器在识别封存后读取，不能视为独立盲测。模型自报high/medium/low并非校准概率，仍可能高置信度出错。','']
    if pilot_rows:lines+= [f"第一轮Qwen2.5-VL-3B出现过度预测drawer，原始结果保留在同级auto_semantic_axes_20261029_v1；一致率为{100*summary['pilot_semantic_reference_agreement']:.1f}%。随后调整目标图的呈现方式并更换模型，因此本轮属于开发验证。",'']
    if protocol.get('semantics_sealed_before_motion_questions'):
        lines+=['本轮采用两阶段识别：先只回答类别问题，并在CATEGORY_SEAL.json封存所有类别；然后单独询问门/提手的运动与父关系。运动回答不能覆盖已经封存的类别。目标像素裁剪放大到独立图像，避免小部件在白色拼图中占比过小。','']
    if a.previous_auto_root:lines+=[f"上一轮Qwen3-VL-4B开发结果为{100*summary['previous_auto_semantic_reference_agreement']:.1f}%；其输出与导出均保留在{a.previous_auto_root.name}，未覆盖。",'']
    if 'ENTIRE segmented assembly' in protocol.get('semantic_prompt',''):lines+=['类别判定以整个mask覆盖的部件为单位：门板及附带拉手应识别为门，整个容器/框架应识别为主体，独立提手区域才标为handle。这个粒度定义对所有case统一使用，不包含按case指定的标签。','']
    if protocol.get('category_predictions_reused_from'):lines+=['本轮复用了上一轮已经封存的原始视觉类别回答，没有重新询问或人工修改类别。仅按通用词表将window/window panel/shutter归入PAct的door；此前被严格解析器拒绝的三条window回答因此恢复为有效类别。只对这些新有效类别补做运动询问，审计在[vocabulary_canonicalization.json](vocabulary_canonicalization.json)。','']
    lines+=['| Case | 自动语义一致数 | 改变关节数 | 图片 | 自动标签与证据 | 轴选择审计 |','|---|---:|---:|---|---|---|']
    for case,record in records.items():
        rows=record['parts'];lines.append(f"| {case} | {sum(x['semantic_matches_reference'] for x in rows)}/{len(rows)} | {sum(x['joint_changed'] for x in rows)} | [图片]({case}/automatic_semantics.png) | [JSON]({case}/semantics.json) | [JSON]({case}/axis_requests.json) |")
    lines+=['','每个case的 `axis_only_export/object.json` 引用原始几何文件，只包含自动接受的关节修改；被拒绝的候选保持原模型运动。所有原始几何哈希和q0变换均已核验。没有重训PAct，也没有运行几何优化。','',
    '复现命令保存在[commands.json](commands.json)。识别与轴适配器代码在[source_snapshot](source_snapshot/)，原始模型响应保存在每个case的raw文本中。12项测试覆盖类别别名、未知门型、固定拉手父关系、活动提手支撑要求、低置信度、父关系成环、下翻运动方向、生成节点错序与共享部件拒绝。','']
    (root/'REPORT.md').write_text('\n'.join(lines))
    for name in ['report_auto_semantic_axes.py']:shutil.copyfile(pathlib.Path(__file__).parent/name,root/'source_snapshot'/name)
    write_json(root/'FINAL_SEAL.json',dict(state='complete',summary=summary,recognition_seal_sha256=sha256(root/'RECOGNITION_SEAL.json'),results_sha256=sha256(root/'results.json'),report_sha256=sha256(root/'REPORT.md'),source_geometry_unchanged=True,original_reference_pose_unchanged=True,previous_outputs_preserved=True,manual_semantics_used_only_for_post_seal_comparison=True))
    (pathlib.Path(__file__).resolve().parents[2]/'repro/current_auto_semantic_axes.txt').write_text(str(root)+'\n');print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
