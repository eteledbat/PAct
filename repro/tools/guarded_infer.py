"""Run original inference with load checks and optional read-only joint tracing."""
import json,os,pathlib,runpy,sys
import torch
ROOT=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
original=torch.nn.Module.load_state_dict

def checked(self,state_dict,*args,**kwargs):
 result=original(self,state_dict,*args,**kwargs)
 required=set(dict(self.named_parameters()))
 missing_parameters=sorted(required.intersection(result.missing_keys))
 record={'class':type(self).__name__,'loaded_keys':len(state_dict),'missing_keys':list(result.missing_keys),'unexpected_keys':list(result.unexpected_keys),'missing_parameters':missing_parameters}
 if os.environ.get('PACT_LOAD_AUDIT'):
  with open(os.environ['PACT_LOAD_AUDIT'],'a') as f:f.write(json.dumps(record)+'\n')
 print('[checkpoint-audit]',json.dumps(record),flush=True)
 if missing_parameters or result.unexpected_keys:raise RuntimeError('Official checkpoint/model parameter mismatch; refusing unverified random parameters')
 return result

torch.nn.Module.load_state_dict=checked
# Read-only diagnostics capture raw head outputs before axis snapping and bbox projection.
if os.environ.get('PACT_JOINT_TRACE'):
 import copy,numpy as np
 from modules.pact.pipelines import PActPipeline
 trace=pathlib.Path(os.environ['PACT_JOINT_TRACE']);trace.mkdir(parents=True,exist_ok=True)
 original_decode=PActPipeline.decode_articulation
 def decode(self,arti,*args,**kwargs):
  if arti is not None:
   for i in range(arti.shape[0]):np.save(trace/f'raw_head_{i:02d}.npy',arti[i].feats.detach().cpu().numpy())
  result=original_decode(self,arti,*args,**kwargs)
  if result is not None:(trace/'decoded_before_bbox.json').write_text(json.dumps(copy.deepcopy(result['info']),indent=2))
  return result
 PActPipeline.decode_articulation=decode
 for method,label in [('animate_with_articulation','animation'),('export_arti_objects','export')]:
  old=getattr(PActPipeline,method)
  def capture(self,reps,infos,*args,_old=old,_label=label,**kwargs):
   (trace/f'{_label}_before.json').write_text(json.dumps(copy.deepcopy(infos),indent=2))
   result=_old(self,reps,infos,*args,**kwargs)
   (trace/f'{_label}_after.json').write_text(json.dumps(copy.deepcopy(infos),indent=2))
   return result
  setattr(PActPipeline,method,capture)
sys.argv[0]='infer_imgs.py'
runpy.run_path(str(ROOT/'infer_imgs.py'),run_name='__main__')
