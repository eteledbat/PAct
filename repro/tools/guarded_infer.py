"""Run original infer_imgs.py with checkpoint compatibility telemetry only."""
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
sys.argv[0]='infer_imgs.py'
runpy.run_path(str(ROOT/'infer_imgs.py'),run_name='__main__')
