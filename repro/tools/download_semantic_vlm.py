"""Resumable public weight download, pinned revision and official LFS hashes."""
import concurrent.futures
import hashlib
import json
import os
import pathlib
import time
import requests
import argparse

ROOT=pathlib.Path(__file__).resolve().parents[2]
DEST=ROOT/'repro/cache/Qwen2.5-VL-3B-Instruct'
REV='66285546d2b821cf421d4f5eb2576359d3770cd3'
MODEL_ID='Qwen/Qwen2.5-VL-3B-Instruct'
FILES=[('model-00001-of-00002.safetensors',3982649232,'41a8895c164b4d32bae6b302f4603fcbc1797f32dafa45c7e9bcda23c6755df8'),('model-00002-of-00002.safetensors',3526688744,'365531ff8752420e89dee707b79d021fb2d6e25abafe486f080555a4fe6972e4')]
CHUNK=8*1024*1024

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for data in iter(lambda:f.read(16*1024*1024),b''):h.update(data)
 return h.hexdigest()

def main():
 global MODEL_ID,REV,DEST,FILES
 parser=argparse.ArgumentParser();parser.add_argument('--repo');args=parser.parse_args()
 if args.repo:
  MODEL_ID=args.repo
  response=requests.get('https://huggingface.co/api/models/'+MODEL_ID,params={'blobs':'true'},timeout=30);response.raise_for_status();metadata=response.json();REV=metadata['sha'];DEST=ROOT/'repro/cache'/MODEL_ID.split('/')[-1];DEST.mkdir(exist_ok=True,parents=True)
  FILES=[(f['rfilename'],f['size'],f['lfs']['sha256']) for f in metadata['siblings'] if f['rfilename'].endswith('.safetensors') and '/' not in f['rfilename']]
  (DEST/'pinned_download_metadata.json').write_text(json.dumps(dict(model=MODEL_ID,revision=REV,weight_files=FILES),indent=2)+'\n')
  for f in metadata['siblings']:
   name=f['rfilename']
   if '/' not in name and name.endswith(('.json','.txt')) and name!='pinned_download_metadata.json':
    path=DEST/name
    if not path.exists():
     r=requests.get(f'https://huggingface.co/{MODEL_ID}/resolve/{REV}/{name}',timeout=30);r.raise_for_status();path.write_bytes(r.content)
 DEST.mkdir(exist_ok=True,parents=True);completed=[]
 for name,size,expected in FILES:
  final=DEST/name
  if final.exists() and final.stat().st_size==size and digest(final)==expected:
   completed.append(dict(file=name,size=size,sha256=expected));continue
  part=DEST/(name+'.partial');state=DEST/(name+'.ranges.json');done=set(json.loads(state.read_text())) if state.exists() else set()
  fd=os.open(part,os.O_CREAT|os.O_RDWR,0o644);os.ftruncate(fd,size)
  # Official Qwen mirror; accept bytes only when the pinned HF LFS hash matches.
  url=f'https://modelscope.cn/models/{MODEL_ID}/resolve/master/{name}'
  def get(start):
   end=min(size,start+CHUNK)-1
   for attempt in range(5):
    try:
     r=requests.get(url,params={'download':'true','segment':str(start)},headers={'Range':f'bytes={start}-{end}'},timeout=(20,60))
     if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}' or len(r.content)!=end-start+1:raise ValueError('Range status/size mismatch')
     offset=0
     while offset<len(r.content):offset+=os.pwrite(fd,r.content[offset:],start+offset)
     return start
    except (requests.RequestException,ValueError):
     if attempt==4:raise RuntimeError(f'Range download failed: {name} at {start}') from None
     time.sleep(min(2**attempt,8))
  try:
   tasks=[s for s in range(0,size,CHUNK) if s not in done];last=time.monotonic()
   with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
    for f in concurrent.futures.as_completed([pool.submit(get,s) for s in tasks]):
     done.add(f.result());state.write_text(json.dumps(sorted(done)))
     if time.monotonic()-last>10:
      print(name,len(done),'/',(size+CHUNK-1)//CHUNK,'chunks',flush=True);last=time.monotonic()
  finally:os.close(fd)
  actual=digest(part)
  if actual!=expected:raise ValueError('Official weight SHA256 mismatch: '+name)
  part.replace(final);completed.append(dict(file=name,size=size,sha256=actual));print('VERIFIED',name,flush=True)
 output=dict(model=MODEL_ID,revision=REV,path=str(DEST),download_complete=True,weight_source='Official Qwen ModelScope mirror, verified against pinned Hugging Face LFS hashes',weights=completed)
 destination='auto_semantics_model_'+MODEL_ID.split('/')[-1]+'.json' if args.repo else 'auto_semantics_model.json'
 (ROOT/'repro/evidence'/destination).write_text(json.dumps(output,indent=2)+'\n')
 print('COMPLETE',flush=True)

if __name__=='__main__':main()
