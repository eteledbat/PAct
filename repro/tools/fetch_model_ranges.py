"""Resume pinned official weights in bounded HTTP ranges; verify full official SHA256.
Existing SDK downloads are retained. Progress counts confirmed bytes, not sparse file size.
"""
import concurrent.futures,datetime,hashlib,json,os,pathlib,shutil,threading,time
import requests
ROOT=pathlib.Path(__file__).resolve().parents[2]
manifest=json.loads((ROOT/'repro/evidence/official-model-manifest.json').read_text())
folder=ROOT/'repro/cache/PAct-official';status=ROOT/'repro/evidence/model-download-status.json'
progress_path=ROOT/'repro/cache/model-range-progress.json'
lock=threading.Lock();local=threading.local();started=time.monotonic();chunk=4*1024*1024
progress=json.loads(progress_path.read_text()) if progress_path.exists() else {'revision':manifest['revision'],'files':{}}
assert progress['revision']==manifest['revision']
verified_bytes=0;resumed_bytes=0;network_bytes=0

def atomic(path,value):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def report(state='downloading',**extra):
 elapsed=time.monotonic()-started;rate=network_bytes/elapsed if elapsed else 0
 record={'state':state,'method':'bounded_HTTP_ranges','revision':manifest['revision'],'total_bytes':manifest['total_bytes'],'confirmed_downloaded_bytes':verified_bytes,'resumed_bytes':resumed_bytes,'newly_downloaded_bytes':network_bytes,'bytes_per_second':rate,'eta_seconds':(manifest['total_bytes']-verified_bytes)/rate if rate else None,'elapsed_seconds':elapsed,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra}
 atomic(status,record)

try:
 tasks=[]
 for f in manifest['files']:
  path=folder/f['path'];path.parent.mkdir(parents=True,exist_ok=True)
  if path.exists():
   assert path.stat().st_size==f['size'],path
   if f['sha256']:assert sha(path)==f['sha256'],path
   verified_bytes+=f['size'];continue
  assert f['sha256'],'Only official hash-verified weight files may be missing'
  partial=path.with_suffix(path.suffix+'.ranged-part')
  rec=progress['files'].get(f['path'])
  if rec is None:
   candidates=[p for p in (folder/'.cache/huggingface/download'/path.parent.name).glob('*.incomplete') if f['sha256'] in p.name and p.stat().st_size<=f['size']]
   src=max(candidates,key=lambda p:p.stat().st_size) if candidates else None
   prefix=src.stat().st_size if src else 0
   if src:shutil.copyfile(src,partial)
   else:partial.touch(exist_ok=False)
   rec={'prefix_bytes':prefix,'completed_starts':[]};progress['files'][f['path']]=rec
   atomic(progress_path,progress)
  assert partial.exists(),partial
  prefix=rec['prefix_bytes'];verified_bytes+=prefix;resumed_bytes+=prefix
  done=set(rec['completed_starts'])
  for lo in range(prefix,f['size'],chunk):
   hi=min(lo+chunk,f['size'])-1
   if lo in done:verified_bytes+=hi-lo+1;resumed_bytes+=hi-lo+1
   else:tasks.append((f,lo,hi))
  print('RESUME',f['path'],'prefix',prefix,'completed_ranges',len(done),flush=True)
 report()
 # Interleave files, so neither decoder stage waits behind the largest weight.
 tasks.sort(key=lambda x:x[1])
 def fetch(task):
  global verified_bytes,network_bytes
  f,lo,hi=task;size=hi-lo+1
  if not hasattr(local,'session'):local.session=requests.Session()
  url=f"https://huggingface.co/{manifest['repo']}/resolve/{manifest['revision']}/{f['path']}?download=true"
  for attempt in range(10):
   try:
    with local.session.get(url,headers={'Range':f'bytes={lo}-{hi}'},timeout=(20,60),stream=True) as r:
     r.raise_for_status()
     expected=f"bytes {lo}-{hi}/{f['size']}"
     if r.status_code!=206 or r.headers.get('Content-Range')!=expected:raise RuntimeError(f'Incorrect range response: {r.status_code} {r.headers.get("Content-Range")}')
     blocks=[];received=0
     for b in r.iter_content(1024*1024):
      received+=len(b)
      if received>size:raise RuntimeError('Response exceeds requested range')
      blocks.append(b)
     if received!=size:raise RuntimeError(f'Short range {received}/{size}')
    partial=(folder/f['path']).with_suffix('.safetensors.ranged-part')
    with partial.open('r+b') as stream:stream.seek(lo);stream.write(b''.join(blocks))
    with lock:
     progress['files'][f['path']]['completed_starts'].append(lo)
     verified_bytes+=size;network_bytes+=size;atomic(progress_path,progress);report()
     print(f'PROGRESS {verified_bytes}/{manifest["total_bytes"]} {100*verified_bytes/manifest["total_bytes"]:.1f}%',flush=True)
    return
   except Exception as e:
    print('RETRY',f['path'],lo,attempt+1,type(e).__name__,str(e)[:160],flush=True)
    if attempt==9:raise
    time.sleep(min(2**attempt,10))
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
  for future in concurrent.futures.as_completed([executor.submit(fetch,t) for t in tasks]):future.result()
 report('verifying')
 for f in manifest['files']:
  path=folder/f['path']
  if not path.exists():
   partial=path.with_suffix('.safetensors.ranged-part')
   assert partial.stat().st_size==f['size'],path
   assert sha(partial)==f['sha256'],f'SHA256 mismatch: {path}'
   partial.replace(path)
  assert path.stat().st_size==f['size'],path
  if f['sha256']:assert sha(path)==f['sha256'],path
 report('verified',path=str(folder),file_count=len(manifest['files']))
 print('ALL OFFICIAL FILES VERIFIED',flush=True)
except BaseException as e:
 report('failed',error=f'{type(e).__name__}: {e}');raise
