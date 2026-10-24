"""Download only the pinned official inference pipeline; retain hashes and provenance."""
import pathlib,os,json,time,hashlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'repro/vendor'))
os.environ['HF_HOME']=str(ROOT/'repro/cache/huggingface');os.environ['HF_HUB_DISABLE_XET']='1'
from huggingface_hub import snapshot_download,HfApi
meta=json.loads((ROOT/'repro/evidence/provenance.json').read_text());repo=meta['model_repo'];revision=meta['model_revision'];status=ROOT/'repro/evidence/model-download-status.json';started=time.time()
def save(value):
 tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(status)
try:
 info=HfApi().model_info(repo,revision=revision,files_metadata=True)
 assert info.sha==revision
 files=[{'path':s.rfilename,'size':s.size,'sha256':s.lfs.sha256 if s.lfs else None} for s in info.siblings]
 (ROOT/'repro/evidence/official-model-manifest.json').write_text(json.dumps({'repo':repo,'revision':revision,'files':files,'total_bytes':sum(x['size'] or 0 for x in files)},indent=2))
 save({'state':'downloading','revision':revision,'total_bytes':sum(x['size'] or 0 for x in files)})
 folder=pathlib.Path(snapshot_download(repo,revision=revision,local_dir=str(ROOT/'repro/cache/PAct-official'),max_workers=4))
 for f in files:
  p=folder/f['path'];assert p.stat().st_size==f['size'],p
  if f['sha256']:
   h=hashlib.sha256()
   with p.open('rb') as stream:
    for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
   assert h.hexdigest()==f['sha256'],p
 save({'state':'verified','revision':revision,'path':str(folder),'file_count':len(files),'elapsed_seconds':time.time()-started})
except BaseException as e:
 save({'state':'failed','revision':revision,'error':f'{type(e).__name__}: {e}'});raise
