"""Read persistent model download progress without a running Codex session."""
import pathlib,json
root=pathlib.Path(__file__).resolve().parents[2];path=root/'repro/evidence/model-download-status.json'
s=json.loads(path.read_text());print('State:',s['state'])
if 'confirmed_downloaded_bytes' in s:
 done,total=s['confirmed_downloaded_bytes'],s['total_bytes'];print(f'{done/1e9:.2f}/{total/1e9:.2f} GB ({done/total:.1%})')
 if s.get('bytes_per_second'):print(f"Average download: {s['bytes_per_second']/1e6:.2f} MB/s")
 if s.get('eta_seconds') is not None:print(f"Linear ETA: {s['eta_seconds']/60:.0f} minutes (network dependent)")
print('Updated:',s.get('updated_utc','unavailable'))
if s.get('error'):print('Error:',s['error'])
